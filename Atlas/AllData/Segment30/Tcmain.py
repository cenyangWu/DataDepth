import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib  # 处理nii文件
from scipy.ndimage import label, generate_binary_structure

# =========================
# GPU Kernels (numba.cuda)
# =========================

from numba import cuda, float32, uint8
import math
import numpy as np

@cuda.jit(fastmath=True)
def depth_reduce_kernel_u8(masks_u8, original_center, area_center, depths):
    # 一个 block 处理一个 mask；block 内 256 线程做并行归约
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x

    n_masks = masks_u8.shape[0]
    n_voxels = masks_u8.shape[1]
    if bid >= n_masks:
        return

    sm = cuda.shared.array(shape=512, dtype=float32)  # 256*2
    sm_area = sm
    sm_dot  = sm[256:]

    partial_area = 0.0
    partial_dot  = 0.0

    for j in range(tid, n_voxels, bdim):
        v = float32(masks_u8[bid, j])       # u8 -> f32 (0/1)
        c = original_center[j]              # f32
        partial_area += v
        partial_dot  += v * c

    sm_area[tid] = partial_area
    sm_dot[tid]  = partial_dot
    cuda.syncthreads()

    offset = bdim // 2
    while offset > 0:
        if tid < offset:
            sm_area[tid] += sm_area[tid + offset]
            sm_dot[tid]  += sm_dot[tid + offset]
        cuda.syncthreads()
        offset >>= 1  # 每次减半，直到 1 合并所有thread的值


    if tid == 0:
        area_mask = sm_area[0]
        dot_c     = sm_dot[0]
        if area_mask > 0.0 and area_center > 0.0:
            s1 = dot_c / area_mask
            s2 = dot_c / area_center
            depths[bid] = s1 if s1 < s2 else s2
        else:
            depths[bid] = 0.0
def compute_inclusion_scores(masks_data, use_gpu=True, batch_size=128):
    # ===== CPU 回退 =====
    if (not use_gpu) or (not cuda.is_available()):
        print("检测到未启用CUDA或无可用GPU，自动退回CPU计算。")
        # ——你的原CPU实现保留——
        num_samples = masks_data.shape[0]
        masks_flattened = masks_data.reshape(num_samples, -1)
        original_center = np.mean(masks_flattened, axis=0, dtype=np.float32)
        area_masks = np.sum(masks_flattened, axis=1, dtype=np.float32)
        area_center = np.sum(original_center, dtype=np.float32)
        inv_center = 1.0 - original_center
        s1 = 1.0 - np.sum(inv_center * masks_flattened, axis=1, dtype=np.float32) / np.where(area_masks>0, area_masks, 1.0)
        inv_masks = 1.0 - masks_flattened
        s2 = 1.0 - np.sum(inv_masks * original_center, axis=1, dtype=np.float32) / (area_center if area_center>0 else 1.0)
        s1 = np.nan_to_num(s1); s2 = np.nan_to_num(s2)
        depths = np.minimum(s1, s2)
        depths[area_masks == 0] = 0
        if area_center == 0: depths[:] = 0
        return depths.astype(np.float32, copy=False)

    # ===== GPU 路径（双缓冲 + 两流） =====
    print("\n=== 使用GPU加速 (双缓冲 + 传输/计算重叠) ===")
    # 启动总计时（包含预热）
    t_total0 = time.time()

    # 预处理：展平 + u8 压缩 + 计算中心
    t0 = time.time()
    num_samples = masks_data.shape[0]
    masks_flat = masks_data.reshape(num_samples, -1)              # (N, V) float32 assumed
    original_center = masks_flat.mean(axis=0, dtype=np.float32)   # (V,)
    area_center = np.float32(original_center.sum(dtype=np.float32))
    V = masks_flat.shape[1]
    preprocessing_time = time.time() - t0

    # 两条流
    t_setup0 = time.time()
    stream_h2d = cuda.stream()
    stream_k   = cuda.stream()
    # 复用的H2D事件（双缓冲）
    ev_h2d = [cuda.event(timing=False), cuda.event(timing=False)]
    setup_streams_time = time.time() - t_setup0

    # 把中心一次性上卡
    t0 = time.time()
    d_center = cuda.to_device(original_center, stream=stream_h2d)
    stream_h2d.synchronize()
    h2d_time = time.time() - t0  # 仅供参考，不计入后续累计

    # 预热：触发JIT编译与流/事件初始化，避免计入主统计
    t_pw0 = time.time()
    d_masks_pw = cuda.to_device(np.zeros((1, V), dtype=np.float32), stream=stream_h2d)
    d_out_pw = cuda.device_array(1, dtype=np.float32)
    ev_h2d[0].record(stream_h2d)
    ev_h2d[0].wait(stream_k)
    depth_reduce_kernel_u8[1, 256, stream_k](d_masks_pw[:1, :], d_center, area_center, d_out_pw[:1])
    d_out_pw[:1].copy_to_host(cuda.pinned_array(1, dtype=np.float32), stream=stream_k)
    stream_k.synchronize()
    stream_h2d.synchronize()
    prewarm_time = time.time() - t_pw0

    # 结果容器
    all_depths = np.empty(num_samples, dtype=np.float32)

    # 配置
    B = int(batch_size)
    threadsperblock = 256
    # 双份设备缓冲 + 双份 pinned host 缓冲（输入/输出）计时
    t_alloc0 = time.time()
    d_masks = [cuda.device_array((B, V), dtype=np.float32),
               cuda.device_array((B, V), dtype=np.float32)]
    d_out   = [cuda.device_array(B, dtype=np.float32),
               cuda.device_array(B, dtype=np.float32)]
    h_in  = [cuda.pinned_array((B, V), dtype=np.float32),
             cuda.pinned_array((B, V), dtype=np.float32)]
    h_out = [cuda.pinned_array(B, dtype=np.float32),
             cuda.pinned_array(B, dtype=np.float32)]
    alloc_buffers_time = time.time() - t_alloc0

    # 复用的D2H完成事件（双缓冲）
    done_evt   = [None, None]
    done_slice = [None, None]   # (start_idx, length) 记录结果范围

    # 预热后子计时清零（总计时不重置，以包含预热）
    gpu_compute_time = 0.0
    gpu_h2d_time = 0.0
    gpu_d2h_time = 0.0
    h2h_copy_time = 0.0  # Host->Host 拷贝 all_depths <- h_out 的时间

    # 主循环：交替使用缓冲 0 / 1
    # 主循环：交替使用缓冲 0 / 1
    start = 0
    batch_id = 0
    while start < num_samples:
        end = min(start + B, num_samples)
        this_bs = end - start
        buf = batch_id & 1  # 0/1

        # 如该缓冲上一次有未写回结果：等待事件 -> 写回 all_depths
        if done_evt[buf] is not None:
            t_d2h_wait = time.time()
            done_evt[buf].synchronize()
            gpu_d2h_time += (time.time() - t_d2h_wait)
            s0, ln = done_slice[buf]
            t_h2h = time.time()
            all_depths[s0:s0+ln] = h_out[buf][:ln]
            h2h_copy_time += (time.time() - t_h2h)
            done_evt[buf] = None
            done_slice[buf] = None

        # 1) 填充 pinned host 输入缓冲
        t1 = time.time()
        np.copyto(h_in[buf][:this_bs, :], masks_flat[start:end, :])

        # 2) 异步 H2D -> 记录事件（注意：事件在流上 record，后续 event.wait(另一个流)）
        d_masks[buf][:this_bs, :].copy_to_device(h_in[buf][:this_bs, :], stream=stream_h2d)
        ev_h2d[buf].record(stream_h2d)
        gpu_h2d_time += (time.time() - t1)

        # 3) 计算流等待 H2D 完成 → 启动 kernel（grid=this_bs）
        ev_h2d[buf].wait(stream_k)  # <<< 关键：用 event.wait(stream)
        blockspergrid = this_bs
        t2 = time.time()
        depth_reduce_kernel_u8[blockspergrid, threadsperblock, stream_k](
            d_masks[buf][:this_bs, :], d_center, area_center, d_out[buf][:this_bs]
        )

        # 4) 异步 D2H 回到 pinned host 输出缓冲，并在计算流上记录“完成事件”
        d_out[buf][:this_bs].copy_to_host(h_out[buf][:this_bs], stream=stream_k)
        if done_evt[buf] is None:
            done_evt[buf] = cuda.event(timing=False)
        done_evt[buf].record(stream_k)
        done_slice[buf] = (start, this_bs)

        # 统计
        gpu_compute_time += (time.time() - t2)

        # 前进到下一批
        start   = end
        batch_id += 1


    # 循环结束后，冲洗两块缓冲区的剩余结果
    for buf in (0, 1):
        if done_evt[buf] is not None:
            t_d2h_wait = time.time()
            done_evt[buf].synchronize()
            gpu_d2h_time += (time.time() - t_d2h_wait)
            s0, ln = done_slice[buf]
            t_h2h = time.time()
            all_depths[s0:s0+ln] = h_out[buf][:ln]
            h2h_copy_time += (time.time() - t_h2h)
            done_evt[buf] = None
            done_slice[buf] = None

    t_total = time.time() - t_total0
    other_time = t_total - preprocessing_time - prewarm_time - gpu_compute_time - gpu_h2d_time - gpu_d2h_time

    # 进一步拆解 “其他开销”
    other_setup_alloc_copy = setup_streams_time + alloc_buffers_time + h2h_copy_time
    misc_time = other_time - other_setup_alloc_copy
    if misc_time < 0:
        misc_time = 0.0

    print("\n=== GPU计算性能统计（双缓冲） ===")
    print(f" 总计算时间: {t_total:.3f}秒")
    print(f"   数据预处理: {preprocessing_time:.3f}秒 ({preprocessing_time/t_total*100:.1f}%)")
    print(f"   GPU计算时间(含D2H排队): {gpu_compute_time:.3f}秒 ({gpu_compute_time/t_total*100:.1f}%)")
    print(f"   H2D传输时间(累计): {gpu_h2d_time:.3f}秒 ({gpu_h2d_time/t_total*100:.1f}%)")
    print(f"   预热时间: {prewarm_time:.3f}秒 ({prewarm_time/t_total*100:.1f}%)")
    print(f"   D2H等待时间(累计): {gpu_d2h_time:.3f}秒 ({gpu_d2h_time/t_total*100:.1f}%)")
    print(f"   其他开销: {other_time:.3f}秒")
    print("     ├─ 流/事件创建: {:.3f}秒".format(setup_streams_time))
    print("     ├─ 设备/固定内存分配: {:.3f}秒".format(alloc_buffers_time))
    print("     ├─ Host→Host结果拷贝: {:.3f}秒".format(h2h_copy_time))
    print("     └─ 其余杂项: {:.3f}秒".format(misc_time))
    return all_depths


# =========================
# 数据加载/分析/可视化
# =========================

def load_all_masks(mask_dir):
    """
    加载目录中的所有mask文件
    返回: masks_data, mask_names, affine, header
    """
    print(f"\n=== 开始加载Mask文件 ===")
    start_time = time.time()

    nii_files = glob.glob(os.path.join(mask_dir, "*.nii"))
    if not nii_files:
        print(f"在目录 {mask_dir} 中未找到nii文件")
        return None, None, None, None

    print(f"找到 {len(nii_files)} 个nii文件")

    first_img = nib.load(nii_files[0])
    first_data = first_img.get_fdata()
    mask_shape = first_data.shape
    affine = first_img.affine
    header = first_img.header

    print(f"Mask文件形状: {mask_shape}")
    print(f"数据类型: {first_data.dtype}")

    masks_data = np.zeros((len(nii_files),) + mask_shape, dtype=np.float32)
    mask_names = []

    successful_loads = 0
    for i, nii_file in enumerate(nii_files):
        try:
            print(f"正在加载 ({i+1}/{len(nii_files)}): {os.path.basename(nii_file)}")
            img = nib.load(nii_file)
            data = img.get_fdata()
            if data.shape != mask_shape:
                print(f"  警告: 形状不匹配 {data.shape} vs {mask_shape}，跳过")
                continue
            data_f32 = data.astype(np.float32, copy=False)
            masks_data[successful_loads] = data_f32
            mask_names.append(os.path.basename(nii_file))
            successful_loads += 1
            voxel_count = np.count_nonzero(data_f32)
            print(f"  激活体素数: {voxel_count:,}")
        except Exception as e:
            print(f"  加载失败: {str(e)}")
            continue

    if successful_loads < len(nii_files):
        masks_data = masks_data[:successful_loads]
        print(f"实际成功加载: {successful_loads} 个文件")

    end_time = time.time()
    print(f"Mask文件加载完成，耗时: {end_time - start_time:.2f}秒")
    print(f"最终数据形状: {masks_data.shape}")

    return masks_data, mask_names, affine, header


def analyze_depth_ranking(inclusion_scores, mask_names):
    print(f"\n=== 开始深度排序分析 ===")
    sorted_indices = np.argsort(inclusion_scores)[::-1]
    sorted_scores = inclusion_scores[sorted_indices]
    sorted_mask_names = [mask_names[i] for i in sorted_indices]

    print(f"排序结果:")
    print(f"深度最高的5个mask:")
    for i in range(min(5, len(sorted_mask_names))):
        print(f"  {i+1}. {sorted_mask_names[i]} (深度: {sorted_scores[i]:.4f})")

    print(f"\n深度最低的5个mask:")
    for i in range(max(0, len(sorted_mask_names)-5), len(sorted_mask_names)):
        rank = len(sorted_mask_names) - i
        print(f"  倒数{rank}. {sorted_mask_names[i]} (深度: {sorted_scores[i]:.4f})")

    return sorted_indices, sorted_mask_names, sorted_scores


def visualize_depth_analysis(inclusion_scores, mask_names, sorted_indices, output_dir=None):
    print(f"\n=== 开始生成深度分析可视化 ===")

    fig = plt.figure(figsize=(20, 12))

    # 1. 深度分布直方图
    plt.subplot(2, 4, 1)
    plt.hist(inclusion_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Inclusion Score Depth')
    plt.ylabel('Frequency')
    plt.title('Distribution of Inclusion Score Depths')
    plt.grid(True, alpha=0.3)

    # 2. 深度排序图
    plt.subplot(2, 4, 2)
    sorted_scores = inclusion_scores[sorted_indices]
    plt.plot(range(len(sorted_scores)), sorted_scores, 'o-', alpha=0.7)
    plt.xlabel('Rank (sorted by depth)')
    plt.ylabel('Inclusion Score Depth')
    plt.title('Sorted Inclusion Score Depths')
    plt.grid(True, alpha=0.3)

    # 3. 累计深度分布
    plt.subplot(2, 4, 3)
    cumulative_percent = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    plt.plot(sorted_scores, cumulative_percent, 'o-', alpha=0.7)
    plt.xlabel('Inclusion Score Depth')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Depth Distribution')
    plt.grid(True, alpha=0.3)

    # 4. 前后50%对比
    plt.subplot(2, 4, 4)
    mid_point = len(sorted_scores) // 2
    top_50_scores = sorted_scores[:mid_point]
    bottom_50_scores = sorted_scores[mid_point:]
    plt.hist([top_50_scores, bottom_50_scores], bins=15, alpha=0.7, label=['Top 50%', 'Bottom 50%'])
    plt.xlabel('Inclusion Score Depth')
    plt.ylabel('Frequency')
    plt.title('Top 50% vs Bottom 50% Depth Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. 箱线图
    plt.subplot(2, 4, 5)
    plt.boxplot(inclusion_scores, vert=True)
    plt.ylabel('Inclusion Score Depth')
    plt.title('Inclusion Score Depth Box Plot')
    plt.grid(True, alpha=0.3)

    # 6. 统计文本
    plt.subplot(2, 4, 6)
    stats_text = f"""Inclusion Score Depth Statistics:

Count: {len(inclusion_scores)}
Mean: {np.mean(inclusion_scores):.4f}
Std: {np.std(inclusion_scores):.4f}
Min: {np.min(inclusion_scores):.4f}
Max: {np.max(inclusion_scores):.4f}
Median: {np.median(inclusion_scores):.4f}

Percentiles:
25%: {np.percentile(inclusion_scores, 25):.4f}
50%: {np.percentile(inclusion_scores, 50):.4f}
75%: {np.percentile(inclusion_scores, 75):.4f}
90%: {np.percentile(inclusion_scores, 90):.4f}
95%: {np.percentile(inclusion_scores, 95):.4f}"""
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    plt.axis('off')
    plt.title('Statistical Summary')

    # 7. Top10
    plt.subplot(2, 4, 7)
    top_10_indices = sorted_indices[:10]
    top_10_scores = inclusion_scores[top_10_indices]
    top_10_names = [mask_names[i][:15] for i in top_10_indices]
    plt.barh(range(len(top_10_names)), top_10_scores, alpha=0.7)
    plt.yticks(range(len(top_10_names)), top_10_names)
    plt.xlabel('Inclusion Score Depth')
    plt.title('Top 10 Highest Depth Masks')
    plt.grid(True, alpha=0.3)

    # 8. 原顺序散点
    plt.subplot(2, 4, 8)
    plt.plot(range(len(inclusion_scores)), inclusion_scores, 'o', alpha=0.6, markersize=4)
    plt.xlabel('Mask Index (original order)')
    plt.ylabel('Inclusion Score Depth')
    plt.title('Depth Scores by Original Order')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, "depth_analysis_visualization.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"深度分析可视化结果已保存: {fig_path}")

    plt.show()


def create_mask_operations(masks_data, sorted_indices, mask_names, affine, header, output_dir):
    """
    创建前50%深度的并集、前100%并集、前50%交集三个文件
    在计算前先将数据以0.5为阈值转换为binary mask
    
    Parameters:
    masks_data: numpy array, 所有mask数据
    sorted_indices: numpy array, 按深度排序的索引
    mask_names: list, mask文件名
    affine: numpy array, NIfTI仿射变换矩阵
    header: NIfTI header
    output_dir: str, 输出目录路径
    
    Returns:
    results: dict, 包含三个操作的结果
    """
    print(f"\n=== 开始创建Mask操作结果 ===")
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先将数据以0.5为阈值转换为binary mask
    print(f"\n--- 以0.5为阈值进行二值化处理 ---")
    binary_masks_data = (masks_data >= 0.5).astype(np.float32)
    
    # 统计二值化前后的激活体素数量
    original_voxels = np.count_nonzero(masks_data)
    binary_voxels = np.count_nonzero(binary_masks_data)
    print(f"原始数据激活体素总数: {original_voxels:,}")
    print(f"二值化后激活体素总数: {binary_voxels:,}")
    print(f"二值化保留比例: {binary_voxels/original_voxels*100:.2f}%")
    
    # 保存二值化后的个别mask作为参考
    binary_masks_dir = os.path.join(output_dir, "binary_masks")
    os.makedirs(binary_masks_dir, exist_ok=True)
    
    print(f"\n--- 保存前5个二值化mask作为参考 ---")
    for i in range(min(5, len(mask_names))):
        binary_mask = binary_masks_data[i]
        binary_mask_uint8 = binary_mask.astype(np.uint8)
        binary_img = nib.Nifti1Image(binary_mask_uint8, affine, header)
        binary_path = os.path.join(binary_masks_dir, f"binary_{mask_names[i]}")
        nib.save(binary_img, binary_path)
        voxel_count = np.count_nonzero(binary_mask)
        print(f"  保存: {mask_names[i]} -> binary_{mask_names[i]} (激活体素: {voxel_count:,})")
    
    total_count = len(sorted_indices)
    top_50_count = total_count // 2
    
    print(f"\n总mask数量: {total_count}")
    print(f"前50%数量: {top_50_count}")
    
    # 1. 前50%深度的并集 (使用二值化数据)
    print(f"\n--- 计算前50%深度的并集（二值化数据） ---")
    top_50_indices = sorted_indices[:top_50_count]
    top_50_binary_masks = binary_masks_data[top_50_indices]
    union_50 = np.any(top_50_binary_masks > 0, axis=0).astype(np.uint8)
    
    print(f"前50%深度的mask文件:")
    for i, idx in enumerate(top_50_indices):
        print(f"  {i+1:2d}. {mask_names[idx]}")
    
    # 2. 前100%（全部）的并集 (使用二值化数据)
    print(f"\n--- 计算前100%（全部）的并集（二值化数据） ---")
    union_100 = np.any(binary_masks_data > 0, axis=0).astype(np.uint8)
    
    # 3. 前50%深度的交集 (使用二值化数据)
    print(f"\n--- 计算前50%深度的交集（二值化数据） ---")
    intersection_50 = np.all(top_50_binary_masks > 0, axis=0).astype(np.uint8)
    
    # 计算统计信息
    union_50_voxels = np.sum(union_50)
    union_100_voxels = np.sum(union_100)
    intersection_50_voxels = np.sum(intersection_50)
    total_voxels = union_50.size
    
    print(f"\n=== 二值化后的操作结果统计 ===")
    print(f"前50%深度并集激活体素: {union_50_voxels:,} ({union_50_voxels/total_voxels*100:.2f}%)")
    print(f"前100%并集激活体素: {union_100_voxels:,} ({union_100_voxels/total_voxels*100:.2f}%)")
    print(f"前50%深度交集激活体素: {intersection_50_voxels:,} ({intersection_50_voxels/total_voxels*100:.2f}%)")
    
    # 计算交集/并集比例
    if union_50_voxels > 0:
        intersection_union_ratio = intersection_50_voxels / union_50_voxels
        print(f"前50%交集/并集比例: {intersection_union_ratio:.4f} ({intersection_union_ratio*100:.2f}%)")
    
    # 计算覆盖度（重叠度分析）
    if top_50_count > 1:
        overlap_counts = np.sum(top_50_binary_masks > 0, axis=0)
        unique_counts, count_frequencies = np.unique(overlap_counts, return_counts=True)
        print(f"\n=== 前50%mask重叠度分析 ===")
        for count, freq in zip(unique_counts, count_frequencies):
            if count > 0:
                print(f"  被{count}个mask覆盖的体素: {freq:,} ({freq/total_voxels*100:.3f}%)")
    
    # 保存文件
    results = {}
    
    print(f"\n--- 保存二值化后的操作结果 ---")
    
    # 保存前50%深度的并集 (二值化)
    union_50_img = nib.Nifti1Image(union_50, affine, header)
    union_50_path = os.path.join(output_dir, "binary_top50percent_depth_union.nii")
    nib.save(union_50_img, union_50_path)
    results['union_50'] = {'data': union_50, 'path': union_50_path, 'voxels': union_50_voxels}
    print(f"前50%深度并集(二值化)已保存: {union_50_path}")
    
    # 保存前100%并集 (二值化)
    union_100_img = nib.Nifti1Image(union_100, affine, header)
    union_100_path = os.path.join(output_dir, "binary_all_masks_union.nii")
    nib.save(union_100_img, union_100_path)
    results['union_100'] = {'data': union_100, 'path': union_100_path, 'voxels': union_100_voxels}
    print(f"前100%并集(二值化)已保存: {union_100_path}")
    
    # 保存前50%深度交集 (二值化)
    intersection_50_img = nib.Nifti1Image(intersection_50, affine, header)
    intersection_50_path = os.path.join(output_dir, "binary_top50percent_depth_intersection.nii")
    nib.save(intersection_50_img, intersection_50_path)
    results['intersection_50'] = {'data': intersection_50, 'path': intersection_50_path, 'voxels': intersection_50_voxels}
    print(f"前50%深度交集(二值化)已保存: {intersection_50_path}")
    
    # 额外保存二值化处理信息
    binary_info_path = os.path.join(output_dir, "binary_conversion_info.txt")
    with open(binary_info_path, 'w', encoding='utf-8') as f:
        f.write(f"二值化处理信息\n")
        f.write(f"阈值: 0.5\n")
        f.write(f"原始数据激活体素总数: {original_voxels:,}\n")
        f.write(f"二值化后激活体素总数: {binary_voxels:,}\n")
        f.write(f"二值化保留比例: {binary_voxels/original_voxels*100:.2f}%\n")
    print(f"二值化信息已保存: {binary_info_path}")
    
    # 保存详细报告
    report_path = os.path.join(output_dir, "depth_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"包含分数深度分析报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总mask文件数: {total_count}\n")
        f.write(f"前50%数量: {top_50_count}\n\n")
        
        f.write(f"结果统计:\n")
        f.write(f"前50%深度并集激活体素: {union_50_voxels:,} ({union_50_voxels/total_voxels*100:.2f}%)\n")
        f.write(f"前100%并集激活体素: {union_100_voxels:,} ({union_100_voxels/total_voxels*100:.2f}%)\n")
        f.write(f"前50%深度交集激活体素: {intersection_50_voxels:,} ({intersection_50_voxels/total_voxels*100:.2f}%)\n")
        
        if union_50_voxels > 0:
            intersection_union_ratio = intersection_50_voxels / union_50_voxels
            f.write(f"前50%交集/并集比例: {intersection_union_ratio:.4f} ({intersection_union_ratio*100:.2f}%)\n")
        f.write(f"\n注意: 所有结果均基于0.5阈值二值化后的数据\n\n")
        
        f.write(f"前50%深度最高的mask文件列表:\n")
        for i, idx in enumerate(top_50_indices):
            f.write(f"{i+1:2d}. {mask_names[idx]}\n")
        
        f.write(f"\n生成的文件:\n")
        f.write(f"- {os.path.basename(union_50_path)}: 前50%深度并集(二值化)\n")
        f.write(f"- {os.path.basename(union_100_path)}: 前100%并集(二值化)\n")
        f.write(f"- {os.path.basename(intersection_50_path)}: 前50%深度交集(二值化)\n")
        f.write(f"- binary_masks/: 前5个二值化mask示例\n")
        f.write(f"- binary_conversion_info.txt: 二值化处理详细信息\n")
    
    print(f"详细报告已保存: {report_path}")
    
    end_time = time.time()
    print(f"Mask操作完成，耗时: {end_time - start_time:.2f}秒")
    
    return results


def find_largest_connected_component(binary_data, connectivity=26):
    """
    找到3D二值数据中的最大连通域
    
    Parameters:
    binary_data: numpy array, 3D二值数据 (0和1)
    connectivity: int, 连通性定义 (6, 18, 或 26)
    
    Returns:
    largest_component: numpy array, 只包含最大连通域的数据
    stats: dict, 连通域统计信息
    """
    print(f"开始连通域分析...")
    print(f"输入数据形状: {binary_data.shape}")
    print(f"连通性: {connectivity}")
    
    # 确保数据是二值的
    binary_data = (binary_data > 0).astype(np.uint8)
    original_voxels = np.sum(binary_data)
    print(f"原始激活体素数: {original_voxels:,}")
    
    if original_voxels == 0:
        print("警告: 没有激活体素")
        return binary_data, {'component_count': 0, 'largest_size': 0, 'original_size': 0}
    
    # 定义连通性结构
    if connectivity == 6:
        structure = generate_binary_structure(3, 1)  # 6-连通
    elif connectivity == 18:
        structure = generate_binary_structure(3, 2)  # 18-连通
    else:  # 26-连通
        structure = generate_binary_structure(3, 3)  # 26-连通
    
    # 标记连通域
    labeled_array, num_features = label(binary_data, structure=structure)
    print(f"找到 {num_features} 个连通域")
    
    if num_features == 0:
        return binary_data * 0, {'component_count': 0, 'largest_size': 0, 'original_size': original_voxels}
    
    # 计算每个连通域的大小
    component_sizes = []
    for i in range(1, num_features + 1):
        size = np.sum(labeled_array == i)
        component_sizes.append(size)
    
    component_sizes = np.array(component_sizes)
    largest_component_idx = np.argmax(component_sizes) + 1
    largest_size = component_sizes[largest_component_idx - 1]
    
    print(f"最大连通域包含 {largest_size:,} 个体素 ({largest_size/original_voxels*100:.2f}%)")
    
    # 只保留最大连通域
    largest_component = (labeled_array == largest_component_idx).astype(np.uint8)
    
    # 统计信息
    stats = {
        'component_count': num_features,
        'largest_size': largest_size,
        'original_size': original_voxels,
        'component_sizes': sorted(component_sizes, reverse=True),
        'retention_ratio': largest_size / original_voxels if original_voxels > 0 else 0.0
    }
    
    print(f"保留比例: {stats['retention_ratio']*100:.2f}%")
    
    # 显示前3个最大连通域的大小
    print(f"前3个最大连通域大小:")
    for i, size in enumerate(stats['component_sizes'][:3]):
        print(f"  {i+1}. {size:,} 体素 ({size/original_voxels*100:.2f}%)")
    
    return largest_component, stats


def process_largest_connected_components(results, output_dir, affine, header):
    """
    对mask操作结果进行连通域分析，保留最大连通域
    
    Parameters:
    results: dict, mask操作结果
    output_dir: str, 输出目录
    affine: numpy array, NIfTI仿射变换矩阵
    header: NIfTI header
    
    Returns:
    connected_results: dict, 连通域分析结果
    """
    print(f"\n=== 开始连通域分析处理 ===")
    
    # 创建连通域分析输出目录
    connected_dir = os.path.join(output_dir, "largest_components")
    os.makedirs(connected_dir, exist_ok=True)
    
    connected_results = {}
    connectivity = 26  # 使用26连通
    
    # 处理每个mask操作结果
    for key, result in results.items():
        print(f"\n--- 处理 {key} ---")
        
        # 获取原始数据
        mask_data = result['data']
        
        # 进行连通域分析
        largest_component, stats = find_largest_connected_component(mask_data, connectivity)
        
        # 生成输出文件名
        original_path = result['path']
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_filename = f"largest_component_{base_name}.nii"
        output_path = os.path.join(connected_dir, output_filename)
        
        # 保存结果
        try:
            output_img = nib.Nifti1Image(largest_component, affine, header)
            nib.save(output_img, output_path)
            print(f"最大连通域已保存: {output_path}")
            
            # 存储结果
            connected_results[key] = {
                'data': largest_component,
                'path': output_path,
                'voxels': stats['largest_size'],
                'stats': stats
            }
            
        except Exception as e:
            print(f"保存失败: {e}")
    
    # 生成连通域分析报告
    report_path = os.path.join(connected_dir, "connected_components_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("连通域分析处理报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"连通性设置: {connectivity}\n")
        f.write(f"处理文件数: {len(connected_results)}\n\n")
        
        for key, result in connected_results.items():
            stats = result['stats']
            f.write(f"{key}:\n")
            f.write(f"  原始体素数: {stats['original_size']:,}\n")
            f.write(f"  连通域数量: {stats['component_count']}\n")
            f.write(f"  最大连通域: {stats['largest_size']:,} 体素\n")
            f.write(f"  保留比例: {stats['retention_ratio']*100:.2f}%\n")
            f.write(f"  输出文件: {os.path.basename(result['path'])}\n")
            f.write("\n")
    
    print(f"连通域分析报告已保存: {report_path}")
    
    # 打印处理摘要
    print(f"\n=== 连通域分析处理摘要 ===")
    for key, result in connected_results.items():
        stats = result['stats']
        print(f"{key}:")
        print(f"  连通域: {stats['component_count']} -> 1")
        print(f"  体素保留: {stats['retention_ratio']*100:.2f}%")
        print(f"  输出: {os.path.basename(result['path'])}")
    
    return connected_results


def visualize_mask_operations(results, output_dir=None):
    """
    可视化三个mask操作的结果
    
    Parameters:
    results: dict, 包含三个操作的结果
    output_dir: str, 输出目录（可选）
    """
    print(f"\n=== 开始可视化Mask操作结果 ===")
    
    fig = plt.figure(figsize=(20, 15))
    
    operations = [
        ('union_50', 'Top 50% Depth Union', 'Reds'),
        ('union_100', 'All Masks Union', 'Blues'), 
        ('intersection_50', 'Top 50% Depth Intersection', 'Greens')
    ]
    
    for op_idx, (op_key, op_title, colormap) in enumerate(operations):
        if op_key not in results:
            continue
            
        mask_data = results[op_key]['data']
        voxel_count = results[op_key]['voxels']
        
        # 计算中心切片
        depth, height, width = mask_data.shape
        center_z = depth // 2
        center_y = height // 2
        center_x = width // 2
        
        # 轴状切片
        plt.subplot(3, 4, op_idx*4 + 1)
        axial_slice = mask_data[center_z, :, :]
        plt.imshow(axial_slice, cmap=colormap, origin='lower')
        plt.title(f'{op_title}\nAxial (Z={center_z})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        
        # 冠状切片
        plt.subplot(3, 4, op_idx*4 + 2)
        coronal_slice = mask_data[:, center_y, :]
        plt.imshow(coronal_slice, cmap=colormap, origin='lower')
        plt.title(f'Coronal (Y={center_y})')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        
        # 矢状切片
        plt.subplot(3, 4, op_idx*4 + 3)
        sagittal_slice = mask_data[:, :, center_x]
        plt.imshow(sagittal_slice, cmap=colormap, origin='lower')
        plt.title(f'Sagittal (X={center_x})')
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        
        # 统计信息
        plt.subplot(3, 4, op_idx*4 + 4)
        total_voxels = mask_data.size
        activation_ratio = voxel_count / total_voxels * 100
        
        # 每个切片的激活体素数量
        axial_activation = np.sum(mask_data, axis=(1, 2))
        max_axial = np.max(axial_activation)
        
        stats_text = f"""Statistics for {op_title}:

Total Voxels: {total_voxels:,}
Active Voxels: {voxel_count:,}
Activation Ratio: {activation_ratio:.2f}%

Max Axial Activation: {max_axial:,}
Mean Axial Activation: {np.mean(axial_activation):.1f}

Shape: {mask_data.shape}"""
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        plt.axis('off')
        plt.title(f'{op_title} - Statistics')
    
    plt.tight_layout()
    plt.show()


# =========================
# 主流程
# =========================

def main():
    print("=== TC文件深度分析工具（numba.cuda 版本） ===")
    total_start_time = time.time()

    preprocessing_time = 0.0
    gpu_compute_time = 0.0
    data_transfer_time = 0.0
    other_overhead_time = 0.0

    # 设置输出目录路径
    output_dir = r"D:\MyProjects\DataDepthData\TC_DepthAnalysis"

    print(f"数据源: soft_masks中的TC文件")
    print(f"输出目录: {output_dir}")

    # 1. 加载TC文件作为mask数据
    t0 = time.time()
    masks_data, mask_names, affine, header = load_TC_files_as_masks()
    # 确保传入 compute_inclusion_scores 的数据已为 float32
    if masks_data is not None:
        masks_data = masks_data.astype(np.float32, copy=False)
    preprocessing_time += time.time() - t0

    if masks_data is None:
        print("没有成功加载任何TC文件")
        return
    def pick_batch_size(V, bytes_per_voxel=1, safety=0.85, max_batch=1024):
        free_mem, total_mem = cuda.current_context().get_memory_info()
        avail = int(free_mem * safety)
        # 需要双缓冲两份输入 + 一份输出(忽略) + center (≈ V*4)
        overhead = V * 4
        per_batch = V * bytes_per_voxel
        if avail <= overhead + per_batch:
            return 32  # 退而求其次
        max_by_mem = (avail - overhead) // (2 * per_batch)
        B = int(max(1, min(max_by_mem, max_batch)))
        # 对齐到 32 的倍数
        return max(64, (B // 64) * 64)

    # 在进入 GPU 分支前：
    V = np.prod(masks_data.shape[1:], dtype=np.int64)
    #B = pick_batch_size(V, bytes_per_voxel=4)   # 使用 float32，每体素 4 字节
    B = 32   # 固定批次大小为128
    # 然后用 B 作为 batch_size
    print(f"B: {B}")
    # 2. 计算（>50 用批处理）
    use_batch_processing = masks_data.shape[0] > 50
    want_gpu = True  # 想用GPU就设 True
    t0 = time.time()
    inclusion_scores = compute_inclusion_scores(
        masks_data,
        use_gpu=want_gpu,
        batch_size=B
    )
    gpu_compute_time = time.time() - t0  # 粗略记录（包含函数内部预处理/传输）

    # 3. 排序分析
    t0 = time.time()
    sorted_indices, sorted_mask_names, sorted_scores = analyze_depth_ranking(inclusion_scores, mask_names)
    other_overhead_time += time.time() - t0

    # 4. 可视化
    t0 = time.time()
    # 遵循"不另行要求不保存图像"的规则，这里仅显示，不保存
    visualize_depth_analysis(inclusion_scores, mask_names, sorted_indices, output_dir=None)
    other_overhead_time += time.time() - t0

    # 5. 创建三个mask操作结果
    t0 = time.time()
    results = create_mask_operations(masks_data, sorted_indices, mask_names, affine, header, output_dir)
    other_overhead_time += time.time() - t0

    # 6. 连通域分析处理
    if results:
        t0 = time.time()
        connected_results = process_largest_connected_components(results, output_dir, affine, header)
        other_overhead_time += time.time() - t0

    # 7. 可视化mask操作结果
    if results:
        t0 = time.time()
        visualize_mask_operations(results, output_dir=None)
        other_overhead_time += time.time() - t0

    # 8. 保存结果
    print(f"\n=== 保存分析结果 ===")
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "inclusion_scores.npy"), inclusion_scores)
    np.save(os.path.join(output_dir, "sorted_indices.npy"), sorted_indices)
    data_transfer_time = time.time() - t0
    print(f"包含分数深度分析结果已保存到: {output_dir}")

    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    gpu_utilization = (gpu_compute_time / total_runtime) * 100 if total_runtime > 0 else 0.0

    print(f"\n=== 性能统计（总） ===")
    print(f" 总计算时间: {total_runtime:.3f}秒")
    print(f"   数据预处理: {preprocessing_time:.3f}秒 ({preprocessing_time/total_runtime*100:.1f}%)")
    print(f"   GPU计算时间(含函数内部): {gpu_compute_time:.3f}秒 ({gpu_compute_time/total_runtime*100:.1f}%)")
    print(f"   数据传输时间(保存): {data_transfer_time:.3f}秒 ({data_transfer_time/total_runtime*100:.1f}%)")
    print(f"   其他开销: {other_overhead_time:.3f}秒")
    print(f" GPU利用率(粗略): {gpu_utilization:.1f}%")

    print(f"\n=== 分析总结 ===")
    print(f"处理了 {len(mask_names)} 个mask文件")
    print(f"深度范围: {np.min(inclusion_scores):.4f} - {np.max(inclusion_scores):.4f}")
    print(f"平均深度: {np.mean(inclusion_scores):.4f}")
    
    if results:
        print(f"\n生成的原始文件:")
        for key, result in results.items():
            print(f"- {os.path.basename(result['path'])}: {result['voxels']:,} 激活体素")
    
    # 显示连通域分析结果
    if 'connected_results' in locals() and connected_results:
        print(f"\n生成的最大连通域文件:")
        for key, result in connected_results.items():
            stats = result['stats']
            print(f"- {os.path.basename(result['path'])}: {result['voxels']:,} 激活体素 (保留{stats['retention_ratio']*100:.1f}%)")
        print(f"最大连通域文件保存在: {os.path.join(output_dir, 'largest_components')}")
    
    print(f"\n所有结果已保存到: {output_dir}")


def load_TC_files_as_masks():
    """
    加载TC文件并转换为与load_all_masks相同的格式
    返回: masks_data, mask_names, affine, header
    """
    print(f"\n=== 开始加载TC文件作为Mask数据 ===")
    start_time = time.time()
    
    # 首先提取TC文件
    tc_data = load_TC_files_as_arrays()
    
    if not tc_data:
        print("没有找到任何TC文件")
        return None, None, None, None
    
    print(f"找到 {len(tc_data)} 个TC文件")
    
    # 获取第一个文件的信息作为参考
    first_folder = list(tc_data.keys())[0]
    first_data_info = tc_data[first_folder]
    mask_shape = first_data_info['shape']
    affine = first_data_info['affine']
    header = first_data_info['header']
    
    print(f"TC文件形状: {mask_shape}")
    print(f"数据类型: {first_data_info['dtype']}")
    
        # 创建masks_data数组
    masks_data = np.zeros((len(tc_data),) + mask_shape, dtype=np.float32)
    mask_names = []

    successful_loads = 0
    for i, (folder_name, data_info) in enumerate(tc_data.items()):
        try:
            print(f"正在处理 ({i+1}/{len(tc_data)}): {folder_name}")
            data = data_info['data']
            
            if data.shape != mask_shape:
                print(f"  警告: 形状不匹配 {data.shape} vs {mask_shape}，跳过")
                continue
                
            # 转换为float32并确保是二值化数据（0或1）
            data_f32 = data.astype(np.float32, copy=False)
            # 如果数据不是二值化的，进行二值化处理
            
            masks_data[successful_loads] = data_f32
            mask_names.append(f"{folder_name}_TC")
            successful_loads += 1
            voxel_count = np.count_nonzero(data_f32)
            print(f"  激活体素数: {voxel_count:,}")
            
        except Exception as e:
            print(f"  处理失败: {str(e)}")
            continue
    
    if successful_loads < len(tc_data):
        masks_data = masks_data[:successful_loads]
        print(f"实际成功处理: {successful_loads} 个文件")
    
    end_time = time.time()
    print(f"TC文件加载完成，耗时: {end_time - start_time:.2f}秒")
    print(f"最终数据形状: {masks_data.shape}")
    print(f"总激活体素: {np.count_nonzero(masks_data):,}")
    
    return masks_data, mask_names, affine, header

def extract_TC_files_from_soft_masks():
    """
    从D:\MyProjects\DataDepthData\soft_masks中的31个文件夹提取包含TC的nii.gz文件
    """
    source_dir = r"D:\MyProjects\DataDepthData\soft_masks"
    
    print("=== TC文件提取工具 ===")
    print(f"源目录: {source_dir}")
    print("-" * 60)
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误：源目录 {source_dir} 不存在！")
        return []
    
    tc_files = []
    folder_count = 0
    tc_file_count = 0
    
    # 遍历所有文件夹
    for folder_name in sorted(os.listdir(source_dir)):
        folder_path = os.path.join(source_dir, folder_name)
        
        # 确保是文件夹
        if os.path.isdir(folder_path):
            folder_count += 1
            print(f"正在处理文件夹 {folder_count}: {folder_name}")
            
                        # 查找文件夹中包含'TC'的nii.gz文件
            tc_pattern = os.path.join(folder_path, "*TC*.nii.gz")
            tc_files_in_folder = glob.glob(tc_pattern)

            for tc_file in tc_files_in_folder:
                tc_files.append(tc_file)
                tc_file_count += 1
                file_name = os.path.basename(tc_file)
                file_size = os.path.getsize(tc_file) / (1024 * 1024)  # 转换为MB
                print(f"  找到TC文件: {file_name} (大小: {file_size:.2f} MB)")
    
    print("-" * 60)
    print(f"扫描完成！")
    print(f"共扫描了 {folder_count} 个文件夹")
    print(f"找到了 {tc_file_count} 个TC文件")
    
    # 显示所有TC文件的详细信息
    print("\n" + "=" * 80)
    print("所有包含TC的nii.gz文件列表:")
    print("=" * 80)

    for i, file_path in enumerate(tc_files, 1):
        file_name = os.path.basename(file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
        
        print(f"{i:2d}. 文件夹: {folder_name:15s} | 文件: {file_name:25s} | 大小: {file_size:.2f} MB")
    
    print("=" * 80)
    print(f"总计: {len(tc_files)} 个TC文件")

    return tc_files

def load_TC_files_as_arrays(tc_files_list=None):
    """
    将TC文件加载为numpy数组，便于后续处理
    
    Args:
        tc_files_list: TC文件路径列表，如果为None则自动提取
    
    Returns:
        dict: 包含文件夹名作为key，nii数据作为value的字典
    """
    if tc_files_list is None:
        tc_files_list = extract_TC_files_from_soft_masks()

    if not tc_files_list:
        print("没有找到TC文件！")
        return {}
    
    print("\n=== 开始加载TC文件为numpy数组 ===")
    tc_data = {}

    for i, file_path in enumerate(tc_files_list, 1):
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        
        try:
            print(f"加载 {i}/{len(tc_files_list)}: {folder_name}/{file_name}")
            
            # 使用nibabel加载nii.gz文件
            nii_img = nib.load(file_path)
            nii_data = nii_img.get_fdata()
            
            # 存储数据和元信息
            tc_data[folder_name] = {
                'data': nii_data,
                'affine': nii_img.affine,
                'header': nii_img.header,
                'file_path': file_path,
                'shape': nii_data.shape,
                'dtype': nii_data.dtype,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
            }
            
            print(f"  成功加载 - 形状: {nii_data.shape}, 数据类型: {nii_data.dtype}")
            
        except Exception as e:
            print(f"  加载失败: {e}")
    
    print(f"\n成功加载了 {len(tc_data)} 个TC文件")
    print("=" * 60)
    
    # 显示加载摘要
    print("加载摘要:")
    total_memory = 0
    for folder_name, data_info in tc_data.items():
        memory_mb = data_info['data'].nbytes / (1024 * 1024)
        total_memory += memory_mb
        print(f"  {folder_name:15s}: {str(data_info['shape']):20s} | 内存: {memory_mb:.2f} MB")
    
    print(f"总内存使用: {total_memory:.2f} MB")
    print("=" * 60)
    
    return tc_data

if __name__ == "__main__":
    # 使用TC文件进行深度分析
    main()
