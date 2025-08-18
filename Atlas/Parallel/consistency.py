import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib  # 处理nii文件

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
        masks_flattened = masks_data.reshape(num_samples, -1).astype(np.float32, copy=False)
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
    t_total0 = time.time()

    # 预处理：展平 + u8 压缩 + 计算中心
    t0 = time.time()
    num_samples = masks_data.shape[0]
    masks_flat = masks_data.reshape(num_samples, -1)              # (N, V)
    masks_u8   = (masks_flat > 0).astype(np.uint8, copy=False)    # (N, V) u8
    original_center = masks_flat.mean(axis=0, dtype=np.float32)   # (V,)
    area_center = np.float32(original_center.sum(dtype=np.float32))
    V = masks_u8.shape[1]
    preprocessing_time = time.time() - t0

    # 两条流
    stream_h2d = cuda.stream()
    stream_k   = cuda.stream()

    # 把中心一次性上卡
    t0 = time.time()
    d_center = cuda.to_device(original_center, stream=stream_h2d)
    stream_h2d.synchronize()
    h2d_time = time.time() - t0

    # 结果容器
    all_depths = np.empty(num_samples, dtype=np.float32)

    # 配置
    B = int(batch_size)
    threadsperblock = 256
    # 双份设备缓冲
    d_masks = [cuda.device_array((B, V), dtype=np.uint8),
               cuda.device_array((B, V), dtype=np.uint8)]
    d_out   = [cuda.device_array(B, dtype=np.float32),
               cuda.device_array(B, dtype=np.float32)]
    # 双份 pinned host 缓冲（输入/输出）
    h_in  = [cuda.pinned_array((B, V), dtype=np.uint8),
             cuda.pinned_array((B, V), dtype=np.uint8)]
    h_out = [cuda.pinned_array(B, dtype=np.float32),
             cuda.pinned_array(B, dtype=np.float32)]

    # 用事件跟踪每个缓冲的“D2H 完成”
    done_evt   = [None, None]
    done_slice = [None, None]   # (start_idx, length) 记录结果范围

    gpu_compute_time = 0.0
    gpu_h2d_time = h2d_time  # 累加 H2D
    gpu_d2h_time = 0.0

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
            done_evt[buf].synchronize()
            s0, ln = done_slice[buf]
            all_depths[s0:s0+ln] = h_out[buf][:ln]
            done_evt[buf] = None
            done_slice[buf] = None

        # 1) 填充 pinned host 输入缓冲
        t1 = time.time()
        np.copyto(h_in[buf][:this_bs, :], masks_u8[start:end, :])

        # 2) 异步 H2D -> 记录事件（注意：事件在流上 record，后续 event.wait(另一个流)）
        d_masks[buf][:this_bs, :].copy_to_device(h_in[buf][:this_bs, :], stream=stream_h2d)
        ev_h2d = cuda.event(timing=False)
        ev_h2d.record(stream_h2d)
        gpu_h2d_time += (time.time() - t1)

        # 3) 计算流等待 H2D 完成 → 启动 kernel（grid=this_bs）
        ev_h2d.wait(stream_k)  # <<< 关键：用 event.wait(stream)
        blockspergrid = this_bs
        t2 = time.time()
        depth_reduce_kernel_u8[blockspergrid, threadsperblock, stream_k](
            d_masks[buf][:this_bs, :], d_center, area_center, d_out[buf][:this_bs]
        )

        # 4) 异步 D2H 回到 pinned host 输出缓冲，并在计算流上记录“完成事件”
        d_out[buf][:this_bs].copy_to_host(h_out[buf][:this_bs], stream=stream_k)
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
            done_evt[buf].synchronize()
            s0, ln = done_slice[buf]
            all_depths[s0:s0+ln] = h_out[buf][:ln]
            done_evt[buf] = None
            done_slice[buf] = None

    t_total = time.time() - t_total0
    other_time = t_total - preprocessing_time - gpu_compute_time - gpu_h2d_time - gpu_d2h_time

    print("\n=== GPU计算性能统计（双缓冲） ===")
    print(f" 总计算时间: {t_total:.3f}秒")
    print(f"   数据预处理: {preprocessing_time:.3f}秒 ({preprocessing_time/t_total*100:.1f}%)")
    print(f"   GPU计算时间(含D2H排队): {gpu_compute_time:.3f}秒 ({gpu_compute_time/t_total*100:.1f}%)")
    print(f"   H2D传输时间(累计): {gpu_h2d_time:.3f}秒 ({gpu_h2d_time/t_total*100:.1f}%)")
    print(f"   其他开销: {other_time:.3f}秒")
    return all_depths


# =========================
# 数据加载/分析/可视化
# =========================

def compute_inclusion_scores_loop(masks_data):
    """
    方法二：逐样本循环版的包含分数深度计算（CPU循环实现）
    与方法一相同公式，但逐样本遍历，便于与矢量化/GPU结果进行一致性对比。
    """
    print("\n=== 开始计算包含分数深度（方法二：逐样本循环） ===")
    start_time = time.time()

    num_samples = masks_data.shape[0]
    masks_flattened = masks_data.reshape(num_samples, -1).astype(np.float32, copy=False)
    original_center = np.mean(masks_flattened, axis=0, dtype=np.float32)

    inclusion_scores = []
    area_center = float(np.sum(original_center, dtype=np.float32))
    inv_center = (1.0 - original_center).astype(np.float32, copy=False)

    for i in range(num_samples):
        mask = masks_flattened[i]
        area_mask = float(np.sum(mask, dtype=np.float32))

        if area_mask > 0.0 and area_center > 0.0:
            inclusion_score1 = 1.0 - float(np.sum(inv_center * mask, dtype=np.float32)) / area_mask
            inv_mask = (1.0 - mask).astype(np.float32, copy=False)
            inclusion_score2 = 1.0 - float(np.sum(inv_mask * original_center, dtype=np.float32)) / area_center
            depth = min(inclusion_score1, inclusion_score2)
        else:
            depth = 0.0

        inclusion_scores.append(depth)
        if (i + 1) % 10 == 0 or i == num_samples - 1:
            print(f"已处理 {i + 1}/{num_samples} 个mask（方法二）")

    inclusion_scores = np.asarray(inclusion_scores, dtype=np.float32)

    end_time = time.time()
    print(f"包含分数深度（方法二）计算完成，耗时: {end_time - start_time:.2f}秒")
    print("深度分数（方法二）统计:")
    print(f"  最小深度: {np.min(inclusion_scores):.4f}")
    print(f"  最大深度: {np.max(inclusion_scores):.4f}")
    print(f"  平均深度: {np.mean(inclusion_scores):.4f}")
    print(f"  深度标准差: {np.std(inclusion_scores):.4f}")

    return inclusion_scores


def compare_inclusion_methods(scores1, scores2, atol=1e-6, rtol=1e-6, make_plot=True):
    """
    比较两种方法的深度分数是否一致，并可视化对比。
    """
    print("\n=== 开始比较两种深度计算方法的结果一致性 ===")
    scores1 = np.asarray(scores1, dtype=np.float32)
    scores2 = np.asarray(scores2, dtype=np.float32)

    diffs = scores1 - scores2
    abs_diffs = np.abs(diffs)
    max_abs_diff = float(np.max(abs_diffs))
    mean_abs_diff = float(np.mean(abs_diffs))
    tol = atol + rtol * np.abs(scores1)
    num_out_of_tol = int(np.sum(abs_diffs > tol))
    ratio_out = num_out_of_tol / len(scores1) if len(scores1) > 0 else 0.0
    consistent = (num_out_of_tol == 0)

    print(f"样本数: {len(scores1)}")
    print(f"最大绝对差: {max_abs_diff:.6e}")
    print(f"平均绝对差: {mean_abs_diff:.6e}")
    print(f"超出容差数量: {num_out_of_tol} ({ratio_out*100:.2f}%)")
    print(f"是否在容差内一致: {consistent}")

    if make_plot:
        plt.figure(figsize=(12, 5))
        # 左：散点对比
        plt.subplot(1, 2, 1)
        plt.scatter(scores1, scores2, s=10, alpha=0.6)
        mn = float(min(np.min(scores1), np.min(scores2)))
        mx = float(max(np.max(scores1), np.max(scores2)))
        plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='y=x')
        plt.xlabel('Method 1 Depth')
        plt.ylabel('Method 2 Depth')
        plt.title('Depth Comparison: Method 1 vs Method 2')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 右：差值直方图
        plt.subplot(1, 2, 2)
        plt.hist(diffs, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Difference (Method1 - Method2)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Depth Differences')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        'consistent': consistent,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'num_out_of_tol': num_out_of_tol,
        'ratio_out_of_tol': ratio_out,
    }


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
            data_binary = (data > 0).astype(np.float32)
            masks_data[successful_loads] = data_binary
            mask_names.append(os.path.basename(nii_file))
            successful_loads += 1
            voxel_count = np.sum(data_binary)
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


# =========================
# 主流程
# =========================

def main():
    print("=== 包含分数深度分析工具（numba.cuda 版本） ===")
    total_start_time = time.time()

    preprocessing_time = 0.0
    gpu_compute_time = 0.0
    data_transfer_time = 0.0
    other_overhead_time = 0.0

    # 路径自行修改
    mask_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\Allseg\haimaprocessed_binary"
    output_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\Allseg\haimaprocessed_binary\DepthAnalysis"

    print(f"Mask目录: {mask_dir}")
    print(f"输出目录: {output_dir}")

    if not os.path.exists(mask_dir):
        print(f"错误: 目录不存在: {mask_dir}")
        return

    # 1. 加载
    t0 = time.time()
    masks_data, mask_names, affine, header = load_all_masks(mask_dir)
    preprocessing_time += time.time() - t0

    if masks_data is None:
        print("没有成功加载任何mask文件")
        return
    def pick_batch_size(V, bytes_per_voxel=1, safety=0.6, max_batch=256):
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
        return max(32, (B // 32) * 32)

    # 在进入 GPU 分支前：
    V = np.prod(masks_data.shape[1:], dtype=np.int64)
    B = pick_batch_size(V, bytes_per_voxel=1)   # 我们传的是 uint8，所以 1 字节
    # 然后用 B 作为 batch_size

    # 2. 计算（>50 用批处理）
    use_batch_processing = masks_data.shape[0] > 50
    want_gpu = True  # 想用GPU就设 True
    t0 = time.time()
    # 方法一：GPU/矢量化路径
    scores_method1 = compute_inclusion_scores(
        masks_data,
        use_gpu=want_gpu,
        batch_size=B
    )
    gpu_compute_time = time.time() - t0  # 粗略记录（包含函数内部预处理/传输）

    # 方法二：逐样本循环路径（CPU）
    scores_method2 = compute_inclusion_scores_loop(masks_data)

    # 对比两种方法结果
    _cmp = compare_inclusion_methods(scores_method1, scores_method2, atol=1e-6, rtol=1e-6, make_plot=True)
    inclusion_scores = scores_method1

    # 3. 排序分析
    t0 = time.time()
    sorted_indices, sorted_mask_names, sorted_scores = analyze_depth_ranking(inclusion_scores, mask_names)
    other_overhead_time += time.time() - t0

    # 4. 可视化
    t0 = time.time()
    # 遵循不保存图片的要求，这里仅展示，不进行保存
    visualize_depth_analysis(inclusion_scores, mask_names, sorted_indices, output_dir=None)
    other_overhead_time += time.time() - t0

    # 5. 保存结果
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


if __name__ == "__main__":
    main()
