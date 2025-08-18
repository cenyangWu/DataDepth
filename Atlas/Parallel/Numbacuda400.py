import numpy as np
import os
import glob
import time
import math
from numba import cuda
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib  # 处理nii文件

# =========================
# GPU Kernels (numba.cuda)
# =========================

from numba import cuda, float32, uint8
import math
import numpy as np

# 每个 block 处理 1 个 mask；block 内 256 线程做并行归约
# 只计算 area_mask 与 dot_center 两个量
@cuda.jit(fastmath=True)
def depth_reduce_kernel_u8(masks_u8, original_center, area_center, depths):
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x

    n_masks = masks_u8.shape[0]
    n_voxels = masks_u8.shape[1]
    if bid >= n_masks:
        return

    # 共享内存（两段）：前半存 area 的部分和，后半存 dot 的部分和
    sm = cuda.shared.array(shape=512, dtype=float32)  # 256*2
    sm_area = sm
    sm_dot  = sm[256:]

    # 条件：threadsperblock = 256
    partial_area = 0.0
    partial_dot  = 0.0

    # 并行遍历本行的体素，步长 = 线程数
    for j in range(tid, n_voxels, bdim):
        v = float32(masks_u8[bid, j])          # 0/1 -> float
        c = original_center[j]                 # float32
        partial_area += v
        partial_dot  += v * c

    sm_area[tid] = partial_area
    sm_dot[tid]  = partial_dot
    cuda.syncthreads()

    # 块内归约
    offset = bdim // 2
    while offset > 0:
        if tid < offset:
            sm_area[tid] += sm_area[tid + offset]
            sm_dot[tid]  += sm_dot[tid + offset]
        cuda.syncthreads()
        offset //= 2

    if tid == 0:
        area_mask = sm_area[0]
        dot_c     = sm_dot[0]

        if area_mask > 0.0 and area_center > 0.0:
            s1 = dot_c / area_mask
            s2 = dot_c / area_center
            depths[bid] = s1 if s1 < s2 else s2
        else:
            depths[bid] = 0.0


def compute_inclusion_scores(masks_data, use_gpu=True, batch_size=50):
    # ... 省略 CPU 分支不变 ...

    if not use_gpu or not cuda.is_available():
        # 原 CPU 路径
        ...

    print("\n=== 使用GPU加速计算包含分数深度 (block-per-mask + u8 传输) ===")
    start_time = time.time()

    preprocessing_time = 0.0
    gpu_transfer_time = 0.0
    gpu_compute_time = 0.0

    num_samples = masks_data.shape[0]
    print(f"样本数量: {num_samples}")
    print(f"Mask形状: {masks_data.shape[1:]}")
    print(f"批处理大小: {batch_size}")

    # 预处理：展平 + 计算全局中心 + u8 压缩
    t0 = time.time()
    masks_flat = masks_data.reshape(num_samples, -1)
    # 二值转 u8，H2D 传输更快 ×4
    masks_u8 = (masks_flat > 0).astype(np.uint8, copy=False)
    original_center = masks_flat.mean(axis=0, dtype=np.float32)
    area_center = np.float32(original_center.sum(dtype=np.float32))
    preprocessing_time += time.time() - t0

    # 把中心放上 GPU（只拷一次）
    t0 = time.time()
    d_center = cuda.to_device(original_center.astype(np.float32, copy=False))
    gpu_transfer_time += time.time() - t0

    all_depths = np.empty(num_samples, dtype=np.float32)

    threadsperblock = 256

    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        this_bs = batch_end - i

        # H2D：当前批（u8），bytes 只有 float32 的 1/4
        t0 = time.time()
        d_masks_u8 = cuda.to_device(masks_u8[i:batch_end])
        d_depths   = cuda.device_array(this_bs, dtype=np.float32)
        gpu_transfer_time += time.time() - t0

        blockspergrid = this_bs  # 一个 mask 一个 block
        t0 = time.time()
        depth_reduce_kernel_u8[blockspergrid, threadsperblock](d_masks_u8, d_center, area_center, d_depths)
        cuda.synchronize()
        gpu_compute_time += time.time() - t0

        t0 = time.time()
        all_depths[i:batch_end] = d_depths.copy_to_host()
        gpu_transfer_time += time.time() - t0

        del d_masks_u8, d_depths

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n=== GPU计算性能统计（优化后） ===")
    print(f" 总计算时间: {total_time:.3f}秒")
    print(f"   数据预处理: {preprocessing_time:.3f}秒 ({preprocessing_time/total_time*100:.1f}%)")
    print(f"   GPU计算时间: {gpu_compute_time:.3f}秒 ({gpu_compute_time/total_time*100:.1f}%)")
    print(f"   数据传输时间: {gpu_transfer_time:.3f}秒 ({gpu_transfer_time/total_time*100:.1f}%)")
    other_time = total_time - preprocessing_time - gpu_compute_time - gpu_transfer_time
    print(f"   其他开销: {other_time:.3f}秒")
    print(f" GPU利用率: {(gpu_compute_time/total_time*100 if total_time>0 else 0):.1f}%")

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

    # 2. 计算（>50 用批处理）
    use_batch_processing = masks_data.shape[0] > 50
    want_gpu = True  # 想用GPU就设 True
    t0 = time.time()
    inclusion_scores = compute_inclusion_scores(
        masks_data,
        use_gpu=want_gpu,
        batch_size=10 if use_batch_processing else masks_data.shape[0]
    )
    gpu_compute_time = time.time() - t0  # 粗略记录（包含函数内部预处理/传输）

    # 3. 排序分析
    t0 = time.time()
    sorted_indices, sorted_mask_names, sorted_scores = analyze_depth_ranking(inclusion_scores, mask_names)
    other_overhead_time += time.time() - t0

    # 4. 可视化
    t0 = time.time()
    visualize_depth_analysis(inclusion_scores, mask_names, sorted_indices, output_dir)
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
