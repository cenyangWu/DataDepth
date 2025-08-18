import cupy as cp
import numpy as np
import os
import glob
import time
from multiprocessing import Pool, cpu_count
import functools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import nibabel as nib  # 添加nibabel用于处理nii文件

def compute_inclusion_scores(masks_data, use_gpu=True, batch_size=50):
    """
    计算包含分数深度
    
    Parameters:
    masks_data: numpy array, 形状为 (num_samples, *mask_shape)
    
    Returns:
    inclusion_scores: numpy array, 每个mask的包含分数深度
    """
    if not use_gpu:
        # CPU计算逻辑保持不变
        print("\n=== 使用CPU计算包含分数深度 ===")
        start_time = time.time()
        num_samples = masks_data.shape[0]
        masks_flattened = masks_data.reshape(num_samples, -1)
        original_center = np.mean(masks_flattened, axis=0)
        area_masks = np.sum(masks_flattened, axis=1)
        area_center = np.sum(original_center)
        inv_center = 1 - original_center
        inclusion_scores1 = 1 - np.sum(inv_center * masks_flattened, axis=1) / area_masks
        inv_masks = 1 - masks_flattened
        inclusion_scores2 = 1 - np.sum(inv_masks * original_center, axis=1) / area_center
        inclusion_scores1 = np.nan_to_num(inclusion_scores1)
        inclusion_scores2 = np.nan_to_num(inclusion_scores2)
        depths = np.minimum(inclusion_scores1, inclusion_scores2)
        depths[area_masks == 0] = 0
        depths[area_center == 0] = 0
        inclusion_scores = depths
        end_time = time.time()
        print(f"CPU计算完成，耗时: {end_time - start_time:.2f}秒")
        return inclusion_scores

    # GPU计算逻辑（带批处理）
    print("\n=== 使用GPU加速计算包含分数深度 (批处理) ===")
    start_time = time.time()
    
    # 性能统计变量
    preprocessing_time = 0
    gpu_transfer_time = 0
    gpu_compute_time = 0
    
    num_samples = masks_data.shape[0]
    print(f"样本数量: {num_samples}")
    print(f"Mask形状: {masks_data.shape[1:]}")
    print(f"批处理大小: {batch_size}")

    # 在CPU上计算整体平均mask，以节省GPU内存
    print("在CPU上计算平均mask...")
    prep_start = time.time()
    masks_flattened_cpu = masks_data.reshape(num_samples, -1)
    original_center_cpu = np.mean(masks_flattened_cpu, axis=0)
    preprocessing_time += time.time() - prep_start
    
    # 将中心和其逆转移到GPU
    print("将中心数据转移到GPU...")
    transfer_start = time.time()
    original_center_gpu = cp.asarray(original_center_cpu)
    inv_center_gpu = 1 - original_center_gpu
    area_center_gpu = cp.sum(original_center_gpu)
    gpu_transfer_time += time.time() - transfer_start
    print("中心数据已在GPU上准备就绪")

    all_depths = np.zeros(num_samples, dtype=np.float32)

    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        print(f"\n处理批次 {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size} (样本 {i}-{batch_end-1})...")
        
        # 获取当前批次数据并转移到GPU
        batch_transfer_start = time.time()
        batch_masks_cpu = masks_data[i:batch_end]
        batch_masks_gpu = cp.asarray(batch_masks_cpu)
        batch_masks_flattened_gpu = batch_masks_gpu.reshape(batch_masks_gpu.shape[0], -1)
        gpu_transfer_time += time.time() - batch_transfer_start
        
        # 计算包含分数
        batch_compute_start = time.time()
        area_masks_gpu = cp.sum(batch_masks_flattened_gpu, axis=1)
        
        # Score 1
        inclusion_scores1 = 1 - cp.sum(inv_center_gpu * batch_masks_flattened_gpu, axis=1) / area_masks_gpu
        
        # Score 2
        inv_masks_gpu = 1 - batch_masks_flattened_gpu
        inclusion_scores2 = 1 - cp.sum(inv_masks_gpu * original_center_gpu, axis=1) / area_center_gpu
        
        # 清理中间变量，释放显存
        del inv_masks_gpu
        cp.get_default_memory_pool().free_all_blocks()

        # 处理除零
        inclusion_scores1 = cp.nan_to_num(inclusion_scores1)
        inclusion_scores2 = cp.nan_to_num(inclusion_scores2)
        
        # 计算深度
        depths_gpu = cp.minimum(inclusion_scores1, inclusion_scores2)
        depths_gpu[area_masks_gpu == 0] = 0
        if area_center_gpu == 0:
            depths_gpu[:] = 0
        gpu_compute_time += time.time() - batch_compute_start
        
        # 将结果移回CPU
        result_transfer_start = time.time()
        all_depths[i:batch_end] = cp.asnumpy(depths_gpu)
        gpu_transfer_time += time.time() - result_transfer_start
        
        # 清理显存
        del batch_masks_gpu, batch_masks_flattened_gpu, area_masks_gpu, inclusion_scores1, inclusion_scores2, depths_gpu
        cp.get_default_memory_pool().free_all_blocks()
        print(f"批次 {i//batch_size + 1} 处理完毕，显存已清理")

    inclusion_scores = all_depths
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 输出详细性能统计
    print(f"\n=== GPU计算性能统计 ===")
    print(f" 总计算时间: {total_time:.3f}秒")
    print(f"   数据预处理: {preprocessing_time:.3f}秒 ({preprocessing_time/total_time*100:.1f}%)")
    print(f"   GPU计算时间: {gpu_compute_time:.3f}秒 ({gpu_compute_time/total_time*100:.1f}%)")
    print(f"   数据传输时间: {gpu_transfer_time:.3f}秒 ({gpu_transfer_time/total_time*100:.1f}%)")
    other_time = total_time - preprocessing_time - gpu_compute_time - gpu_transfer_time
    print(f"   其他开销: {other_time:.3f}秒")
    gpu_utilization = (gpu_compute_time / total_time) * 100
    print(f" GPU利用率: {gpu_utilization:.1f}%")
    
    print(f"\n包含分数深度计算完成，总耗时: {total_time:.2f}秒")
    print(f"深度分数统计:")
    print(f"  最小深度: {np.min(inclusion_scores):.4f}")
    print(f"  最大深度: {np.max(inclusion_scores):.4f}")
    print(f"  平均深度: {np.mean(inclusion_scores):.4f}")
    print(f"  深度标准差: {np.std(inclusion_scores):.4f}")
    
    return inclusion_scores

def load_all_masks(mask_dir):
    """
    加载目录中的所有mask文件
    
    Parameters:
    mask_dir: str, mask文件目录路径
    
    Returns:
    masks_data: numpy array, 所有mask数据 (num_files, depth, height, width)
    mask_names: list, mask文件名列表
    affine: numpy array, NIfTI的仿射变换矩阵
    header: NIfTI header
    """
    print(f"\n=== 开始加载Mask文件 ===")
    start_time = time.time()
    
    # 查找所有nii文件
    nii_files = glob.glob(os.path.join(mask_dir, "*.nii"))
    
    if not nii_files:
        print(f"在目录 {mask_dir} 中未找到nii文件")
        return None, None, None, None
    
    print(f"找到 {len(nii_files)} 个nii文件")
    
    # 读取第一个文件获取形状信息
    first_img = nib.load(nii_files[0])
    first_data = first_img.get_fdata()
    mask_shape = first_data.shape
    affine = first_img.affine
    header = first_img.header
    
    print(f"Mask文件形状: {mask_shape}")
    print(f"数据类型: {first_data.dtype}")
    
    # 初始化数组存储所有mask
    masks_data = np.zeros((len(nii_files),) + mask_shape, dtype=np.float32)
    mask_names = []
    
    # 加载所有mask文件
    successful_loads = 0
    for i, nii_file in enumerate(nii_files):
        try:
            print(f"正在加载 ({i+1}/{len(nii_files)}): {os.path.basename(nii_file)}")
            
            img = nib.load(nii_file)
            data = img.get_fdata()
            
            # 检查形状是否一致
            if data.shape != mask_shape:
                print(f"  警告: 形状不匹配 {data.shape} vs {mask_shape}，跳过")
                continue
            
            # 二值化处理（确保只有0和1）
            data_binary = (data > 0).astype(np.float32)
            masks_data[successful_loads] = data_binary
            mask_names.append(os.path.basename(nii_file))
            successful_loads += 1
            
            voxel_count = np.sum(data_binary)
            print(f"  激活体素数: {voxel_count:,}")
            
        except Exception as e:
            print(f"  加载失败: {str(e)}")
            continue
    
    # 裁剪数组到实际加载的文件数
    if successful_loads < len(nii_files):
        masks_data = masks_data[:successful_loads]
        print(f"实际成功加载: {successful_loads} 个文件")
    
    end_time = time.time()
    print(f"Mask文件加载完成，耗时: {end_time - start_time:.2f}秒")
    print(f"最终数据形状: {masks_data.shape}")
    
    return masks_data, mask_names, affine, header

def analyze_depth_ranking(inclusion_scores, mask_names):
    """
    分析深度排序结果
    
    Parameters:
    inclusion_scores: numpy array, 包含分数深度
    mask_names: list, mask文件名
    
    Returns:
    sorted_indices: numpy array, 按深度排序的索引（深度从高到低）
    sorted_mask_names: list, 按深度排序的mask名称
    sorted_scores: numpy array, 排序后的深度分数
    """
    print(f"\n=== 开始深度排序分析 ===")
    
    # 按深度从高到低排序（深度越高越重要）
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
    """
    可视化深度分析结果
    
    Parameters:
    inclusion_scores: numpy array, 包含分数深度
    mask_names: list, mask文件名
    sorted_indices: numpy array, 排序索引
    """
    print(f"\n=== 开始生成深度分析可视化 ===")
    
    fig = plt.figure(figsize=(20, 12))
    
    # 深度分布直方图
    plt.subplot(2, 4, 1)
    plt.hist(inclusion_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Inclusion Score Depth')
    plt.ylabel('Frequency')
    plt.title('Distribution of Inclusion Score Depths')
    plt.grid(True, alpha=0.3)
    
    # 深度排序图
    plt.subplot(2, 4, 2)
    sorted_scores = inclusion_scores[sorted_indices]
    plt.plot(range(len(sorted_scores)), sorted_scores, 'o-', color='red', alpha=0.7)
    plt.xlabel('Rank (sorted by depth)')
    plt.ylabel('Inclusion Score Depth')
    plt.title('Sorted Inclusion Score Depths')
    plt.grid(True, alpha=0.3)
    
    # 累计深度分布
    plt.subplot(2, 4, 3)
    cumulative_percent = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    plt.plot(sorted_scores, cumulative_percent, 'o-', color='purple', alpha=0.7)
    plt.xlabel('Inclusion Score Depth')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Depth Distribution')
    plt.grid(True, alpha=0.3)
    
    # 前50%和后50%的深度对比
    plt.subplot(2, 4, 4)
    mid_point = len(sorted_scores) // 2
    top_50_scores = sorted_scores[:mid_point]
    bottom_50_scores = sorted_scores[mid_point:]
    
    plt.hist([top_50_scores, bottom_50_scores], bins=15, alpha=0.7, 
             label=['Top 50%', 'Bottom 50%'], color=['red', 'blue'])
    plt.xlabel('Inclusion Score Depth')
    plt.ylabel('Frequency')
    plt.title('Top 50% vs Bottom 50% Depth Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 深度分数箱线图
    plt.subplot(2, 4, 5)
    plt.boxplot(inclusion_scores, vert=True)
    plt.ylabel('Inclusion Score Depth')
    plt.title('Inclusion Score Depth Box Plot')
    plt.grid(True, alpha=0.3)
    
    # 深度分数的统计信息文本
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
    
    # 前10名mask的深度分数条形图
    plt.subplot(2, 4, 7)
    top_10_indices = sorted_indices[:10]
    top_10_scores = inclusion_scores[top_10_indices]
    top_10_names = [mask_names[i][:15] for i in top_10_indices]  # 截断名称
    
    plt.barh(range(len(top_10_names)), top_10_scores, alpha=0.7, color='green')
    plt.yticks(range(len(top_10_names)), top_10_names)
    plt.xlabel('Inclusion Score Depth')
    plt.title('Top 10 Highest Depth Masks')
    plt.grid(True, alpha=0.3)
    
    # 深度分数的变化趋势
    plt.subplot(2, 4, 8)
    plt.plot(range(len(inclusion_scores)), inclusion_scores, 'o', alpha=0.6, markersize=4)
    plt.xlabel('Mask Index (original order)')
    plt.ylabel('Inclusion Score Depth')
    plt.title('Depth Scores by Original Order')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, "depth_analysis_visualization.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"深度分析可视化结果已保存: {fig_path}")
    
    plt.show()




def main():
    """主函数"""
    print("=== 包含分数深度分析工具 ===")
    total_start_time = time.time()
    
    # 性能统计变量
    preprocessing_time = 0
    gpu_compute_time = 0
    data_transfer_time = 0
    other_overhead_time = 0
    
    # 设置路径
    mask_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\Allseg\haimaprocessed_binary"
    output_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\Allseg\haimaprocessed_binary\DepthAnalysis"
    
    print(f"Mask目录: {mask_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(mask_dir):
        print(f"错误: 目录不存在: {mask_dir}")
        return
    
    # 1. 加载所有mask文件 (数据预处理)
    preprocessing_start = time.time()
    masks_data, mask_names, affine, header = load_all_masks(mask_dir)
    preprocessing_time += time.time() - preprocessing_start
    
    if masks_data is None:
        print("没有成功加载任何mask文件")
        return
    
    # 2. 计算包含分数深度
    # 当样本数大于50时，使用批处理来避免GPU内存溢出
    use_batch_processing = masks_data.shape[0] > 50
    
    # 记录GPU计算时间
    gpu_start = time.time()
    inclusion_scores = compute_inclusion_scores(masks_data, use_gpu=True, batch_size=10 if use_batch_processing else masks_data.shape[0])
    gpu_compute_time = time.time() - gpu_start
    
    # 3. 进行深度排序分析 (其他开销)
    other_start = time.time()
    sorted_indices, sorted_mask_names, sorted_scores = analyze_depth_ranking(inclusion_scores, mask_names)
    other_overhead_time += time.time() - other_start
    
    # 4. 可视化深度分析结果 (其他开销)
    vis_start = time.time()
    visualize_depth_analysis(inclusion_scores, mask_names, sorted_indices, output_dir)
    other_overhead_time += time.time() - vis_start
    
    # 5. 保存包含分数深度结果 (数据传输)
    print(f"\n=== 保存分析结果 ===")
    transfer_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "inclusion_scores.npy"), inclusion_scores)
    np.save(os.path.join(output_dir, "sorted_indices.npy"), sorted_indices)
    data_transfer_time = time.time() - transfer_start
    print(f"包含分数深度分析结果已保存到: {output_dir}")
    
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    
    # 计算GPU利用率
    gpu_utilization = (gpu_compute_time / total_runtime) * 100
    
    # 输出详细性能统计
    print(f"\n=== 性能统计 ===")
    print(f" 总计算时间: {total_runtime:.3f}秒")
    print(f"   数据预处理: {preprocessing_time:.3f}秒 ({preprocessing_time/total_runtime*100:.1f}%)")
    print(f"   GPU计算时间: {gpu_compute_time:.3f}秒 ({gpu_compute_time/total_runtime*100:.1f}%)")
    print(f"   数据传输时间: {data_transfer_time:.3f}秒 ({data_transfer_time/total_runtime*100:.1f}%)")
    print(f"   其他开销: {other_overhead_time:.3f}秒")
    print(f" GPU利用率: {gpu_utilization:.1f}%")
    
    print(f"\n=== 分析总结 ===")
    print(f"处理了 {len(mask_names)} 个mask文件")
    print(f"成功计算包含分数深度")
    print(f"深度范围: {np.min(inclusion_scores):.4f} - {np.max(inclusion_scores):.4f}")
    print(f"平均深度: {np.mean(inclusion_scores):.4f}")

if __name__ == "__main__":
    main()
