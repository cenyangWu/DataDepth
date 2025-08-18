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

def compute_inclusion_scores(masks_data):
    """
    计算包含分数深度
    
    Parameters:
    masks_data: numpy array, 形状为 (num_samples, *mask_shape)
    
    Returns:
    inclusion_scores: numpy array, 每个mask的包含分数深度
    """
    print(f"\n=== 开始计算包含分数深度 ===")
    start_time = time.time()
    
    num_samples = masks_data.shape[0]
    print(f"样本数量: {num_samples}")
    print(f"Mask形状: {masks_data.shape[1:]}")
    
    # 将mask数据展平
    masks_flattened = masks_data.reshape(num_samples, -1)
    print(f"展平后形状: {masks_flattened.shape}")
    
    # 计算原始中心（所有mask的平均值）
    original_center = np.mean(masks_flattened, axis=0)
    print(f"计算平均mask完成")
    
    # 计算包含分数
    inclusion_scores = []
    for i in range(num_samples):
        mask = masks_flattened[i]
        
        # 计算mask的面积
        area_mask = np.sum(mask)
        
        # 计算original_center的总和（相当于"面积"）
        area_center = np.sum(original_center)
        
        if area_mask > 0 and area_center > 0:
            # 计算original_center与mask的包含分数
            inv_center = 1 - original_center
            inclusion_score1 = 1 - np.sum(inv_center * mask) / area_mask
            
            # 计算mask与original_center的包含分数
            inv_mask = 1 - mask
            inclusion_score2 = 1 - np.sum(inv_mask * original_center) / area_center
            
            # 取两个包含分数的最小值作为深度
            depth = min(inclusion_score1, inclusion_score2)
        else:
            depth = 0
        
        inclusion_scores.append(depth)
        
        if (i + 1) % 10 == 0 or i == num_samples - 1:
            print(f"已处理 {i + 1}/{num_samples} 个mask")
    
    inclusion_scores = np.array(inclusion_scores)
    
    end_time = time.time()
    print(f"包含分数深度计算完成，耗时: {end_time - start_time:.2f}秒")
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

def visualize_depth_analysis(inclusion_scores, mask_names, sorted_indices):
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
    plt.show()

def create_mask_operations(masks_data, sorted_indices, mask_names, affine, header, output_dir):
    """
    创建前50%深度的并集、前100%并集、前50%交集三个文件
    
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
    
    total_count = len(sorted_indices)
    top_50_count = total_count // 2
    
    print(f"总mask数量: {total_count}")
    print(f"前50%数量: {top_50_count}")
    
    # 1. 前50%深度的并集
    print(f"\n--- 计算前50%深度的并集 ---")
    top_50_indices = sorted_indices[:top_50_count]
    top_50_masks = masks_data[top_50_indices]
    union_50 = np.any(top_50_masks > 0, axis=0).astype(np.uint8)
    
    print(f"前50%深度的mask文件:")
    for i, idx in enumerate(top_50_indices):
        print(f"  {i+1:2d}. {mask_names[idx]}")
    
    # 2. 前100%（全部）的并集
    print(f"\n--- 计算前100%（全部）的并集 ---")
    union_100 = np.any(masks_data > 0, axis=0).astype(np.uint8)
    
    # 3. 前50%深度的交集
    print(f"\n--- 计算前50%深度的交集 ---")
    intersection_50 = np.all(top_50_masks > 0, axis=0).astype(np.uint8)
    
    # 计算统计信息
    union_50_voxels = np.sum(union_50)
    union_100_voxels = np.sum(union_100)
    intersection_50_voxels = np.sum(intersection_50)
    total_voxels = union_50.size
    
    print(f"\n=== 操作结果统计 ===")
    print(f"前50%深度并集激活体素: {union_50_voxels:,} ({union_50_voxels/total_voxels*100:.2f}%)")
    print(f"前100%并集激活体素: {union_100_voxels:,} ({union_100_voxels/total_voxels*100:.2f}%)")
    print(f"前50%深度交集激活体素: {intersection_50_voxels:,} ({intersection_50_voxels/total_voxels*100:.2f}%)")
    
    # 保存文件
    results = {}
    
    # 保存前50%深度的并集
    union_50_img = nib.Nifti1Image(union_50, affine, header)
    union_50_path = os.path.join(output_dir, "top50percent_depth_union.nii")
    nib.save(union_50_img, union_50_path)
    results['union_50'] = {'data': union_50, 'path': union_50_path, 'voxels': union_50_voxels}
    print(f"前50%深度并集已保存: {union_50_path}")
    
    # 保存前100%并集
    union_100_img = nib.Nifti1Image(union_100, affine, header)
    union_100_path = os.path.join(output_dir, "all_masks_union.nii")
    nib.save(union_100_img, union_100_path)
    results['union_100'] = {'data': union_100, 'path': union_100_path, 'voxels': union_100_voxels}
    print(f"前100%并集已保存: {union_100_path}")
    
    # 保存前50%深度交集
    intersection_50_img = nib.Nifti1Image(intersection_50, affine, header)
    intersection_50_path = os.path.join(output_dir, "top50percent_depth_intersection.nii")
    nib.save(intersection_50_img, intersection_50_path)
    results['intersection_50'] = {'data': intersection_50, 'path': intersection_50_path, 'voxels': intersection_50_voxels}
    print(f"前50%深度交集已保存: {intersection_50_path}")
    
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
        f.write(f"前50%深度交集激活体素: {intersection_50_voxels:,} ({intersection_50_voxels/total_voxels*100:.2f}%)\n\n")
        
        f.write(f"前50%深度最高的mask文件列表:\n")
        for i, idx in enumerate(top_50_indices):
            f.write(f"{i+1:2d}. {mask_names[idx]}\n")
        
        f.write(f"\n生成的文件:\n")
        f.write(f"- {os.path.basename(union_50_path)}: 前50%深度并集\n")
        f.write(f"- {os.path.basename(union_100_path)}: 前100%并集\n")
        f.write(f"- {os.path.basename(intersection_50_path)}: 前50%深度交集\n")
    
    print(f"详细报告已保存: {report_path}")
    
    end_time = time.time()
    print(f"Mask操作完成，耗时: {end_time - start_time:.2f}秒")
    
    return results

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
    
    if output_dir:
        fig_path = os.path.join(output_dir, "mask_operations_visualization.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存: {fig_path}")
    
    plt.show()

def main():
    """主函数"""
    print("=== 包含分数深度分析工具 ===")
    total_start_time = time.time()
    
    # 设置路径
    mask_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\50seg\haimaprocessed_binary"
    output_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\50seg\haimaprocessed_binary\DepthAnalysis"
    
    print(f"Mask目录: {mask_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(mask_dir):
        print(f"错误: 目录不存在: {mask_dir}")
        return
    
    # 1. 加载所有mask文件
    masks_data, mask_names, affine, header = load_all_masks(mask_dir)
    
    if masks_data is None:
        print("没有成功加载任何mask文件")
        return
    
    # 2. 计算包含分数深度
    inclusion_scores = compute_inclusion_scores(masks_data)
    
    # 3. 进行深度排序分析
    sorted_indices, sorted_mask_names, sorted_scores = analyze_depth_ranking(inclusion_scores, mask_names)
    
    # 4. 可视化深度分析结果
    visualize_depth_analysis(inclusion_scores, mask_names, sorted_indices)
    
    # 5. 创建三个mask操作结果
    results = create_mask_operations(masks_data, sorted_indices, mask_names, affine, header, output_dir)
    
    # 6. 可视化mask操作结果
    if results:
        visualize_mask_operations(results, output_dir)
    
    # 7. 保存包含分数深度结果
    print(f"\n=== 保存分析结果 ===")
    np.save(os.path.join(output_dir, "inclusion_scores.npy"), inclusion_scores)
    np.save(os.path.join(output_dir, "sorted_indices.npy"), sorted_indices)
    print(f"包含分数深度分析结果已保存到: {output_dir}")
    
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    
    print(f"\n=== 运行时间总结 ===")
    print(f"总运行时间: {total_runtime:.2f}秒 ({total_runtime/60:.2f}分钟)")
    
    print(f"\n=== 分析总结 ===")
    print(f"处理了 {len(mask_names)} 个mask文件")
    print(f"成功计算包含分数深度")
    print(f"深度范围: {np.min(inclusion_scores):.4f} - {np.max(inclusion_scores):.4f}")
    print(f"平均深度: {np.mean(inclusion_scores):.4f}")
    
    if results:
        print(f"\n生成的文件:")
        for key, result in results.items():
            print(f"- {os.path.basename(result['path'])}: {result['voxels']:,} 激活体素")
        print(f"所有结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
