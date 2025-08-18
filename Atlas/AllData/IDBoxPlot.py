import numpy as np
import os
import glob
import time
import nibabel as nib

def _get_agg_axes(masks):
    return tuple(range(1, masks.ndim))

def compute_inclusion_scores(masks_data):
    """
    计算包含分数深度 (方法一)
    
    Parameters:
    masks_data: numpy array, 形状为 (num_samples, *mask_shape)
    
    Returns:
    inclusion_scores: numpy array, 每个mask的包含分数深度
    """
    print(f"\n=== 开始计算包含分数深度 (方法一) ===")
    start_time = time.time()
    
    num_samples = masks_data.shape[0]
    
    masks_flattened = masks_data.reshape(num_samples, -1)
    
    original_center = np.mean(masks_flattened, axis=0)
    
    inclusion_scores = []
    for i in range(num_samples):
        mask = masks_flattened[i]
        area_mask = np.sum(mask)
        area_center = np.sum(original_center)
        
        if area_mask > 0 and area_center > 0:
            inv_center = 1 - original_center
            inclusion_score1 = 1 - np.sum(inv_center * mask) / area_mask
            
            inv_mask = 1 - mask
            inclusion_score2 = 1 - np.sum(inv_mask * original_center) / area_center
            
            depth = min(inclusion_score1, inclusion_score2)
        else:
            depth = 0
        
        inclusion_scores.append(depth)
        
    inclusion_scores = np.array(inclusion_scores)
    
    end_time = time.time()
    print(f"方法一计算完成，耗时: {end_time - start_time:.2f}秒")
    
    return inclusion_scores

def compute_epsilon_inclusion_depth(masks):
    """第二种eID算法实现 - O(N)版本"""
    print(f"\n=== 开始计算包含分数深度 (方法二) ===")
    start_time = time.time()
    masks = np.array(masks)
    agg_axes = _get_agg_axes(masks)

    inverted_masks = 1 - masks
    # Normalize area only for non-empty masks to avoid division by zero
    sum_masks = np.sum(masks, axis=agg_axes)
    # Add a small epsilon to avoid division by zero for empty masks
    area_normalized_masks = (masks.T / (sum_masks.T + 1e-8)).T
    
    precompute_in = np.sum(inverted_masks, axis=0)
    precompute_out = np.sum(area_normalized_masks, axis=0)

    num_masks = len(masks)
    IN_in = num_masks - np.sum(area_normalized_masks * precompute_in, axis=agg_axes)
    IN_out = num_masks - np.sum(inverted_masks * precompute_out, axis=agg_axes)
    
    # We remove from the count in_ci, which we do not consider as it adds to both IN_in and IN_out equally
    depths = (np.minimum(IN_in, IN_out) - 1) / len(masks)
    end_time = time.time()
    print(f"方法二计算完成，耗时: {end_time - start_time:.2f}秒")
    return depths

def load_all_masks(mask_dir):
    """
    加载目录中的所有mask文件
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
    
    masks_data = np.zeros((len(nii_files),) + mask_shape, dtype=np.float32)
    mask_names = []
    
    successful_loads = 0
    for i, nii_file in enumerate(nii_files):
        try:
            img = nib.load(nii_file)
            data = img.get_fdata()
            
            if data.shape != mask_shape:
                print(f"  警告: 形状不匹配 {data.shape} vs {mask_shape}，跳过")
                continue
            
            data_binary = (data > 0).astype(np.float32)
            masks_data[successful_loads] = data_binary
            mask_names.append(os.path.basename(nii_file))
            successful_loads += 1
            
        except Exception as e:
            print(f"  加载失败: {str(e)}")
            continue
    
    if successful_loads < len(nii_files):
        masks_data = masks_data[:successful_loads]
    
    end_time = time.time()
    print(f"Mask文件加载完成，耗时: {end_time - start_time:.2f}秒")
    
    return masks_data, mask_names, affine, header

def analyze_depth_ranking(inclusion_scores, mask_names, method_name):
    """
    分析深度排序结果
    """
    print(f"\n=== {method_name} 深度排序分析 ===")
    
    sorted_indices = np.argsort(inclusion_scores)[::-1]
    sorted_scores = inclusion_scores[sorted_indices]
    sorted_mask_names = [mask_names[i] for i in sorted_indices]
    
    print(f"深度最高的5个mask:")
    for i in range(min(5, len(sorted_mask_names))):
        print(f"  {i+1}. {sorted_mask_names[i]} (深度: {sorted_scores[i]:.4f})")
    
    print(f"\n深度最低的5个mask:")
    for i in range(max(0, len(sorted_mask_names)-5), len(sorted_mask_names)):
        rank = len(sorted_mask_names) - i
        print(f"  倒数{rank}. {sorted_mask_names[i]} (深度: {sorted_scores[i]:.4f})")
    
    return sorted_indices, sorted_mask_names, sorted_scores

def main():
    """主函数"""
    print("=== 包含分数深度分析工具 ===")
    total_start_time = time.time()
    
    # NOTE: You might need to change this path to your data directory
    mask_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\AllSeg\haimaprocessed_binary"
    
    print(f"Mask目录: {mask_dir}")
    
    if not os.path.exists(mask_dir):
        print(f"错误: 目录不存在: {mask_dir}")
        return
    
    masks_data, mask_names, affine, header = load_all_masks(mask_dir)
    
    if masks_data is None:
        print("没有成功加载任何mask文件")
        return
    
    # Method 1
    inclusion_scores1 = compute_inclusion_scores(masks_data)
    analyze_depth_ranking(inclusion_scores1, mask_names, "方法一")

    # Method 2
    inclusion_scores2 = compute_epsilon_inclusion_depth(masks_data)
    analyze_depth_ranking(inclusion_scores2, mask_names, "方法二")
    
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    
    print(f"\n=== 运行时间总结 ===")
    print(f"总运行时间: {total_runtime:.2f}秒 ({total_runtime/60:.2f}分钟)")

if __name__ == "__main__":
    main()
