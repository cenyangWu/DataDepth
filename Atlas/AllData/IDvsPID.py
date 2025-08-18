import numpy as np
import os
import glob
import time
import nibabel as nib
import matplotlib.pyplot as plt

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

# === New helpers for ranking consistency and visualization ===

def compute_ranks(values, descending=True):
    """返回每个元素的名次（1为最佳）；对并列值使用平均名次。"""
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return np.array([])
    # 为避免浮点比较问题，使用排序后区间平均名次
    sort_idx = np.argsort(-values if descending else values)
    sorted_vals = values[sort_idx]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    next_rank = 1
    while i < n:
        j = i + 1
        # 将完全相等的值视为并列（通常浮点不完全相等，若有并列则按平均名次）
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (next_rank + (next_rank + (j - i) - 1)) / 2.0
        ranks[sort_idx[i:j]] = avg_rank
        next_rank += (j - i)
        i = j
    return ranks

def spearman_correlation(scores1, scores2):
    r1 = compute_ranks(scores1, descending=True)
    r2 = compute_ranks(scores2, descending=True)
    r1c = r1 - r1.mean()
    r2c = r2 - r2.mean()
    denom = np.sqrt((r1c ** 2).sum() * (r2c ** 2).sum())
    if denom == 0:
        return 0.0
    return float((r1c * r2c).sum() / denom)

def top_k_overlap_ratio(scores1, scores2, k):
    n = len(scores1)
    if n == 0:
        return 0.0
    k = max(1, min(k, n))
    top1 = set(np.argsort(scores1)[::-1][:k])
    top2 = set(np.argsort(scores2)[::-1][:k])
    return len(top1 & top2) / k

def evaluate_consistency(scores1, scores2, mask_names):
    print("\n=== 排序一致性评估 ===")
    rho = spearman_correlation(scores1, scores2)
    print(f"Spearman秩相关: {rho:.4f}")
    for k in [5, 10, 20]:
        ratio = top_k_overlap_ratio(scores1, scores2, k)
        print(f"Top-{k} 重叠比例: {ratio*100:.1f}%")

def visualize_rankings(scores1, scores2, mask_names, method1_name="Method 1", method2_name="Method 2"):
    n = len(scores1)
    ranks1 = compute_ranks(scores1, descending=True)
    ranks2 = compute_ranks(scores2, descending=True)
    rho = spearman_correlation(scores1, scores2)

    plt.figure(figsize=(14, 5))

    # Scatter: rank vs rank
    plt.subplot(1, 2, 1)
    plt.scatter(ranks1, ranks2, alpha=0.7)
    lim = [0.5, max(n + 0.5, 5)]
    plt.plot(lim, lim, 'r--', label='y = x')
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel(f'Rank ({method1_name})')
    plt.ylabel(f'Rank ({method2_name})')
    plt.title(f'Rank Agreement (Spearman = {rho:.3f})')
    plt.legend()

    # Line plot: depths sorted by method 1
    order = np.argsort(scores1)[::-1]
    x = np.arange(n)
    plt.subplot(1, 2, 2)
    plt.plot(x, np.asarray(scores1)[order], label=method1_name, linewidth=2)
    plt.plot(x, np.asarray(scores2)[order], label=method2_name, linewidth=2)
    plt.xlabel('Items (sorted by Method 1)')
    plt.ylabel('Depth score')
    plt.title('Depth by ranking (sorted by Method 1)')
    plt.legend()

    plt.tight_layout()
    plt.show()

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

    # Consistency evaluation and visualization
    evaluate_consistency(inclusion_scores1, inclusion_scores2, mask_names)
    visualize_rankings(inclusion_scores1, inclusion_scores2, mask_names, method1_name="Method 1", method2_name="Method 2")
    
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    
    print(f"\n=== 运行时间总结 ===")
    print(f"总运行时间: {total_runtime:.2f}秒 ({total_runtime/60:.2f}分钟)")

if __name__ == "__main__":
    main()
