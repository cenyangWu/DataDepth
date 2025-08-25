import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours
from sklearn.decomposition import KernelPCA,PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import pandas as pd
import time

# 添加新的辅助函数
def _get_agg_axes(masks):
    """获取聚合轴，用于在掩码的空间维度上求和"""
    masks = np.array(masks)
    if masks.ndim == 3:  # (num_masks, height, width)
        return (1, 2)
    elif masks.ndim == 4:  # (num_masks, depth, height, width)
        return (1, 2, 3)
    else:
        raise ValueError(f"Unsupported mask dimensions: {masks.ndim}")

def compute_epsilon_inclusion_depth(masks) -> np.array: 
    """第一种eID算法实现 - O(N)版本"""
    masks = np.array(masks)
    agg_axes = _get_agg_axes(masks)

    inverted_masks = 1 - masks
    area_normalized_masks = (masks.T / np.sum(masks, axis=agg_axes).T).T
    precompute_in = np.sum(inverted_masks, axis=0)
    precompute_out = np.sum(area_normalized_masks, axis=0)

    num_masks = len(masks)
    IN_in = num_masks - np.sum(area_normalized_masks * precompute_in, axis=agg_axes)
    IN_out = num_masks - np.sum(inverted_masks * precompute_out, axis=agg_axes)
    # We remove from the count in_ci, which we do not consider as it adds to both IN_in and IN_out equally
    return (np.minimum(IN_in, IN_out) - 1) / len(masks)

# Define necessary functions
def sphere_ensemble(num_masks, num_depth, num_rows, num_cols, center_mean=(0.5, 0.5, 0.5), center_std=(0, 0, 0),
                    radius_mean=0.25, radius_std=0.25 * 0.1, seed=None):

    rng = np.random.default_rng(seed)
    RADIUS_MEAN = np.minimum(num_depth, np.minimum(num_rows, num_cols)) * radius_mean
    RADIUS_STD = np.minimum(num_depth, np.minimum(num_rows, num_cols)) * radius_std
    radii = rng.normal(RADIUS_MEAN, RADIUS_STD, num_masks)
    centers_depth = rng.normal(num_depth * center_mean[0], num_depth * center_std[0], num_masks)
    centers_rows = rng.normal(num_rows * center_mean[1], num_rows * center_std[1], num_masks)
    centers_cols = rng.normal(num_cols * center_mean[2], num_cols * center_std[2], num_masks)

    masks = []
    for i in range(num_masks):
        mask = np.zeros((num_depth, num_rows, num_cols))
        # Create 3D sphere
        z, y, x = np.ogrid[:num_depth, :num_rows, :num_cols]
        distance = np.sqrt((z - centers_depth[i])**2 + (y - centers_rows[i])**2 + (x - centers_cols[i])**2)
        sphere_mask = distance <= radii[i]
        mask[sphere_mask] = 1
        masks.append(mask)

    return masks

def generate_efficient_ellipsoid_3d(num_depth, num_rows, num_cols, center, radii, rotation_angles=None, seed=None):
    """高效生成3D椭球体mask"""
    rng = np.random.default_rng(seed)
    
    # 创建坐标网格
    z, y, x = np.ogrid[:num_depth, :num_rows, :num_cols]
    
    # 计算椭球体方程
    center_z, center_y, center_x = center
    radius_z, radius_y, radius_x = radii
    
    # 基本椭球体方程
    ellipsoid_eq = ((x - center_x) / radius_x) ** 2 + \
                   ((y - center_y) / radius_y) ** 2 + \
                   ((z - center_z) / radius_z) ** 2
    
    # 添加小的随机扰动以增加变化
    if rotation_angles is not None:
        # 可以添加旋转变换，但为了效率暂时跳过
        pass
    
    mask = (ellipsoid_eq <= 1).astype(int)
    return mask

def main_shape_with_outliers_3d(num_masks, num_depth, num_rows, num_cols, 
                                 population_radius=0.3,
                                 normal_scale=0.1, outlier_scale=0.4,
                                 p_contamination=0.5, return_labels=False, seed=None):
    """高效生成3D椭球体masks，包含正常样本和异常样本"""
    
    rng = np.random.default_rng(seed)
    
    # 确定异常样本
    should_contaminate = rng.random(num_masks) < p_contamination
    
    contours = []
    labels = []
    
    # 基础半径计算
    min_dimension = min(num_depth, num_rows, num_cols)
    base_radius = population_radius * min_dimension
    
    for i in range(num_masks):
        # 随机生成椭球体中心（确保在合理范围内）
        center_z = rng.uniform(num_depth * 0.25, num_depth * 0.75)
        center_y = rng.uniform(num_rows * 0.25, num_rows * 0.75)
        center_x = rng.uniform(num_cols * 0.25, num_cols * 0.75)
        center = (center_z, center_y, center_x)
        
        # 根据是否为异常样本设置不同的半径变化
        if should_contaminate[i]:
            # 异常样本：更大的变化范围
            radius_z = base_radius * rng.uniform(0.5, 2.0) * (1 + rng.normal(0, outlier_scale))
            radius_y = base_radius * rng.uniform(0.5, 2.0) * (1 + rng.normal(0, outlier_scale))
            radius_x = base_radius * rng.uniform(0.5, 2.0) * (1 + rng.normal(0, outlier_scale))
            label = 1
        else:
            # 正常样本：较小的变化范围
            radius_z = base_radius * rng.uniform(0.8, 1.2) * (1 + rng.normal(0, normal_scale))
            radius_y = base_radius * rng.uniform(0.8, 1.2) * (1 + rng.normal(0, normal_scale))
            radius_x = base_radius * rng.uniform(0.8, 1.2) * (1 + rng.normal(0, normal_scale))
            label = 0
        
        # 确保半径为正值
        radii = (abs(radius_z), abs(radius_y), abs(radius_x))
        
        # 生成椭球体mask
        mask = generate_efficient_ellipsoid_3d(num_depth, num_rows, num_cols, center, radii, seed=seed+i)
        
        contours.append(mask)
        labels.append(label)
    
    labels = np.array(labels)
    
    if return_labels:
        return contours, labels
    else:
        return contours

def compute_sdf(binary_mask):
    inside_dist = distance_transform_edt(binary_mask)
    outside_dist = distance_transform_edt(1 - binary_mask)
    sdf = inside_dist - outside_dist
    return sdf

def extract_contours(masks):
    contours = []
    for mask in masks:
        contour = find_contours(mask, level=0.5)
        if contour:
            contours.append(contour[0])  # Take the first contour
        else:
            contours.append(np.array([]))
    return contours

def compute_inclusion_matrix(masks):
    """Matrix that, per contour says if its inside (1) or outside (-1).
    The entry is 0 if the relationship is ambiguous. 

    Parameters
    ----------
    masks : list
        list of ndarrays corresponding to an ensemble of binary masks.
    """
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    inclusion_mat = np.zeros((num_masks, num_masks))
    
    # Determine axes based on dimensionality
    if masks.ndim == 3:  # 2D masks: (num_masks, height, width)
        spatial_axes = (1, 2)
    elif masks.ndim == 4:  # 3D masks: (num_masks, depth, height, width)
        spatial_axes = (1, 2, 3)
    else:
        raise ValueError(f"Unsupported mask dimensions: {masks.ndim}")
    
    for i in range(num_masks):
        inclusion_mat[i, :] = np.all((masks & masks[i]) == masks[i], axis=spatial_axes)
        inclusion_mat[i, i] = 0
    return inclusion_mat

def compute_epsilon_inclusion_matrix(masks):
    """Matrix that, per contour says if its inside (1) or outside (-1).
    The entry is 0 if the relationship is ambiguous. 

    Parameters
    ----------
    masks : list
        list of ndarrays corresponding to an ensemble of binary masks.
    """
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    inclusion_mat = np.zeros((num_masks, num_masks))
    inv_masks = 1 - masks
    
    # Determine axes based on dimensionality
    if masks.ndim == 3:  # 2D masks: (num_masks, height, width)
        spatial_axes = (1, 2)
    elif masks.ndim == 4:  # 3D masks: (num_masks, depth, height, width)
        spatial_axes = (1, 2, 3)
    else:
        raise ValueError(f"Unsupported mask dimensions: {masks.ndim}")
    
    area = np.sum(masks, axis=spatial_axes)
    
    for i in range(num_masks):
        if area[i] > 0:  # Check to prevent division by zero
            inclusion_scores = 1 - np.sum(inv_masks & masks[i], axis=spatial_axes) / area[i]
        else:
            inclusion_scores = np.zeros(num_masks)  # If mask area is 0, set all relations to 0
            
        inclusion_mat[i, :] = inclusion_scores
        inclusion_mat[i, i] = 0  # Set diagonal to 0 as a mask cannot include itself

    return inclusion_mat

def sorted_depth(masks, depth="eid", metric="depth"):
    """第二种eID算法实现 - O(N^2)版本"""
    masks = np.array(masks, dtype=np.float32)
    num_masks = masks.shape[0]
    np.set_printoptions(threshold=np.inf)
    assert(depth in ["eid", "id", "cbd", "ecbd"])
    assert(metric in ["depth", "red"])
    if depth == "eid" or depth == "ecbd":
        inclusion_matrix = compute_epsilon_inclusion_matrix(masks)
        np.fill_diagonal(inclusion_matrix, 1)  # Required for feature parity with the O(N) version of eID.
    else:
        inclusion_matrix = compute_inclusion_matrix(masks)
    N = num_masks
    depths = np.zeros(num_masks)  # 用于存储每个掩码的深度值

    for i in range(num_masks):
        if depth in ["cbd", "id", "eid", "ecbd"]:
            N_a = np.sum(inclusion_matrix[i])
            N_b = np.sum(inclusion_matrix.T[i])

        if depth == "cbd" or depth == "ecbd":
            N_ab_range = N
            depth_in_cluster = (N_a * N_b) / (N_ab_range * N_ab_range)
        else:  # ID / eID
            depth_in_cluster = np.minimum(N_a, N_b) / N

        depths[i] = depth_in_cluster  # 存储当前掩码的深度值

    # 对深度值进行排序，获取排序后的索引（从大到小）
    sorted_indices = np.argsort(-depths)

    return sorted_indices, depths

def compute_inclusion_scores(window_flattened, num_samples):
    """Compute PID-mean depth"""
    original_center = np.mean(window_flattened, axis=0)
    
    # Compute PID-mean scores
    inclusion_scores = []
    for i in range(num_samples):
        mask = window_flattened[i]
        
        # Calculate mask area
        area_mask = np.sum(mask)
        
        # Calculate original_center sum (equivalent to "area")
        area_center = np.sum(original_center)
        
        if area_mask > 0 and area_center > 0:
            # Calculate PID-mean score between original_center and mask
            inv_center = 1 - original_center
            inclusion_score1 = 1 - np.sum(inv_center * mask) / area_mask
            
            # Calculate PID-mean score between mask and original_center
            inv_mask = 1 - mask
            inclusion_score2 = 1 - np.sum(inv_mask * original_center) / area_center
            
            # Take minimum of two PID-mean scores as depth
            depth = min(inclusion_score1, inclusion_score2)
        else:
            depth = 0
        
        inclusion_scores.append(depth)

    return np.array(inclusion_scores)

def time_depth_methods(num_samples_list):
    """测试不同样本数量下各种深度方法的计算时间"""
    num_depth = num_rows = num_cols = 50  # 使用3D数据，维度适中以支持大样本数量测试
    size_window = 50
    
    results = {
        'num_samples': [],
        'eid_v1_time': [],
        'pid_mean_time': []
    }
    
    for num_samples in num_samples_list:
        print(f"\nTesting sample size: {num_samples}")
        
        # 生成3D数据
        contours_masks, true_labels = main_shape_with_outliers_3d(
            num_samples, num_depth, num_rows, num_cols, return_labels=True, seed=66
        )
        
        # 创建窗口
        window = np.zeros((num_samples, size_window, size_window, size_window), dtype=np.float32)
        for k in range(num_samples):
            window[k] = contours_masks[k][0:size_window, 0:size_window, 0:size_window]
        
        # 将每个掩码展平成一维向量
        window_flattened = window.reshape(num_samples, -1)
        
        # 测试eID深度V1 (O(N)版本)
        print("  Computing eID depth V1 (O(N) version)...")
        start_time = time.time()
        eid_depths_v1 = compute_epsilon_inclusion_depth(contours_masks)
        eid_v1_time = time.time() - start_time
        
        # Test PID-mean depth
        print("  Computing PID-mean depth...")
        start_time = time.time()
        inclusion_scores = compute_inclusion_scores(window_flattened, num_samples)
        pid_mean_time = time.time() - start_time
        
        # Record results
        results['num_samples'].append(num_samples)
        results['eid_v1_time'].append(eid_v1_time)
        results['pid_mean_time'].append(pid_mean_time)
        
        # Print current results
        print(f"  eID V1 time: {eid_v1_time:.4f} seconds")
        print(f"  PID-mean time: {pid_mean_time:.4f} seconds")
    
    return results

# Test different sample sizes (modified to 100-20000 samples)
sample_sizes = [100, 1000, 3000, 10000, 20000]
print("Starting 3D depth computation time testing for different sample sizes...")

# Run tests
timing_results = time_depth_methods(sample_sizes)

# Create DataFrame and save results
df_results = pd.DataFrame(timing_results)
print(f"\nComplete test results:")
print(df_results)

# Save to CSV file
df_results.to_csv('depth_timing_results.csv', index=False)
print(f"\nResults saved to depth_timing_results.csv")

# Visualize time comparison
plt.figure(figsize=(12, 8))

plt.plot(df_results['num_samples'], df_results['eid_v1_time'], 'ro-', label='eID V1 (O(N))', linewidth=2, markersize=6)
plt.plot(df_results['num_samples'], df_results['pid_mean_time'], 'md-', label='PID-mean', linewidth=2, markersize=6)

plt.xlabel('Number of Samples', fontsize=12)
plt.ylabel('Computation Time (seconds)', fontsize=12)
plt.title('Computation Time vs Number of Samples for Different Depth Methods', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')  # Use log scale to better show differences

plt.tight_layout()
plt.show()

# Calculate and display performance comparison
print(f"\nPerformance comparison (relative to eID V1):")
print(f"{'Samples':>8} | {'PID/V1':>8}")
print("-" * 20)
for i, num_samples in enumerate(df_results['num_samples']):
    pid_ratio = df_results['pid_mean_time'][i] / df_results['eid_v1_time'][i]
    print(f"{num_samples:>8} | {pid_ratio:>8.2f}")

# Performance comparison analysis between eID V1 and PID-mean depth
print(f"\neID V1 and PID-mean depth performance comparison:")
print(f"{'Samples':>8} | {'eID V1 Time':>12} | {'PID Time':>10} | {'Ratio':>8}")
print("-" * 44)
for i, num_samples in enumerate(df_results['num_samples']):
    v1_time = df_results['eid_v1_time'][i]
    pid_time = df_results['pid_mean_time'][i]
    performance_ratio = pid_time / v1_time
    print(f"{num_samples:>8} | {v1_time:>12.4f} | {pid_time:>10.4f} | {performance_ratio:>8.2f}")