import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from skimage.draw import ellipse, polygon2mask
from skimage.measure import find_contours
from sklearn.decomposition import KernelPCA,PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm
from scipy.interpolate import splprep, splev
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr, kendalltau
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
def circle_ensemble(num_masks, num_rows, num_cols, center_mean=(0.5, 0.5), center_std=(0, 0),
                    radius_mean=0.25, radius_std=0.25 * 0.1, seed=None):

    rng = np.random.default_rng(seed)
    RADIUS_MEAN = np.minimum(num_rows, num_cols) * radius_mean
    RADIUS_STD = np.minimum(num_rows, num_cols) * radius_std
    radii = rng.normal(RADIUS_MEAN, RADIUS_STD, num_masks)
    centers_rows = rng.normal(num_rows * center_mean[0], num_rows * center_std[0], num_masks)
    centers_cols = rng.normal(num_cols * center_mean[1], num_cols * center_std[1], num_masks)

    masks = []
    for i in range(num_masks):
        mask = np.zeros((num_rows, num_cols))
        rr, cc = ellipse(centers_rows[i], centers_cols[i], radii[i], radii[i], shape=(num_rows, num_cols))
        mask[rr.astype(int), cc.astype(int)] = 1
        masks.append(mask)

    return masks

def get_base_gp(num_masks, domain_points, scale=0.01, sigma=1.0, seed=None):
    rng = np.random.default_rng(seed)
    thetas = domain_points.flatten().reshape(-1, 1)
    num_vertices = thetas.size
    gp_mean = np.zeros(num_vertices)

    gp_cov_sin = scale * np.exp(-(1 / (2 * sigma)) * cdist(np.sin(thetas), np.sin(thetas), "sqeuclidean"))
    gp_sample_sin = rng.multivariate_normal(gp_mean, gp_cov_sin, num_masks)
    gp_cov_cos = scale * np.exp(-(1 / (2 * sigma)) * cdist(np.cos(thetas), np.cos(thetas), "sqeuclidean"))
    gp_sample_cos = rng.multivariate_normal(gp_mean, gp_cov_cos, num_masks)

    return gp_sample_sin + gp_sample_cos

def get_xy_coords(angles, radii):
    num_members = radii.shape[0]
    angles = angles.flatten().reshape(1, -1)
    angles = np.repeat(angles, num_members, axis=0)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y

def rasterize_coords(x_coords, y_coords, num_rows, num_cols):
    masks = []
    for xc, yc in zip(x_coords, y_coords):
        coords_arr = np.vstack((xc, yc)).T
        coords_arr *= num_rows // 2
        coords_arr += num_cols // 2
        mask = polygon2mask((num_rows, num_cols), coords_arr).astype(float)
        masks.append(mask)
    return masks

def main_shape_with_outliers(num_masks, num_rows, num_cols, num_vertices=100, 
                             population_radius=0.5,
                             normal_scale=0.003, normal_freq=0.9,
                             outlier_scale=0.009, outlier_freq=0.04,
                             p_contamination=0.5, return_labels=False, seed=None):

    rng = np.random.default_rng(seed)
    thetas = np.linspace(0, 2 * np.pi, num_vertices)
    population_radius = np.ones_like(thetas) * population_radius

    gp_sample_normal = get_base_gp(num_masks, thetas, scale=normal_scale, sigma=normal_freq, seed=seed)+0.1
    gp_sample_outliers = get_base_gp(num_masks, thetas, scale=outlier_scale, sigma=outlier_freq, seed=seed)

    should_contaminate = rng.random(num_masks) < p_contamination
    should_contaminate = should_contaminate.reshape(-1, 1)
    should_contaminate = np.repeat(should_contaminate, len(thetas), axis=1)

    radii = population_radius + gp_sample_normal * (~should_contaminate) + gp_sample_outliers * should_contaminate

    xs, ys = get_xy_coords(thetas, radii)
    contours = rasterize_coords(xs, ys, num_rows, num_cols)
    labels = should_contaminate[:, 0].astype(int)

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
    for i in range(num_masks):
        inclusion_mat[i, :] = np.all((masks & masks[i]) == masks[i], axis=(1, 2))
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
    area = np.sum(masks, axis=(1, 2))
    
    for i in range(num_masks):
        if area[i] > 0:  # Check to prevent division by zero
            inclusion_scores = 1 - np.sum(inv_masks & masks[i], axis=(1, 2)) / area[i]
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
    """计算包含分数深度"""
    original_center = np.mean(window_flattened, axis=0)
    
    # 计算包含分数
    inclusion_scores = []
    for i in range(num_samples):
        mask = window_flattened[i]
        
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

    return np.array(inclusion_scores)

def time_depth_methods(num_samples_list):
    """测试不同样本数量下各种深度方法的计算时间"""
    num_rows = num_cols = 100
    size_window = 100
    
    results = {
        'num_samples': [],
        'eid_v1_time': [],
        'inclusion_time': []
    }
    
    for num_samples in num_samples_list:
        print(f"\n测试样本数量: {num_samples}")
        
        # 生成数据
        contours_masks, true_labels = main_shape_with_outliers(
            num_samples, num_rows, num_cols, return_labels=True, seed=66
        )
        
        # 创建窗口
        window = np.zeros((num_samples, size_window, size_window), dtype=np.float32)
        for k in range(num_samples):
            window[k] = contours_masks[k][0:size_window, 0:size_window]
        
        # 将每个掩码展平成一维向量
        window_flattened = window.reshape(num_samples, -1)
        
        # 测试eID深度V1 (O(N)版本)
        print("  计算eID深度V1 (O(N)版本)...")
        start_time = time.time()
        eid_depths_v1 = compute_epsilon_inclusion_depth(contours_masks)
        eid_v1_time = time.time() - start_time
        
        # 测试包含分数深度
        print("  计算包含分数深度...")
        start_time = time.time()
        inclusion_scores = compute_inclusion_scores(window_flattened, num_samples)
        inclusion_time = time.time() - start_time
        
        # 记录结果
        results['num_samples'].append(num_samples)
        results['eid_v1_time'].append(eid_v1_time)
        results['inclusion_time'].append(inclusion_time)
        
        # 打印当前结果
        print(f"  eID V1时间: {eid_v1_time:.4f}秒")
        print(f"  包含分数时间: {inclusion_time:.4f}秒")
    
    return results

# 测试不同的样本数量
sample_sizes = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
print("开始测试不同样本数量下的深度计算时间...")

# 运行测试
timing_results = time_depth_methods(sample_sizes)

# 创建DataFrame并保存结果
df_results = pd.DataFrame(timing_results)
print(f"\n完整测试结果:")
print(df_results)

# 保存到CSV文件
df_results.to_csv('depth_timing_results.csv', index=False)
print(f"\n结果已保存到 depth_timing_results.csv")

# 可视化时间对比
plt.figure(figsize=(12, 8))

plt.plot(df_results['num_samples'], df_results['eid_v1_time'], 'ro-', label='eID V1 (O(N))', linewidth=2, markersize=6)
plt.plot(df_results['num_samples'], df_results['inclusion_time'], 'md-', label='Inclusion Score', linewidth=2, markersize=6)

plt.xlabel('Number of Samples', fontsize=12)
plt.ylabel('Computation Time (seconds)', fontsize=12)
plt.title('Computation Time vs Number of Samples for Different Depth Methods', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')  # 使用对数坐标更好地显示差异

plt.tight_layout()
plt.show()

# 计算和显示性能比较
print(f"\n性能比较 (相对于eID V1的倍数):")
print(f"{'样本数':>8} | {'包含/V1':>8}")
print("-" * 20)
for i, num_samples in enumerate(df_results['num_samples']):
    inclusion_ratio = df_results['inclusion_time'][i] / df_results['eid_v1_time'][i]
    print(f"{num_samples:>8} | {inclusion_ratio:>8.2f}")

# eID V1和包含分数深度的性能对比分析
print(f"\neID V1和包含分数深度性能对比:")
print(f"{'样本数':>8} | {'eID V1时间':>10} | {'包含时间':>10} | {'性能比':>8}")
print("-" * 44)
for i, num_samples in enumerate(df_results['num_samples']):
    v1_time = df_results['eid_v1_time'][i]
    inclusion_time = df_results['inclusion_time'][i]
    performance_ratio = inclusion_time / v1_time
    print(f"{num_samples:>8} | {v1_time:>10.4f} | {inclusion_time:>10.4f} | {performance_ratio:>8.2f}")