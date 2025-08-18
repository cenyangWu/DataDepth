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

# Generate contours and true labels
num_samples = 50
num_rows = num_cols = 100
size_window = 100
contours_masks, true_labels = main_shape_with_outliers(num_samples, num_rows, num_cols, return_labels=True, seed=66)

window = np.zeros((num_samples, size_window, size_window), dtype=np.float32)
i = 0
j = 0
for k in range(num_samples):
    window[k] = contours_masks[k][i:i+size_window, j:j+size_window]

# Extract contours from window
contours = extract_contours(window)

# Number of points to sample along each contour
N = 50
sampled_contours = []
# Sampling
for contour in contours:
    if contour.size == 0:
        # If contour is empty, use zeros
        sampled_points = np.zeros((N, 2))
    else:
        # Compute cumulative arc length along the contour
        deltas = np.diff(contour, axis=0)
        segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
        total_length = cumulative_lengths[-1]
        if total_length == 0:
            # Contour is a single point, repeat it N times
            sampled_points = np.repeat(contour[0][np.newaxis, :], N, axis=0)
        else:
            # Normalize cumulative lengths to [0,1]
            normalized_lengths = cumulative_lengths / total_length
            # Sample N equally spaced points along the normalized arc length
            sample_points = np.linspace(0, 1, N)
            # Interpolate x and y coordinates
            interp_func_x = interp1d(normalized_lengths, contour[:, 1], kind='linear')
            interp_func_y = interp1d(normalized_lengths, contour[:, 0], kind='linear')
            sampled_x = interp_func_x(sample_points)
            sampled_y = interp_func_y(sample_points)
            sampled_points = np.vstack((sampled_x, sampled_y)).T  # Shape (N, 2)
    sampled_contours.append(sampled_points)

# Now, flatten the sampled points into vectors
flattened_contours = [points.flatten() for points in sampled_contours]
flattened_array = np.array(flattened_contours)
flattened_array = np.array(flattened_contours)
# original_center = np.mean(flattened_array, axis=0)
# original_cov = np.cov(flattened_array, rowvar=False)
# original_mahalanobis_distances = np.array([mahalanobis(x, original_center, np.linalg.inv(original_cov)) for x in flattened_array])
# 直接使用window作为输入，将每个掩码展平成一维向量
window_flattened = window.reshape(num_samples, -1)  # 展平每个掩码

# 执行PCA降维
print("开始计算PCA空间欧氏距离...")
start_time_pca = time.time()

pca = PCA(n_components=8)
window_pca = pca.fit_transform(window_flattened)

# 计算PCA空间中的欧氏距离
pca_center = np.mean(window_pca, axis=0)
pca_euclidean_distances = np.linalg.norm(window_pca - pca_center, axis=1)

end_time_pca = time.time()
pca_time = end_time_pca - start_time_pca
print(f"PCA空间欧氏距离计算时间: {pca_time:.4f} 秒")

# 执行Kernel PCA
kpca = KernelPCA(n_components=8, kernel='rbf', gamma=1e-4)
window_kpca = kpca.fit_transform(window_flattened)

# 计算新的深度方法：使用包含分数计算Center和每个mask之间的关系
print("开始计算包含分数深度...")
start_time_inclusion = time.time()

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

inclusion_scores = np.array(inclusion_scores)

end_time_inclusion = time.time()
inclusion_time = end_time_inclusion - start_time_inclusion
print(f"包含分数深度计算时间: {inclusion_time:.4f} 秒")

# 归一化距离
min_pca_ed = pca_euclidean_distances.min()
max_pca_ed = pca_euclidean_distances.max()
normalized_pca_ed = (pca_euclidean_distances - min_pca_ed) / (max_pca_ed - min_pca_ed)

# 归一化包含分数（注意：分数越大越好，所以这里归一化后用于排序）
min_inclusion = inclusion_scores.min()
max_inclusion = inclusion_scores.max()
if max_inclusion > min_inclusion:
    normalized_inclusion_scores = (inclusion_scores - min_inclusion) / (max_inclusion - min_inclusion)
else:
    normalized_inclusion_scores = inclusion_scores


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

    return sorted_indices

# 获取四种不同的排序结果
print("\n开始计算eID深度...")
start_time_eid = time.time()
# 使用新的eID计算函数
eid_depths = compute_epsilon_inclusion_depth(contours_masks)
id_sorted_indices = np.argsort(-eid_depths)  # 按深度从大到小排序
end_time_eid = time.time()
eid_time = end_time_eid - start_time_eid
print(f"eID深度计算时间: {eid_time:.4f} 秒")

print("开始计算CBD深度...")
start_time_cbd = time.time()
cbd_sorted_indices = sorted_depth(contours_masks, depth="cbd")  # CBD排序
end_time_cbd = time.time()
cbd_time = end_time_cbd - start_time_cbd
print(f"CBD深度计算时间: {cbd_time:.4f} 秒")

pca_sorted_indices = np.argsort(normalized_pca_ed)  # PCA空间欧氏距离排序
original_sorted_indices = np.argsort(-normalized_inclusion_scores)  # 包含分数排序（分数大的排在前面）






# 将original_center还原为二维图像并可视化
print(f"\noriginal_center形状: {original_center.shape}")
print(f"size_window: {size_window}")

# 将一维的original_center重新reshape为二维图像
original_center_2d = original_center.reshape(size_window, size_window)

# 可视化原空间的平均中心图像
# 第一张图：原始平均中心图像
plt.figure(figsize=(8, 6))
plt.imshow(original_center_2d, cmap='viridis', interpolation='nearest')
plt.title('Probability Mask')
plt.axis('off')  # 删除刻度
plt.show()

# 第二张图：以0.5为阈值的二值化图像
plt.figure(figsize=(8, 6))
binary_center = (original_center_2d > 0.5).astype(float)
plt.imshow(binary_center, cmap='gray', interpolation='nearest')
plt.title('Binary Mask')
plt.axis('off')  # 删除刻度
plt.show()



# 显示一些统计信息
print(f"\noriginal_center_2d 统计信息:")
print(f"形状: {original_center_2d.shape}")
print(f"最小值: {original_center_2d.min():.4f}")
print(f"最大值: {original_center_2d.max():.4f}")
print(f"平均值: {original_center_2d.mean():.4f}")
print(f"标准差: {original_center_2d.std():.4f}")