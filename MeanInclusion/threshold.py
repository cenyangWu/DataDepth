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
pca = PCA(n_components=8)
window_pca = pca.fit_transform(window_flattened)

# 计算PCA空间中的欧氏距离
pca_center = np.mean(window_pca, axis=0)
pca_euclidean_distances = np.linalg.norm(window_pca - pca_center, axis=1)

# 执行Kernel PCA
kpca = KernelPCA(n_components=8, kernel='rbf', gamma=1e-4)
window_kpca = kpca.fit_transform(window_flattened)

# 计算新的深度方法：使用包含分数计算Center和每个mask之间的关系
original_center = np.mean(window_flattened, axis=0)

# 将original_center二值化为mask形式（大于某个阈值的为1，否则为0）
center_threshold = 0.5
center_mask = (original_center > center_threshold).astype(float)

# 计算包含分数
inclusion_scores = []
for i in range(num_samples):
    mask = window_flattened[i]
    
    # 计算mask的面积
    area_mask = np.sum(mask)
    
    # 计算center_mask的面积
    area_center = np.sum(center_mask)
    
    if area_mask > 0 and area_center > 0:
        # 计算center_mask与mask的包含分数
        inv_center = 1 - center_mask
        inclusion_score1 = 1 - np.sum(inv_center * mask) / area_mask
        
        # 计算mask与center_mask的包含分数
        inv_mask = 1 - mask
        inclusion_score2 = 1 - np.sum(inv_mask * center_mask) / area_center
        
        # 取两个包含分数的最小值作为深度
        depth = min(inclusion_score1, inclusion_score2)
    else:
        depth = 0
    
    inclusion_scores.append(depth)

inclusion_scores = np.array(inclusion_scores)

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
id_sorted_indices = sorted_depth(contours_masks, depth="eid")  # eID排序
cbd_sorted_indices = sorted_depth(contours_masks, depth="cbd")  # CBD排序
pca_sorted_indices = np.argsort(normalized_pca_ed)  # PCA空间欧氏距离排序
original_sorted_indices = np.argsort(-normalized_inclusion_scores)  # 包含分数排序（分数大的排在前面）
# 创建四种排序的轮廓
id_sorted_contours = [sampled_contours[i] for i in id_sorted_indices]
cbd_sorted_contours = [sampled_contours[i] for i in cbd_sorted_indices]
pca_sorted_contours = [sampled_contours[i] for i in pca_sorted_indices]
original_sorted_contours = [sampled_contours[i] for i in original_sorted_indices]

# 对比可视化四种距离的排序结果
fig, axes = plt.subplots(1, 4, figsize=(32, 8))

# 使用相同的颜色映射
cmap = plt.get_cmap('viridis_r')

# 创建基于排序位置的颜色映射值（0到1）
rank_colors = np.linspace(0, 1, len(pca_sorted_indices))

# 1. 绘制ID排序结果
for idx, contour in enumerate(id_sorted_contours):
    rank_color = rank_colors[idx]
    color = cmap(rank_color)
    axes[0].plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5)

axes[0].invert_yaxis()
axes[0].set_title('Contours Sorted by eID')
axes[0].axis('off')  # 去掉边框和边框上的数字

# 2. 绘制CBD排序结果
for idx, contour in enumerate(cbd_sorted_contours):
    rank_color = rank_colors[idx]
    color = cmap(rank_color)
    axes[1].plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5)

axes[1].invert_yaxis()
axes[1].set_title('Contours Sorted by CBD')
axes[1].axis('off')  # 去掉边框和边框上的数字

# 3. 绘制PCA空间欧氏距离排序结果
for idx, contour in enumerate(pca_sorted_contours):
    rank_color = rank_colors[idx]
    color = cmap(rank_color)
    axes[2].plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5)

axes[2].invert_yaxis()
axes[2].set_title('Contours Sorted by PCA Space Euclidean Distance')
axes[2].axis('off')  # 去掉边框和边框上的数字

# 4. 绘制原空间L1距离排序结果
for idx, contour in enumerate(original_sorted_contours):
    rank_color = rank_colors[idx]
    color = cmap(rank_color)
    axes[3].plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5)

axes[3].invert_yaxis()
axes[3].set_title('Contours Sorted by Inclusion Score Method')
axes[3].axis('off')  # 去掉边框和边框上的数字

# 为每个子图添加颜色条
for i in range(4):
    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Rank Position (0: Nearest, 1: Farthest)')

plt.tight_layout()
plt.savefig('four_methods_comparison.png')
plt.show()

# 在KPCA空间中可视化四种排序方法和聚类结果
fig, axes = plt.subplots(2, 3, figsize=(24, 16))

# 子图1: 聚类标签
scatter1 = axes[0, 0].scatter(window_kpca[:, 0], window_kpca[:, 1], c=true_labels, cmap='viridis')
axes[0, 0].set_xlabel('Kernel PCA Component 1')
axes[0, 0].set_ylabel('Kernel PCA Component 2')
axes[0, 0].set_title('Cluster Labels')
cbar1 = fig.colorbar(scatter1, ax=axes[0, 0])
cbar1.set_label('Cluster Label')

# 子图2: ID排序
scatter2 = axes[0, 1].scatter(window_kpca[:, 0], window_kpca[:, 1], c=id_sorted_indices, cmap='coolwarm')
axes[0, 1].set_xlabel('Kernel PCA Component 1')
axes[0, 1].set_ylabel('Kernel PCA Component 2')
axes[0, 1].set_title('eID Sorting')
cbar2 = fig.colorbar(scatter2, ax=axes[0, 1])
cbar2.set_label('Sort Index')

# 子图3: CBD排序
scatter3 = axes[0, 2].scatter(window_kpca[:, 0], window_kpca[:, 1], c=cbd_sorted_indices, cmap='coolwarm')
axes[0, 2].set_xlabel('Kernel PCA Component 1')
axes[0, 2].set_ylabel('Kernel PCA Component 2')
axes[0, 2].set_title('CBD Sorting')
cbar3 = fig.colorbar(scatter3, ax=axes[0, 2])
cbar3.set_label('Sort Index')

# 子图4: PCA空间欧氏距离排序
scatter4 = axes[1, 0].scatter(window_kpca[:, 0], window_kpca[:, 1], c=pca_sorted_indices, cmap='coolwarm')
axes[1, 0].set_xlabel('Kernel PCA Component 1')
axes[1, 0].set_ylabel('Kernel PCA Component 2')
axes[1, 0].set_title('PCA Space Euclidean Distance Sorting')
cbar4 = fig.colorbar(scatter4, ax=axes[1, 0])
cbar4.set_label('Sort Index')

# 子图5: 原空间L1距离排序
scatter5 = axes[1, 1].scatter(window_kpca[:, 0], window_kpca[:, 1], c=original_sorted_indices, cmap='coolwarm')
axes[1, 1].set_xlabel('Kernel PCA Component 1')
axes[1, 1].set_ylabel('Kernel PCA Component 2')
axes[1, 1].set_title('Inclusion Score Sorting')
cbar5 = fig.colorbar(scatter5, ax=axes[1, 1])
cbar5.set_label('Sort Index')

# 隐藏多余的子图位置
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('kpca_space_comparison.png')
plt.show()

# 分析四种排序方法的差异
print("\nAnalysis of Four Sorting Methods:")
methods = ['eID', 'CBD', 'PCA+ED', 'Inclusion Score']
sorted_indices_list = [id_sorted_indices, cbd_sorted_indices, pca_sorted_indices, original_sorted_indices]

# 计算每个样本在四种排序方法下的排名
all_ranks = np.zeros((num_samples, 4))
for i, sorted_indices in enumerate(sorted_indices_list):
    for j, idx in enumerate(sorted_indices):
        all_ranks[idx, i] = j

# 计算每个样本的最大排名差异
max_differences = np.zeros(num_samples)
for i in range(num_samples):
    pairwise_diffs = []
    for j in range(len(methods)):
        for k in range(j+1, len(methods)):
            pairwise_diffs.append(abs(all_ranks[i, j] - all_ranks[i, k]))
    max_differences[i] = max(pairwise_diffs)

# 找出排名差异最大的5个轮廓
top5_diff_indices = np.argsort(max_differences)[-5:][::-1]  # 按差异从大到小排序
print(f"\nTop 5 contours with largest ranking differences: {top5_diff_indices}")

# 输出详细的排名差异信息
print("\nDetailed Ranking Information:")
print(f"{'Contour ID':^10}|{'eID Rank':^10}|{'CBD Rank':^10}|{'PCA+ED Rank':^15}|{'Inclusion Score':^15}|{'Max Diff':^10}")
print("-" * 75)
for idx in top5_diff_indices:
    ranks = [all_ranks[idx, i] for i in range(4)]
    max_diff = max([abs(ranks[i] - ranks[j]) for i in range(4) for j in range(i+1, 4)])
    print(f"{idx:^10}|{ranks[0]:^10.0f}|{ranks[1]:^10.0f}|{ranks[2]:^15.0f}|{ranks[3]:^15.0f}|{max_diff:^10.0f}")

# 可视化差异最大的5个轮廓
plt.figure(figsize=(14, 10))

# 计算所有轮廓的坐标范围，以确保与第一个图使用相同的比例
all_x = []
all_y = []
for contour in sampled_contours:
    all_x.extend(contour[:, 0])
    all_y.extend(contour[:, 1])
x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)
# 增加一些边距
x_margin = (x_max - x_min) * 0.05
y_margin = (y_max - y_min) * 0.05
x_lim = [x_min - x_margin, x_max + x_margin]
y_lim = [y_max + y_margin, y_min - y_margin]  # 注意y轴是反向的

# 绘制所有轮廓作为背景（灰色）
for i in range(num_samples):
    if i not in top5_diff_indices:
        contour = sampled_contours[i]
        plt.plot(contour[:, 0], contour[:, 1], color='lightgray', linewidth=0.5, alpha=0.3)

# 绘制差异最大的轮廓并添加标签
cmap = plt.cm.tab10
for i, idx in enumerate(top5_diff_indices):
    contour = sampled_contours[idx]
    color = cmap(i % 10)
    plt.plot(contour[:, 0], contour[:, 1], color=color, linewidth=3.0)
    
    # 在轮廓的中心添加索引标签
    center_x = np.mean(contour[:, 0])
    center_y = np.mean(contour[:, 1])
    plt.text(center_x, center_y, str(idx), fontsize=14, weight='bold', color='white',
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='black', boxstyle='circle,pad=0.3'))

plt.gca().invert_yaxis()
plt.title('Top 5 Contours with Largest Ranking Differences', fontsize=16)
plt.axis('off')  # 去掉边框和边框上的数字
plt.xlim(x_lim)  # 设置x轴范围
plt.ylim(y_lim)  # 设置y轴范围
plt.gca().set_aspect('equal')  # 设置等比例

# 添加排名差异的信息表格
cell_text = []
for idx in top5_diff_indices:
    ranks = [all_ranks[idx, i] for i in range(4)]
    max_diff = max([abs(ranks[i] - ranks[j]) for i in range(4) for j in range(i+1, 4)])
    cell_text.append([idx] + [f"{r:.0f}" for r in ranks] + [f"{max_diff:.0f}"])

plt.table(cellText=cell_text,
          colLabels=['Contour ID', 'eID Rank', 'CBD Rank', 'PCA+ED Rank', 'Inclusion Score', 'Max Diff'],
          loc='lower center',
          cellLoc='center',
          bbox=[0.1, -0.25, 0.8, 0.2])

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # 为表格留出空间
plt.savefig('top5_rank_difference_contours.png', dpi=300)
plt.show()

# 绘制不同排序方法之间的散点图对比（选择一些有代表性的对比）
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# eID vs CBD
axes[0, 0].scatter(all_ranks[:, 0], all_ranks[:, 1], alpha=0.6)
axes[0, 0].set_xlabel('eID Rank', fontsize=12)
axes[0, 0].set_ylabel('CBD Rank', fontsize=12)
axes[0, 0].set_title('eID Rank vs CBD Rank', fontsize=14)
axes[0, 0].plot([0, num_samples], [0, num_samples], 'k--', alpha=0.3)  # 对角线
axes[0, 0].grid(True, linestyle='--', alpha=0.5)

# eID vs PCA Space
axes[0, 1].scatter(all_ranks[:, 0], all_ranks[:, 2], alpha=0.6)
axes[0, 1].set_xlabel('eID Rank', fontsize=12)
axes[0, 1].set_ylabel('PCA+ED Rank', fontsize=12)
axes[0, 1].set_title('eID Rank vs PCA+ED Rank', fontsize=14)
axes[0, 1].plot([0, num_samples], [0, num_samples], 'k--', alpha=0.3)  # 对角线
axes[0, 1].grid(True, linestyle='--', alpha=0.5)

# eID vs KPCA Space
axes[0, 2].scatter(all_ranks[:, 0], all_ranks[:, 3], alpha=0.6)
axes[0, 2].set_xlabel('eID Rank', fontsize=12)
axes[0, 2].set_ylabel('Inclusion Score Rank', fontsize=12)
axes[0, 2].set_title('eID Rank vs Inclusion Score Rank', fontsize=14)
axes[0, 2].plot([0, num_samples], [0, num_samples], 'k--', alpha=0.3)  # 对角线
axes[0, 2].grid(True, linestyle='--', alpha=0.5)

# CBD vs PCA Space
axes[1, 0].scatter(all_ranks[:, 1], all_ranks[:, 2], alpha=0.6)
axes[1, 0].set_xlabel('CBD Rank', fontsize=12)
axes[1, 0].set_ylabel('PCA+ED Rank', fontsize=12)
axes[1, 0].set_title('CBD Rank vs PCA+ED Rank', fontsize=14)
axes[1, 0].plot([0, num_samples], [0, num_samples], 'k--', alpha=0.3)  # 对角线
axes[1, 0].grid(True, linestyle='--', alpha=0.5)

# CBD vs KPCA Space
axes[1, 1].scatter(all_ranks[:, 1], all_ranks[:, 3], alpha=0.6)
axes[1, 1].set_xlabel('CBD Rank', fontsize=12)
axes[1, 1].set_ylabel('Inclusion Score Rank', fontsize=12)
axes[1, 1].set_title('CBD Rank vs Inclusion Score Rank', fontsize=14)
axes[1, 1].plot([0, num_samples], [0, num_samples], 'k--', alpha=0.3)  # 对角线
axes[1, 1].grid(True, linestyle='--', alpha=0.5)

# PCA Space vs KPCA Space
axes[1, 2].scatter(all_ranks[:, 2], all_ranks[:, 3], alpha=0.6)
axes[1, 2].set_xlabel('PCA+ED Rank', fontsize=12)
axes[1, 2].set_ylabel('Inclusion Score Rank', fontsize=12)
axes[1, 2].set_title('PCA+ED Rank vs Inclusion Score Rank', fontsize=14)
axes[1, 2].plot([0, num_samples], [0, num_samples], 'k--', alpha=0.3)  # 对角线
axes[1, 2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('rank_correlation_scatter.png', dpi=300)
plt.show()

# 计算四种排序方式两两间的线性相关性和Kendall Tau距离
methods = ['eID', 'CBD', 'PCA+ED', 'Inclusion Score']
sorted_indices_list = [id_sorted_indices, cbd_sorted_indices, pca_sorted_indices, original_sorted_indices]

# 计算线性相关性（Pearson相关系数）
linear_corr_matrix = np.zeros((4, 4))
kendall_corr_matrix = np.zeros((4, 4))
kendall_distance_matrix = np.zeros((4, 4))

for i in range(4):
    for j in range(4):
        # 使用排序后的排名计算相关系数
        linear_corr, _ = pearsonr(all_ranks[:, i], all_ranks[:, j])
        linear_corr_matrix[i, j] = linear_corr
        
        # 计算Kendall Tau相关系数
        kendall_corr, _ = kendalltau(all_ranks[:, i], all_ranks[:, j])
        kendall_corr_matrix[i, j] = kendall_corr
        
        # 计算Kendall Tau距离 = (1 - 相关系数) / 2
        kendall_distance_matrix[i, j] = (1 - kendall_corr) / 2

# 创建DataFrame以美观打印结果
linear_corr_df = pd.DataFrame(linear_corr_matrix, index=methods, columns=methods)
kendall_corr_df = pd.DataFrame(kendall_corr_matrix, index=methods, columns=methods)
kendall_distance_df = pd.DataFrame(kendall_distance_matrix, index=methods, columns=methods)

print("\n线性相关性Pearson相关系数:")
print(linear_corr_df)

print("\nKendall Tau相关系数:")
print(kendall_corr_df)

print("\nKendall Tau距离:")
print(kendall_distance_df)

# 可视化相关性矩阵
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 线性相关性热图
im0 = axes[0].imshow(linear_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_title('Pearson', fontsize=16)
axes[0].set_xticks(np.arange(len(methods)))
axes[0].set_yticks(np.arange(len(methods)))
axes[0].set_xticklabels(methods, fontsize=12)
axes[0].set_yticklabels(methods, fontsize=12)
plt.colorbar(im0, ax=axes[0])

# 添加数值标签
for i in range(len(methods)):
    for j in range(len(methods)):
        text = axes[0].text(j, i, f"{linear_corr_matrix[i, j]:.3f}",
                           ha="center", va="center", color="black" if abs(linear_corr_matrix[i, j]) < 0.7 else "white",
                           fontsize=12)

# Kendall Tau距离热图
im1 = axes[1].imshow(kendall_distance_matrix, cmap='coolwarm', vmin=0, vmax=0.5)
axes[1].set_title('Kendall Tau', fontsize=16)
axes[1].set_xticks(np.arange(len(methods)))
axes[1].set_yticks(np.arange(len(methods)))
axes[1].set_xticklabels(methods, fontsize=12)
axes[1].set_yticklabels(methods, fontsize=12)
plt.colorbar(im1, ax=axes[1])

# 添加数值标签
for i in range(len(methods)):
    for j in range(len(methods)):
        text = axes[1].text(j, i, f"{kendall_distance_matrix[i, j]:.3f}",
                           ha="center", va="center", color="black" if kendall_distance_matrix[i, j] > 0.25 else "white",
                           fontsize=12)

plt.tight_layout()
plt.savefig('correlation_matrices.png', dpi=300)
plt.show()

# 将original_center还原为二维图像并可视化
print(f"\noriginal_center形状: {original_center.shape}")
print(f"size_window: {size_window}")

# 将一维的original_center重新reshape为二维图像
original_center_2d = original_center.reshape(size_window, size_window)

# 可视化原空间的平均中心图像
plt.figure(figsize=(10, 8))
plt.imshow(original_center_2d, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Average Pixel Value')
plt.title('Original Space Average Center (2D Visualization)', fontsize=16)
plt.xlabel('X coordinate', fontsize=12)
plt.ylabel('Y coordinate', fontsize=12)
plt.show()

# 另外用灰度图显示
plt.figure(figsize=(10, 8))
plt.imshow(original_center_2d, cmap='gray', interpolation='nearest')
plt.colorbar(label='Average Pixel Value')
plt.title('Original Space Average Center (Grayscale)', fontsize=16)
plt.xlabel('X coordinate', fontsize=12)
plt.ylabel('Y coordinate', fontsize=12)
plt.show()

# 显示一些统计信息
print(f"\noriginal_center_2d 统计信息:")
print(f"形状: {original_center_2d.shape}")
print(f"最小值: {original_center_2d.min():.4f}")
print(f"最大值: {original_center_2d.max():.4f}")
print(f"平均值: {original_center_2d.mean():.4f}")
print(f"标准差: {original_center_2d.std():.4f}")