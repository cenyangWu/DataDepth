import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 读取.nii.gz文件
file_path = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\train\IXI041-Guys-0706\aligned_seg35_LPI.nii.gz"

# 加载医学图像
img = nib.load(file_path)
data = img.get_fdata()

# 将3D数据展平为1D数组用于绘制直方图
data_flat = data.flatten()

# 移除零值（通常是背景）以获得更好的直方图显示
data_nonzero = data_flat[data_flat > 0]

# 绘制直方图
plt.figure(figsize=(12, 5))

# 绘制包含零值的完整直方图
plt.subplot(1, 2, 1)
plt.hist(data_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram (All Values)')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 绘制不包含零值的直方图
plt.subplot(1, 2, 2)
plt.hist(data_nonzero, bins=100, alpha=0.7, color='green', edgecolor='black')
plt.title('Histogram (Non-zero Values)')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# 打印基本统计信息
print(f"Image shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Min value: {np.min(data)}")
print(f"Max value: {np.max(data)}")
print(f"Mean value: {np.mean(data):.4f}")
print(f"Standard deviation: {np.std(data):.4f}")
print(f"Number of non-zero voxels: {len(data_nonzero)}")
print(f"Total number of voxels: {len(data_flat)}")

plt.show()
