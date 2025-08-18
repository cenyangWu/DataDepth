import nibabel as nib
import numpy as np
from scipy import ndimage
import os
import glob
from pathlib import Path

def extract_surface(binary_mask, connectivity=1):
    """
    提取3D二值mask的表面
    
    Parameters:
    binary_mask: 3D numpy array, 二值mask (0和1)
    connectivity: int, 连接性 (1表示6连通，2表示18连通，3表示26连通)
    
    Returns:
    surface_mask: 3D numpy array, 表面mask
    """
    # 确保输入是二值的
    binary_mask = binary_mask.astype(bool)
    
    # 创建结构元素用于腐蚀操作
    if connectivity == 1:
        # 6-连通 (面连通)
        struct = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 2:
        # 18-连通 (面+边连通)
        struct = ndimage.generate_binary_structure(3, 2)
    else:
        # 26-连通 (面+边+角连通)
        struct = ndimage.generate_binary_structure(3, 3)
    
    # 腐蚀操作
    eroded = ndimage.binary_erosion(binary_mask, structure=struct)
    
    # 表面 = 原始mask - 腐蚀后的mask
    surface = binary_mask & ~eroded
    
    return surface.astype(np.uint8)

def process_nii_files(input_dir, output_dir, connectivity=1):
    """
    处理目录中的所有nii文件，提取表面并保存
    
    Parameters:
    input_dir: str, 输入目录路径
    output_dir: str, 输出目录路径
    connectivity: int, 连接性参数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有nii和nii.gz文件
    nii_files = glob.glob(os.path.join(input_dir, "*.nii")) + \
                glob.glob(os.path.join(input_dir, "*.nii.gz"))
    
    if not nii_files:
        print(f"在目录 {input_dir} 中未找到nii文件")
        return
    
    print(f"找到 {len(nii_files)} 个nii文件")
    
    for file_path in nii_files:
        print(f"正在处理: {os.path.basename(file_path)}")
        
        try:
            # 读取nii文件
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            
            # 转换为二值mask
            binary_mask = (data > 0.5).astype(np.uint8)
            
            # 检查是否有有效区域
            if np.sum(binary_mask) == 0:
                print(f"  警告: {os.path.basename(file_path)} 中没有有效区域")
                continue
            
            # 提取表面
            surface_mask = extract_surface(binary_mask, connectivity)
            
            # 创建新的nii图像
            surface_nii = nib.Nifti1Image(surface_mask, nii_img.affine, nii_img.header)
            
            # 生成输出文件名
            base_name = os.path.basename(file_path)
            if base_name.endswith('.nii.gz'):
                output_name = base_name.replace('.nii.gz', '_surface.nii.gz')
            else:
                output_name = base_name.replace('.nii', '_surface.nii')
            
            output_path = os.path.join(output_dir, output_name)
            
            # 保存文件
            nib.save(surface_nii, output_path)
            
            # 打印统计信息
            original_voxels = np.sum(binary_mask)
            surface_voxels = np.sum(surface_mask)
            print(f"  原始有效体素数: {original_voxels}")
            print(f"  表面体素数: {surface_voxels}")
            print(f"  表面比例: {surface_voxels/original_voxels*100:.2f}%")
            print(f"  已保存到: {output_path}")
            
        except Exception as e:
            print(f"  错误处理文件 {os.path.basename(file_path)}: {str(e)}")
    
    print(f"\n处理完成！结果保存在: {output_dir}")

def main():
    """主函数"""
    # 设置路径
    input_directory = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\50seg\processed_binary"
    output_directory = os.path.join(input_directory, "surface_extracted")
    
    print("=== 3D Mask表面提取工具 ===")
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_directory):
        print(f"错误: 输入目录不存在: {input_directory}")
        return
    
    # 处理文件
    process_nii_files(input_directory, output_directory, connectivity=1)

if __name__ == "__main__":
    main()
