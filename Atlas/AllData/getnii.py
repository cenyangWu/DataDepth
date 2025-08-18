import os
import shutil
from pathlib import Path

def copy_seg_files():
    """
    遍历train文件夹中的所有子文件夹，将包含'seg'的nii文件复制到AllSeg目录
    """
    # 源目录和目标目录
    source_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\train"
    target_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\AllSeg"
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 统计复制的文件数量
    copied_count = 0
    
    print(f"开始遍历源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print("-" * 50)
    
    # 遍历源目录中的所有文件夹
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        
        # 确保是文件夹
        if os.path.isdir(folder_path):
            print(f"正在处理文件夹: {folder_name}")
            
            # 遍历文件夹中的文件
            for file_name in os.listdir(folder_path):
                # 检查是否是nii或nii.gz文件且包含'seg'
                if (file_name.endswith('.nii') or file_name.endswith('.nii.gz')) and 'seg' in file_name.lower():
                    source_file = os.path.join(folder_path, file_name)
                    
                    # 获取文件扩展名
                    if file_name.endswith('.nii.gz'):
                        extension = '.nii.gz'
                    else:
                        extension = '.nii'
                    
                    # 使用文件夹名作为新文件名
                    new_file_name = folder_name + extension
                    target_file = os.path.join(target_dir, new_file_name)
                    
                    try:
                        # 复制文件并重命名
                        shutil.copy2(source_file, target_file)
                        print(f"  已复制: {file_name} -> {new_file_name}")
                        copied_count += 1
                    except Exception as e:
                        print(f"  复制失败: {file_name}, 错误: {e}")
    
    print("-" * 50)
    print(f"复制完成！总共复制了 {copied_count} 个文件")

if __name__ == "__main__":
    copy_seg_files()
