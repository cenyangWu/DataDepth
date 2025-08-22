import os
import numpy as np
import nibabel as nib

# 1. 参数设置
src_dir = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\Allseg"
output_dir = os.path.join(src_dir, "ventricle_binary")

# 2. 创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建输出文件夹: {output_dir}")

# 3. 遍历目录并处理.nii文件
processed_count = 0
for fname in os.listdir(src_dir):
    if not fname.lower().endswith('.nii'):
        continue

    nii_path = os.path.join(src_dir, fname)
    output_path = os.path.join(output_dir, fname)
    
    # 4. 加载和处理数据
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # 创建二值化数据：值为1或20的点设为1，其他设为0
        binary_data = np.where((data == 3) | (data == 22), 1, 0).astype(np.uint8)
        
        # 保存处理后的文件
        new_img = nib.Nifti1Image(binary_data, img.affine, img.header)
        nib.save(new_img, output_path)
        
        processed_count += 1
        print(f"已处理: {fname} (原始值范围: {data.min():.1f}-{data.max():.1f})")
        
    except Exception as e:
        print(f"处理失败 {fname}: {e}")

print(f"处理完成！共处理了 {processed_count} 个文件，保存在: {output_dir}")
