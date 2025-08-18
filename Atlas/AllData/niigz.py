import os
import gzip
import shutil
from pathlib import Path

def convert_niigz_to_nii(source_dir):
    """
    Convert all .nii.gz files to .nii files in the specified directory
    and delete the original .nii.gz files
    
    Args:
        source_dir (str): Path to the directory containing .nii.gz files
    """
    source_path = Path(source_dir)
    
    # Check if directory exists
    if not source_path.exists():
        print(f"Error: Directory {source_dir} does not exist!")
        return
    
    # Find all .nii.gz files
    nii_gz_files = list(source_path.glob("*.nii.gz"))
    
    if not nii_gz_files:
        print(f"No .nii.gz files found in {source_dir}")
        return
    
    print(f"Found {len(nii_gz_files)} .nii.gz files to convert")
    
    successful_conversions = 0
    failed_conversions = 0
    
    for gz_file in nii_gz_files:
        try:
            # Create the output .nii filename
            nii_file = gz_file.with_suffix('')  # Remove .gz extension
            
            print(f"Converting: {gz_file.name} -> {nii_file.name}")
            
            # Decompress the file
            with gzip.open(gz_file, 'rb') as f_in:
                with open(nii_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Verify the conversion was successful
            if nii_file.exists() and nii_file.stat().st_size > 0:
                # Delete the original .nii.gz file
                gz_file.unlink()
                print(f"✓ Successfully converted and deleted: {gz_file.name}")
                successful_conversions += 1
            else:
                print(f"✗ Conversion failed for: {gz_file.name}")
                failed_conversions += 1
                # Clean up failed conversion
                if nii_file.exists():
                    nii_file.unlink()
                    
        except Exception as e:
            print(f"✗ Error converting {gz_file.name}: {str(e)}")
            failed_conversions += 1
            # Clean up in case of error
            nii_file = gz_file.with_suffix('')
            if nii_file.exists():
                nii_file.unlink()
    
    print(f"\nConversion completed:")
    print(f"- Successful conversions: {successful_conversions}")
    print(f"- Failed conversions: {failed_conversions}")
    print(f"- Total files processed: {len(nii_gz_files)}")

# Main execution
if __name__ == "__main__":
    # Target directory
    target_directory = r"C:\Users\xrVis001\Desktop\EyeData\IXI-400-LPI\AllSeg"
    
    print("Starting conversion of .nii.gz files to .nii files...")
    print(f"Target directory: {target_directory}")
    print("-" * 60)
    
    convert_niigz_to_nii(target_directory)
    
    print("-" * 60)
    print("Conversion process completed!")
