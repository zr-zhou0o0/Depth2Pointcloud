import OpenEXR
import numpy as np
from PIL import Image
import os
import sys
import glob
from tqdm import tqdm

def convert_normal_exr(exr_path, png_path):
    """
    Converts a normal map from an EXR file to a PNG file.
    Remaps normal values from [-1, 1] to [0, 255] and preserves transparency.
    """
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = header['channels']['R'].type
    dtype = np.float16 if str(pt) == 'HALF' else np.float32

    # Read RGB channels
    red = np.frombuffer(exr_file.channel('R', pt), dtype=dtype).reshape(height, width)
    green = np.frombuffer(exr_file.channel('G', pt), dtype=dtype).reshape(height, width)
    blue = np.frombuffer(exr_file.channel('B', pt), dtype=dtype).reshape(height, width)
    
    # Remap RGB from [-1, 1] to [0, 1]
    rgb_float = np.dstack((red, green, blue)).astype(np.float32)
    rgb_0_1 = rgb_float * 0.5 + 0.5

    # Handle Alpha channel
    if 'A' in header['channels']:
        alpha = np.frombuffer(exr_file.channel('A', pt), dtype=dtype).reshape(height, width)
        alpha_float = alpha.astype(np.float32)
        # Combine RGB and Alpha
        rgba_0_1 = np.dstack((rgb_0_1, alpha_float))
        # Convert to 8-bit RGBA
        rgba_uint8 = (rgba_0_1 * 255).astype(np.uint8)
        img = Image.fromarray(rgba_uint8, 'RGBA')
    else:
        # Convert to 8-bit RGB
        rgb_uint8 = (rgb_0_1 * 255).astype(np.uint8)
        img = Image.fromarray(rgb_uint8, 'RGB')

    img.save(png_path)

def convert_depth_exr(exr_path, png_path):
    """
    Converts a depth map from an EXR file to a 16-bit grayscale PNG file.
    Normalizes the depth values to the full 16-bit range [0, 65535].
    """
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = header['channels']['R'].type
    dtype = np.float16 if str(pt) == 'HALF' else np.float32

    # Depth is usually in a single channel (e.g., 'R' or 'Z')
    channel_name = 'R' if 'R' in header['channels'] else 'Z'
    depth = np.frombuffer(exr_file.channel(channel_name, pt), dtype=dtype).reshape(height, width)
    
    # Normalize depth values to [0, 1]
    min_val, max_val = np.min(depth), np.max(depth)
    if max_val > min_val:
        normalized_depth = (depth - min_val) / (max_val - min_val)
    else:
        normalized_depth = np.zeros_like(depth)
        
    # Scale to 16-bit integer range [0, 65535]
    depth_uint16 = (normalized_depth * 65535).astype(np.uint16)
    
    # Create a 16-bit grayscale image
    img = Image.fromarray(depth_uint16, 'I;16')
    img.save(png_path)

def process_dataset(dataset_name):
    """
    Finds and converts all EXR files in the specified dataset directory.
    """
    base_path = os.path.join('datasets', dataset_name, 'renders_geo')
    if not os.path.isdir(base_path):
        print(f"Error: Directory not found at '{base_path}'")
        return

    # Find all .exr files
    exr_files = glob.glob(os.path.join(base_path, '**', '*.exr'), recursive=True)
    if not exr_files:
        print("No .exr files found to convert.")
        return

    print(f"Found {len(exr_files)} .exr files. Starting conversion...")

    for exr_path in tqdm(exr_files, desc="Converting EXR to PNG"):
        try:
            png_path = os.path.splitext(exr_path)[0] + '.png'
            
            if '_normal' in exr_path:
                convert_normal_exr(exr_path, png_path)
            # elif '_depth' in exr_path:
                # convert_depth_exr(exr_path, png_path)
            else:
                # Default to normal map conversion for other EXR files
                convert_normal_exr(exr_path, png_path)
        except Exception as e:
            print(f"\nFailed to convert '{exr_path}': {e}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python dataset_toolkits/exr_to_png.py <dataset_name>")
        print("Example: python dataset_toolkits/exr_to_png.py 3D-FUTURE")
        sys.exit(1)
        
    dataset_name_arg = sys.argv[1]
    process_dataset(dataset_name_arg)