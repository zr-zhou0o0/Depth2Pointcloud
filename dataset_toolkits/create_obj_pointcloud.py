import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# def camera_to_world_batched(points_cam, pose):
#     """
#     Convert points from camera coordinates to world coordinates
#     points_cam: [B, N, M, 3] (in use, N is ray number, M is points on a ray)
#     pose: [B, 4, 4] camera pose matrix
#     """
#     B, N, M, _ = points_cam.shape
#     ones = torch.ones((B, N, M), device=points_cam.device)  # homogeneous
#     x_cam = points_cam[:, :, :, 0]
#     y_cam = points_cam[:, :, :, 1]
#     z_cam = points_cam[:, :, :, 2]
#     points_cam_homo = torch.stack((x_cam, y_cam, z_cam, ones), dim=-1)  # [B, N, M, 4]
    
#     points_cam_homo = points_cam_homo.reshape(B, -1, 4)  # [B, N*M, 4]
#     points_cam_homo = points_cam_homo.permute(0, 2, 1)  # [B, 4, N*M]

#     points_world = torch.bmm(pose, points_cam_homo).permute(0, 2, 1)[:, :, :3]  # [B, N*M, 3]
#     # torch.bmm(pose, points_cam_homo): 批量矩阵乘法

#     return points_world.reshape(B, N, M, 3)


def depth_to_pointcloud(depth_map, camera_intrinsics, camera_pose, near, far, mask=None):
    """
    Convert depth map to 3D point cloud in world coordinates
    
    Args:
        depth_map: (H, W) depth values
        camera_intrinsics: (3, 3) camera intrinsic matrix
        camera_pose: (4, 4) camera pose matrix (c2w), OpenCV format (y for down, z for forward)
        mask: (H, W) optional mask to filter points
    
    Returns:
        points: (N, 3) 3D points in world coordinates
    """
    H, W = depth_map.shape
    device = depth_map.device
    # print("Depth range:", depth_map.min().item(), "to", depth_map.max().item())

    # Create pixel coordinates
    # y: top to down is 0 ~ H-1
    # x: left to right is 0 ~ W-1
    # shape: (H, W)
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    # Apply mask if provided
    if mask is not None:
        valid_mask = (depth_map > 0) & (mask > 0)
    else:
        valid_mask = depth_map > 0
    
    # Get valid pixels
    valid_y = y[valid_mask] # from (H, W) to (N,)
    valid_x = x[valid_mask] # from (H, W) to (N,)
    valid_depth = depth_map[valid_mask]
    # e.g. 
    # y = [[0, 0, 0, 0],
    #     [1, 1, 1, 1],
    #     [2, 2, 2, 2]]
    # x = [[0, 1, 2, 3],
    #     [0, 1, 2, 3],
    #     [0, 1, 2, 3]]
    # at the position (1,2), y=1, x=2
    
    if len(valid_depth) == 0:
        return torch.empty((0, 3), device=device)
    
    # Ray directions, pixel to camera coordinates
    pixel_coords = torch.stack([valid_x, valid_y, torch.ones_like(valid_x)], dim=0).float()  # (3, N), (x, y, 1)
    ray_dirs = torch.inverse(camera_intrinsics) @ pixel_coords  # (3, N)
    # 单位化方向向量 (?)，但是会导致mesh无法对齐，删除后mesh对齐正常。说明blender渲染深度的distance也是z轴距离，而不是沿射线的距离。
    # ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=0, keepdim=True)  

    # Camera coordinates to world coordinates
    R = camera_pose[:3, :3]  # (3, 3)
    ray_dirs_world = R @ ray_dirs  # (3, N)

    # Ray origin
    rays_o = camera_pose[:3, 3]  # (3,)
    rays_o = rays_o.view(1,3) # (1, 3)

    # Depth scale
    distance = valid_depth * (far - near) + near  # (N,)

    # Points
    points_world = (ray_dirs_world * distance.unsqueeze(0)) + rays_o.T  # (3, N)
    points_world = points_world.T  # (N, 3)

    # camera_coords = camera_coords * valid_depth.unsqueeze(0)  # Scale by depth
    # Transform to world coordinates
    # world_points = world_coords[:3].T  # (N, 3)
    
    return points_world


def load_camera_params(transforms_path, type='blender'):
    """
    Load camera parameters from transforms.json (Blender format)
    Outputs are opencv format
    """
    with open(transforms_path, 'r') as f:
        data = json.load(f)

    scale = data['scale']
    offset = data['offset']
    
    frames = data['frames']
    camera_params = []
    
    for frame in frames:
        # Get camera pose (transform matrix)
        transform_matrix = np.array(frame['transform_matrix'])
        camera_pose = torch.from_numpy(transform_matrix).float()
        
        # Transform保存的是c2w 但是是blender格式的
        # 已验证 这个转换矩阵是正确的
        if type=='blender':
            blender_to_cv = torch.tensor([
                [1,  0,  0,  0],
                [0, -1,  0,  0],
                [0,  0, -1,  0],
                [0,  0,  0,  1]
            ]).float()
            
            # 应用转换：T_cv = T_blender @ blender_to_cv
            camera_pose = camera_pose @ blender_to_cv
        
        # Get camera intrinsics from fov
        fov = frame['camera_angle_x']
        # Assuming square images and focal length calculation
        # This might need adjustment based on your specific camera model
        H = W = 512  # Default resolution, should match rendered images
        focal_length = W / (2 * np.tan(fov / 2))
        
        # Camera intrinsic matrix
        camera_intrinsics = torch.tensor([
            [focal_length, 0, W/2],
            [0, focal_length, H/2],
            [0, 0, 1]
        ]).float()

        depth_near = frame['depth']['min']
        depth_far = frame['depth']['max']
        
        camera_params.append({
            'pose': camera_pose,
            'intrinsics': camera_intrinsics,
            'file_path': frame['file_path'],
            'near': depth_near,
            'far': depth_far
        })
    
    return camera_params, scale, offset



# --- 新的线程池I/O + 主线程CUDA实现 ---
def load_depth_and_mask(renders_dir, i, device):
    depth_path = os.path.join(renders_dir, f'{i:03d}_depth.png')
    mask_path = os.path.join(renders_dir, f'{i:03d}_mask.png')
    if not os.path.exists(depth_path):
        return None, None
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_img is None:
        return None, None
    # print("Depth range loaded:", depth_img.min(), "to", depth_img.max())
    mask = None
    if os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask_img).float().to(device) / 255.0
    return depth_img, mask


def save_pointcloud_ply(points, filename):
    """Save point cloud as PLY file"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")



if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory containing the rendered data')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)

    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'pointcloud'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))

    # TODO
    if opt.instances is None:
        # Filter for objects that have been rendered with geo
        if 'rendered_geo' in metadata.columns:
            metadata = metadata[metadata['rendered_geo'] == True]
        else:
            print("Warning: 'rendered_geo' column not found in metadata. Make sure to run render_geo.py first.")
            # Fallback: check if renders_geo directories exist
            renders_exists = []
            for _, row in metadata.iterrows():
                sha256 = row['sha256']
                renders_dir = os.path.join(opt.output_dir, 'renders_no-geo', sha256)
                renders_exists.append(os.path.exists(os.path.join(renders_dir, 'transforms.json')))
            metadata = metadata[renders_exists]
        
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        
        # Filter out objects that are already processed
        if 'pointcloud_processed' in metadata.columns:
            metadata = metadata[metadata['pointcloud_processed'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # Filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        pointcloud_file = os.path.join(opt.output_dir, 'pointcloud_obj', f'{sha256}.ply')
        if os.path.exists(pointcloud_file):
            records.append({
                'sha256': sha256, 
                'pointcloud_processed': True, 
                'pointcloud_path': f'pointcloud_obj/{sha256}.ply'
            })
            metadata = metadata[metadata['sha256'] != sha256]

    print(f'Processing {len(metadata)} objects for point cloud generation...')

    # 主进程初始化 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processed = []
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc='Creating point clouds'):
        sha256 = row['sha256']
        renders_dir = os.path.join(opt.output_dir, 'renders_no-geo', sha256)
        pointcloud_dir = os.path.join(opt.output_dir, 'pointcloud_obj')
        os.makedirs(pointcloud_dir, exist_ok=True)
        output_file = os.path.join(pointcloud_dir, f'{sha256}.ply')
        if os.path.exists(output_file):
            processed.append({'sha256': sha256, 'pointcloud_processed': True, 'pointcloud_path': f'pointcloud/{sha256}.ply'})
            continue
        if not os.path.exists(renders_dir):
            print(f"Warning: renders directory not found for {sha256}")
            processed.append({'sha256': sha256, 'pointcloud_processed': False, 'pointcloud_path': None})
            continue
        transforms_path = os.path.join(renders_dir, 'transforms.json')
        if not os.path.exists(transforms_path):
            print(f"Warning: transforms.json not found for {sha256}")
            processed.append({'sha256': sha256, 'pointcloud_processed': False, 'pointcloud_path': None})
            continue
        camera_params, scale, offset = load_camera_params(transforms_path, type='blender')
        all_points = []
        # 线程池只做I/O
        with ThreadPoolExecutor(max_workers=opt.max_workers) as executor:
            future_to_idx = {executor.submit(load_depth_and_mask, renders_dir, i, device): (i, cam_param) for i, cam_param in enumerate(camera_params)}
            for future in as_completed(future_to_idx):
                i, cam_param = future_to_idx[future]
                try:
                    depth_img, mask = future.result()
                    if depth_img is None:
                        continue
                    depth_map = torch.from_numpy(depth_img.astype(np.float32)).to(device)
                    # depth_map = depth_map / 65535.0 * 10.0
                    depth_map = depth_map / 65535.0
                    camera_pose = cam_param['pose'].to(device)
                    camera_intrinsics = cam_param['intrinsics'].to(device)
                    near = cam_param['near']
                    far = cam_param['far']
                    points = depth_to_pointcloud(depth_map, camera_intrinsics, camera_pose, near, far, mask)
                    if len(points) > 0:
                        all_points.append(points)
                except Exception as e:
                    print(f"Error processing view {i} of {sha256}: {e}")
        if not all_points:
            print(f"Warning: No valid points generated for {sha256}")
            processed.append({'sha256': sha256, 'pointcloud_processed': False, 'pointcloud_path': None})
            continue
        combined_points = torch.cat(all_points, dim=0)

        # 反归一化
        # 注意这个顺序必须要与归一化的顺序相反
        combined_points = combined_points - torch.from_numpy(np.array(offset)).float().to(combined_points.device)
        combined_points = combined_points / scale

        points_np = combined_points.cpu().numpy()
        save_pointcloud_ply(points_np, output_file)
        processed.append({'sha256': sha256, 'pointcloud_processed': True, 'pointcloud_path': f'pointcloud/{sha256}.ply'})

    processed = pd.concat([pd.DataFrame(processed), pd.DataFrame.from_records(records)])
    processed.to_csv(os.path.join(opt.output_dir, f'pointcloud_processed_{opt.rank}.csv'), index=False)
