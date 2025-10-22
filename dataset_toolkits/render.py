import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence
import subprocess


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _get_best_gpu(num_gpus=2):
    """Select the GPU with the lowest process percentage usage."""

    # Query GPU index and utilization, format as CSV
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True
    )
    
    gpus = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        index, usage = line.split(',')
        gpus.append({'id': int(index.strip()), 'usage': int(usage.strip())})
    
    if not gpus:
        print("No GPUs found by nvidia-smi.")
        return None
        
    # Sort by usage (ascending) and then by ID (descending, for tie-breaking)
    gpus.sort(key=lambda x: (x['usage'], -x['id']))
    
    # Select the top N GPUs
    selected_gpus = gpus[:num_gpus]
    selected_ids = [gpu['id'] for gpu in selected_gpus]
    
    if not selected_ids:
        print("No GPUs were selected.")
        return []

    print(f"Found {len(gpus)} GPUs. Selected {len(selected_ids)} GPUs: {selected_ids}")
    return selected_ids


def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render(file_path, sha256, output_dir, num_views, save_depth=False, save_normal=False, save_mask=False, gpu_id=None):
    output_folder = os.path.join(output_dir, 'renders', sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views # 半径全部都设置为2
    fov = [40 / 180 * np.pi] * num_views # fov全部都设置为40度，再转换为弧度
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views), # json.dumps() 将Python对象序列化为JSON字符串，将Python数据结构传递给Blender脚本
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--no_geo',
        '--save_mesh',
    ]
    
    if gpu_id is not None:
        # args.extend(['--gpu_id', str(gpu_id)])
        args.extend(['--gpu_ids', ','.join(map(str, gpu_id))])

    if save_depth:
        args.append('--save_depth')
        
    if save_normal:
        args.append('--save_normal')
    
    if save_mask:
        args.append('--save_mask')
        
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    # Uncomment the following line to see the command being run
    # print("Running command: " + ' '.join(args), flush=True)
    
    # Uncomment the following line to view output
    call(args)
    # call(args, stdout=DEVNULL, stderr=DEVNULL)
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'rendered': True}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}') # 动态导入不同模块，例如 datasets.ABO

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')# e.g. datasets/ObjaverseXL_sketchfab
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--save_depth', action='store_true',
                        help='Save depth maps during rendering')
    parser.add_argument('--save_normal', action='store_true',
                        help='Save normal maps during rendering')
    parser.add_argument('--save_mask', action='store_true',
                        help='Save object masks during rendering')
    
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt)) # 将命令行参数转换为易于访问的属性字典

    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()
    
    best_gpu_id = _get_best_gpu()

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' in metadata.columns:
            metadata = metadata[metadata['rendered'] == False] # 筛选出'rendered'列值为False的所有行
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

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_render, 
                   output_dir=opt.output_dir, 
                   num_views=opt.num_views,
                   save_depth=opt.save_depth, 
                   save_normal=opt.save_normal,
                   save_mask=opt.save_mask,
                   gpu_id=best_gpu_id)
    rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)
