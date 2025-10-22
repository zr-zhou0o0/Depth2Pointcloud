"""
Add Geometric Primitives to 3D Objects

This script processes 3D objects from a dataset and adds random geometric primitives 
(cubes, cylinders, spheres) to them using Blender. The primitives are added as children
of an empty object named "random_geometry" and the result is saved as a GLB file.
"""

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


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _add_geo_primitives(file_path, sha256, **kwargs):
    """
    Add geometric primitives to a 3D object using Blender
    
    Args:
        file_path: Path to the original 3D object file
        sha256: SHA256 hash of the object
        **kwargs: Additional arguments including output_dir, num_primitives_min, num_primitives_max, primitive_types, use_time_seed
    
    Returns:
        dict: Result dictionary with sha256 and processed status
    """
    output_dir = kwargs.get('output_dir')
    num_primitives_min = kwargs.get('num_primitives_min', 2)
    num_primitives_max = kwargs.get('num_primitives_max', 8)
    primitive_types = kwargs.get('primitive_types', ['cube', 'cylinder', 'sphere'])
    use_time_seed = kwargs.get('use_time_seed', False)
    
    # Create output directories (simplified: no bucketing)
    output_folder = os.path.join(output_dir, 'geo', 'glbs')
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = os.path.join(output_folder, f'{sha256}.glb')
    
    # Skip if already processed
    if os.path.exists(output_file):
        return {'sha256': sha256, 'processed': True, 'output_path': os.path.relpath(output_file, output_dir)}
    
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'add_geometry.py'),
        '--',
        '--object', os.path.expanduser(file_path),
        '--output_file', output_file,
        '--num_primitives_min', str(num_primitives_min),
        '--num_primitives_max', str(num_primitives_max),
        '--primitive_types', ','.join(primitive_types),
    ]
    
    # Add randomness strategy arguments
    if use_time_seed:
        args.append('--use_time_seed')
    else:
        args.extend(['--sha256', sha256])  # Pass SHA256 hash for reproducible randomness
    
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    # Uncomment the following line to see the command being run
    # print("Running command: " + ' '.join(args), flush=True)
    
    # Run blender script
    result = call(args, stdout=DEVNULL, stderr=DEVNULL)
    # result = call(args)
    
    if result == 0 and os.path.exists(output_file):
        return {'sha256': sha256, 'processed': True, 'output_path': os.path.relpath(output_file, output_dir)}
    else:
        return {'sha256': sha256, 'processed': False, 'output_path': None}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the processed objects')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_primitives', type=str, default='2,8',
                        help='Number range of geometric primitives to add (format: min,max)')
    parser.add_argument('--primitive_types', type=str, default='cube,cylinder,sphere',
                        help='Comma-separated list of primitive types')
    parser.add_argument('--use_time_seed', action='store_true',
                        help='Use current time as random seed for true randomness (non-reproducible)')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # Parse primitive types
    primitive_types = [t.strip() for t in opt.primitive_types.split(',')]
    
    # Parse number of primitives range
    try:
        num_primitives_range = [int(x.strip()) for x in opt.num_primitives.split(',')]
        if len(num_primitives_range) == 1:
            # If only one number provided, use it as both min and max
            num_primitives_range = [num_primitives_range[0], num_primitives_range[0]]
        elif len(num_primitives_range) != 2:
            raise ValueError("num_primitives should be either a single number or min,max")
        
        if num_primitives_range[0] > num_primitives_range[1]:
            raise ValueError("min should be <= max in num_primitives range")
            
        opt.num_primitives_min, opt.num_primitives_max = num_primitives_range
    except ValueError as e:
        print(f"Error parsing num_primitives: {e}")
        print("Using default range: 5,8")
        opt.num_primitives_min, opt.num_primitives_max = 2, 8
    
    # Create output directories
    os.makedirs(os.path.join(opt.output_dir, 'geo'), exist_ok=True)
    
    # Install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # Get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'geo_processed' in metadata.columns:
            metadata = metadata[metadata['geo_processed'] == False]
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
        output_file = os.path.join(opt.output_dir, 'geo', 'glbs', f'{sha256}.glb')
        if os.path.exists(output_file):
            records.append({
                'sha256': sha256, 
                'geo_processed': True, 
                'geo_output_path': os.path.relpath(output_file, opt.output_dir)
            })
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # Process objects
    func = partial(_add_geo_primitives, 
                   output_dir=opt.output_dir, 
                   num_primitives_min=opt.num_primitives_min,
                   num_primitives_max=opt.num_primitives_max,
                   primitive_types=primitive_types,
                   use_time_seed=opt.use_time_seed)
    
    processed = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Adding geometric primitives')
    
    # Rename columns to match expected format
    processed = processed.rename(columns={
        'processed': 'geo_processed',
        'output_path': 'geo_output_path'
    })
    
    processed = pd.concat([processed, pd.DataFrame.from_records(records)])
    processed.to_csv(os.path.join(opt.output_dir, f'geo_processed_{opt.rank}.csv'), index=False)
