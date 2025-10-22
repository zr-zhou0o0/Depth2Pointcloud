'''
An example code for render with blender under windows.
'''


"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
from math import radians
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
import yaml

import bpy
from mathutils import Vector


bpy.context.scene.cycles.device = "GPU"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
# parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--camera_dist", type=int, default=1.2)
parser.add_argument("--render_test_post", action="store_true", help="render test post")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
parser.add_argument("--continue_sequence", type=str, default=None, help="e.g. pre, post-001, post-002...")
parser.add_argument("--continue_uid", type=int, default=None, help="continue from which uid")
parser.add_argument("--log", type=str, default="render_log.log", help="log file path")
    
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')


# context: current scene, active object, view layer, area(工作区), ...
context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"] # ??? the name have to be 'camera'?
cam.location = (0, 1.2, 0)
cam.data.lens = 35 # focal length 
cam.data.sensor_width = 32
cam.data.sensor_height = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z" # -Z axis points to the target
cam_constraint.up_axis = "UP_Y" # Y axis is the up axis

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 384
render.resolution_y = 384
# render.resolution_x = 1600
# render.resolution_y = 1200
render.resolution_percentage = 100

scene.cycles.device = "GPU"
# scene.cycles.device = "CPU"
scene.cycles.samples = 256 # sample number
scene.cycles.diffuse_bounces = 1 # 漫反射光反射次数
scene.cycles.glossy_bounces = 1 # 镜面反射光反射次数
scene.cycles.transparent_max_bounces = 3 # 透明物体最大反射次数
scene.cycles.transmission_bounces = 3 # 透射光线最大反射次数
scene.cycles.filter_width = 0.01 # 滤波宽度
scene.cycles.use_denoising = True # 启用去噪
scene.render.film_transparent = True # 透明背景


import logging

if not os.path.exists(os.path.dirname(args.log)):
    os.makedirs(os.path.dirname(args.log))

logging.basicConfig(
    # filename=r"D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\002-desk\outputs\all\render_log.log",  # Log file name (you can change the path as needed)
    filename=args.log,  
    level=logging.INFO,        # Log level (DEBUG, INFO, WARNING, ERROR)
    # format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    format="%(message)s",  
)

logger = logging.getLogger()




def reset_lighting() -> None:
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 2000
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 10
    bpy.data.objects["Area"].location[2] = 0
    bpy.data.objects["Area"].rotation_euler[0] = radians(270)
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100

    bpy.ops.object.light_add(type="AREA")
    light3 = bpy.data.lights["Area.001"]
    light3.energy = 2000
    bpy.data.objects["Area.001"].location[0] = 0
    bpy.data.objects["Area.001"].location[1] = -10
    bpy.data.objects["Area.001"].location[2] = 0
    bpy.data.objects["Area.001"].rotation_euler[0] = radians(90)
    bpy.data.objects["Area.001"].scale[0] = 100
    bpy.data.objects["Area.001"].scale[1] = 100
    bpy.data.objects["Area.001"].scale[2] = 100

    bpy.ops.object.light_add(type="AREA")
    light4 = bpy.data.lights["Area.002"]
    light4.energy = 2000
    bpy.data.objects["Area.002"].location[0] = 10
    bpy.data.objects["Area.002"].location[1] = 0
    bpy.data.objects["Area.002"].location[2] = 0
    bpy.data.objects["Area.002"].rotation_euler[1] = radians(90)
    bpy.data.objects["Area.002"].scale[0] = 100
    bpy.data.objects["Area.002"].scale[1] = 100
    bpy.data.objects["Area.002"].scale[2] = 100

    bpy.ops.object.light_add(type="AREA")
    light5 = bpy.data.lights["Area.003"]
    light5.energy = 2000
    bpy.data.objects["Area.003"].location[0] = -10
    bpy.data.objects["Area.003"].location[1] = 0
    bpy.data.objects["Area.003"].location[2] = 0
    bpy.data.objects["Area.003"].rotation_euler[1] = radians(270)
    bpy.data.objects["Area.003"].scale[0] = 100
    bpy.data.objects["Area.003"].scale[1] = 100
    bpy.data.objects["Area.003"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    '''
    compute all objects bounding box in the scene
    '''
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        if obj.name == 'dome':
            continue
        print(obj.name)
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def if_occlude(camera, scene):
    # XXX
    # SET
    # room_bbox_x = (-3.2, 8)
    room_bbox_x = (-8, 8)
    room_bbox_y = (-6, 6)
    room_bbox_z = (-3, 3)
    camera_locat = camera.location

    if camera_locat[0] < room_bbox_x[0] or camera_locat[0] > room_bbox_x[1] or camera_locat[1] < room_bbox_y[0] or camera_locat[1] > room_bbox_y[1] or camera_locat[2] < room_bbox_z[0] or camera_locat[2] > room_bbox_z[1]:
        return True
    else:
        return False


def get_all_cd_center():
    # all_center_array = np.zeros((3, len(bpy.data.objects)))
    # all_room_object = []
    neccessary_uid_list = []

    unchange_collection = bpy.context.scene.collection.children['unchange']
    pre_collection = bpy.context.scene.collection.children['pre']
    base_collection = bpy.context.scene.collection.children['base']

    all_objs = []
    for obj in unchange_collection.objects: 
        all_objs.append(obj)
    for obj in pre_collection.objects:
        all_objs.append(obj)
    # for obj in base_collection.objects:
    #     all_objs.append(obj)
    
    # for i, obj in enumerate(all_objs):
    #     if obj.name.startswith('cd_'):
    #         # all_center_array[:, i] = obj.location
    #         all_room_object.append(obj)
    #         obj["uid"] = int(obj.name.split('_')[1])
    #         neccessary_uid_list.append(obj["uid"])
    #     else:
    #         obj["uid"] = None

    all_center_array = np.ones((4, len(all_objs)))
    for i, obj in enumerate(all_objs):
        obj["oid"] = i
        all_center_array[0:3, i] = obj.location
        neccessary_uid_list.append(i)


    return all_center_array, all_objs, neccessary_uid_list


def get_all_center():
    

    unchange_collection = bpy.context.scene.collection.children['unchange']
    pre_collection = bpy.context.scene.collection.children['pre']
    base_collection = bpy.context.scene.collection.children['base']

    all_objs = []
    for obj in unchange_collection.objects: 
        all_objs.append(obj)
    for obj in pre_collection.objects:
        all_objs.append(obj)
    for obj in base_collection.objects:
        all_objs.append(obj)

    all_center_array = np.ones((4, len(all_objs)))
    for i, obj in enumerate(all_objs):
        obj["oid"] = i
        all_center_array[0:3, i] = obj.location

    # for i, obj in enumerate(all_objs):
    #     all_center_array[:, i] = obj.location
    #     all_room_object.append(obj)

        # if obj.name.startswith('cd_'):
        #     obj["uid"] = int(obj.name.split('_')[1])
        # else:
        #     obj["uid"] = None

    return all_center_array, all_objs


def scene_root_objects():
    '''
    return objs without parent
    '''
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    '''
    return all mesh objects
    '''
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
#camera 2 world
#up vector is (0, 0, 1)
def get_3x4_RT_matrix_from_blender(cam):

    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT
#return True means z is the biggest
def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    print('scale:', scale)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    offset[2] = -bbox_min[2]
    print('offset:', offset)
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def render_mask_pre_cd(mask_path, mask_dir):
    '''
    object masks with occlusion, only for objects whose names start with "cd_"
    '''
    # 只选择名称以"cd-"开头的物体
    # mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and obj.name.startswith('cd_')]
    tot_objects = [obj for obj in bpy.context.scene.objects]

    pre_collection = bpy.data.collections['pre']
    mesh_objects_parents = [obj for obj in pre_collection.objects if obj.type == 'MESH' and obj.name.startswith('cd_') and not obj.children]
    mesh_objects_all = [obj for obj in pre_collection.objects if obj.type == 'MESH' and obj.name.startswith('cd_')]

    print("pre mask rendering: mesh_objects_parents:", mesh_objects_parents)
    print("pre mask rendering: mesh_objects_all:", mesh_objects_all)

    max_passidx = 0
    for i in range(len(mesh_objects_all)):
        # mesh_objects[i].pass_index = i + 1  # 渲染时的概念，用来区分不同物体
        if mesh_objects_all[i].parent:
            name = mesh_objects_all[i].parent.name
            objid = int(name.split('_')[1])
            mesh_objects_all[i].pass_index = objid
            max_passidx = max(max_passidx, objid)
        else:
            name = mesh_objects_all[i].name
            objid = int(name.split('_')[1])
            mesh_objects_all[i].pass_index = objid
            max_passidx = max(max_passidx, objid)
    
    # 获取视图层
    view_layer = bpy.context.scene.view_layers[0]

    # 配置视图层以使用Object Index
    view_layer.use_pass_object_index = True
    view_layer.use_pass_z = True  # depth
    view_layer.use_pass_normal = True  # normal

    # 设置合成节点
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # 清除默认节点
    for node in tree.nodes:
        tree.nodes.remove(node)

    # 创建节点
    render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')  # 渲染层节点
    output_file_node_masks = tree.nodes.new('CompositorNodeOutputFile')  # 创建文件输出节点（用于保存掩码）
    # output_file_node_image = tree.nodes.new('CompositorNodeOutputFile')  # 创建文件输出节点（用于保存原图）
    composite_node = tree.nodes.new(type='CompositorNodeComposite')  # 创建合成节点（用于将不同的图层合成在一起）

    # 设置文件输出路径
    output_file_node_masks.base_path = mask_dir  # 替换为实际的掩码保存路径
    # output_file_node_image.base_path = root_dir  # 替换为实际的图像保存路径
    # output_file_node_image.file_slots[0].path = whole_path  # 设置原图的文件名

    # 连接节点以输出原图
    tree.links.new(render_layers_node.outputs['Image'], composite_node.inputs['Image'])  # 用于Blender内显示
    # tree.links.new(render_layers_node.outputs['Image'], output_file_node_image.inputs[0])  # 用于保存原图

    # 确保为每个对象创建足够的文件槽
    # while len(output_file_node_masks.file_slots) < len(mesh_objects):
    while len(output_file_node_masks.file_slots) < max_passidx + 1:
        output_file_node_masks.file_slots.new("")
        

    # hide_idx = []
    for idx, obj in enumerate(mesh_objects_parents):
        
        id_mask_node = tree.nodes.new(type='CompositorNodeIDMask')
        id_mask_node.index = obj.pass_index 

        # if obj.hide_render:
        #     obj.hide_render = False
        #     hide_idx.append(obj.pass_index)
        #     print(f"unhide {obj.name}")

        # 连接渲染层节点的对象索引输出到ID掩码节点的输入
        tree.links.new(render_layers_node.outputs['IndexOB'], id_mask_node.inputs['ID value'])
        # 连接ID掩码节点的Alpha输出到文件输出节点的输入
        tree.links.new(id_mask_node.outputs['Alpha'], output_file_node_masks.inputs[obj.pass_index])
        # 设置掩码的文件路径
        output_file_node_masks.file_slots[obj.pass_index].path = mask_path + f"_{obj.pass_index:03d}_#"


    bpy.ops.render.render(write_still=True)
    # bpy.ops.render.render(write_still=False)

    # for idx, obj in enumerate(mesh_objects):
    #     if obj.pass_index in hide_idx:
    #         obj.hide_render = True


def render_mask_post_cd(mask_path, mask_dir, pre_name, post_name):
    '''
    object masks with occlusion, only for objects whose names start with "cd_"
    '''
    # mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and obj.name.startswith('cd_')]
    # tot_objects = [obj for obj in bpy.context.scene.objects]

    # pre_collection = bpy.data.collections[pre_name]
    post_collection = bpy.data.collections[post_name]

    # pre_objects = [obj for obj in pre_collection.objects if obj.type == 'MESH' and obj.name.startswith('cd_')]
    post_objects_parents = [obj for obj in post_collection.objects if obj.type == 'MESH' and obj.name.startswith('cd_') and not obj.children]
    post_objects_all = [obj for obj in post_collection.objects if obj.type == 'MESH' and obj.name.startswith('cd_')]


    max_passidx = 0
    # 为每个网格物体分配一个唯一的pass_index，用于渲染时区分不同物体
    for i in range(len(post_objects_all)):
        # mesh_objects[i].pass_index = i + 1  # 渲染时的概念，用来区分不同物体
        if post_objects_all[i].parent:
            name = post_objects_all[i].parent.name
            objid = int(name.split('_')[1])
            post_objects_all[i].pass_index = objid
            max_passidx = max(max_passidx, objid)
        else:
            name = post_objects_all[i].name
            objid = int(name.split('_')[1])
            post_objects_all[i].pass_index = objid
            max_passidx = max(max_passidx, objid)
            print("post mask rendering: post_object:", post_objects_all[i].name, "pass_index:", post_objects_all[i].pass_index)

   
    # 获取视图层
    view_layer = bpy.context.scene.view_layers[0]

    # 配置视图层以使用Object Index
    view_layer.use_pass_object_index = True
    view_layer.use_pass_z = True  # depth
    view_layer.use_pass_normal = True  # normal

    # 设置合成节点
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # 清除默认节点
    for node in tree.nodes:
        tree.nodes.remove(node)

    # 创建节点
    render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')  # 渲染层节点
    output_file_node_masks = tree.nodes.new('CompositorNodeOutputFile')  # 创建文件输出节点（用于保存掩码）
    # output_file_node_image = tree.nodes.new('CompositorNodeOutputFile')  # 创建文件输出节点（用于保存原图）
    composite_node = tree.nodes.new(type='CompositorNodeComposite')  # 创建合成节点（用于将不同的图层合成在一起）

    # 设置文件输出路径
    output_file_node_masks.base_path = mask_dir  # 替换为实际的掩码保存路径
    # output_file_node_image.base_path = root_dir  # 替换为实际的图像保存路径
    # output_file_node_image.file_slots[0].path = whole_path  # 设置原图的文件名

    # 连接节点以输出原图
    tree.links.new(render_layers_node.outputs['Image'], composite_node.inputs['Image'])  # 用于Blender内显示
    # tree.links.new(render_layers_node.outputs['Image'], output_file_node_image.inputs[0])  # 用于保存原图

    # 确保为每个对象创建足够的文件槽
    # while len(output_file_node_masks.file_slots) < len(post_objects):
    while len(output_file_node_masks.file_slots) < max_passidx + 1:
        output_file_node_masks.file_slots.new("")
    

    # hide_idx = []
    for idx, obj in enumerate(post_objects_parents):
        
        id_mask_node = tree.nodes.new(type='CompositorNodeIDMask')
        id_mask_node.index = obj.pass_index 

        # 连接渲染层节点的对象索引输出到ID掩码节点的输入
        tree.links.new(render_layers_node.outputs['IndexOB'], id_mask_node.inputs['ID value'])
        # 连接ID掩码节点的Alpha输出到文件输出节点的输入
        tree.links.new(id_mask_node.outputs['Alpha'], output_file_node_masks.inputs[obj.pass_index])
        # 设置掩码的文件路径
        output_file_node_masks.file_slots[obj.pass_index].path = mask_path + f"_{obj.pass_index:03d}_#"


    bpy.ops.render.render(write_still=True)

  



def render_whole_cd(whole_path, root_dir):
    
    # 获取视图层
    view_layer = bpy.context.scene.view_layers[0]

    # 配置视图层以使用Object Index
    view_layer.use_pass_object_index = True
    view_layer.use_pass_z = True  # depth
    view_layer.use_pass_normal = True  # normal

    # 设置合成节点
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # 清除默认节点
    for node in tree.nodes:
        tree.nodes.remove(node)

    # 创建节点
    render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')  # 渲染层节点
    # output_file_node_masks = tree.nodes.new('CompositorNodeOutputFile')  # 创建文件输出节点（用于保存掩码）
    output_file_node_image = tree.nodes.new('CompositorNodeOutputFile')  # 创建文件输出节点（用于保存原图）
    composite_node = tree.nodes.new(type='CompositorNodeComposite')  # 创建合成节点（用于将不同的图层合成在一起）

    # 设置文件输出路径
    # output_file_node_masks.base_path = mask_dir  # 替换为实际的掩码保存路径
    output_file_node_image.base_path = root_dir  # 替换为实际的图像保存路径
    output_file_node_image.file_slots[0].path = whole_path  # 设置原图的文件名

    # 连接节点以输出原图
    tree.links.new(render_layers_node.outputs['Image'], composite_node.inputs['Image'])  # 用于Blender内显示
    tree.links.new(render_layers_node.outputs['Image'], output_file_node_image.inputs[0])  # 用于保存原图

    bpy.ops.render.render(write_still=True)


def render_whole_post_cd(whole_path, root_dir, pre_name, post_name):
    
    pre_collection = bpy.data.collections[pre_name]
    post_collection = bpy.data.collections[post_name]

    pre_objects_parents = [obj for obj in pre_collection.objects if obj.type == 'MESH' and obj.name.startswith('cd_') and not obj.children]
    pre_objects_all = [obj for obj in pre_collection.objects if obj.type == 'MESH' and obj.name.startswith('cd_')]

    post_objects_parents = [obj for obj in post_collection.objects if obj.type == 'MESH' and obj.name.startswith('cd_') and not obj.children]
    post_objects_all = [obj for obj in post_collection.objects if obj.type == 'MESH' and obj.name.startswith('cd_')]

    for i in range(len(pre_objects_all)):
        if pre_objects_all[i].parent:
            name = pre_objects_all[i].parent.name
        else:
            name = pre_objects_all[i].name

        objid = int(name.split('_')[1])
        pre_objects_all[i].pass_index = objid
    
    for i in range(len(post_objects_all)):
        if post_objects_all[i].parent:
            name = post_objects_all[i].parent.name
        else:
            name = post_objects_all[i].name

        objid = int(name.split('_')[1])
        post_objects_all[i].pass_index = objid
    
    pre_objects_idx = [obj.pass_index for obj in pre_objects_parents]
    post_objects_idx = [obj.pass_index for obj in post_objects_parents]

    pre_open_idx = []
    # post_hide_idx = []

    # 让pre的物体hide 让post的物体出现
    # 特例：如果pre本来就hide 那就不要动
    # 特例：如果post本来就hide 那也不要动
    # 因为post正常就是出现的 所以post都不用动
    # 由于articulated object都是放作子物体，因此必须是pre在外层循环 一次hide物体的所有部分
    # 不过其实对于whole 考不考虑父子都一样
    for i, preobj in enumerate(pre_objects_all):
        for j, postobj in enumerate(post_objects_parents):  
            if postobj.pass_index == preobj.pass_index:
                if not preobj.hide_render: # 如果本来就hide了，那就是in的物体
                    preobj.hide_render = True
                    pre_open_idx.append(preobj.pass_index)
        # 如果post的hide了 那就是out的物体 本来就不需要渲染
        # postobj.hide_render = False
        # post_hide_idx.append(postobj.pass_index)
        
   
    # 获取视图层
    view_layer = bpy.context.scene.view_layers[0]

    # 配置视图层以使用Object Index
    view_layer.use_pass_object_index = True
    view_layer.use_pass_z = True  # depth
    view_layer.use_pass_normal = True  # normal

    # 设置合成节点
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # 清除默认节点
    for node in tree.nodes:
        tree.nodes.remove(node)

    # 创建节点
    render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')  # 渲染层节点
    # output_file_node_masks = tree.nodes.new('CompositorNodeOutputFile')  # 创建文件输出节点（用于保存掩码）
    output_file_node_image = tree.nodes.new('CompositorNodeOutputFile')  # 创建文件输出节点（用于保存原图）
    composite_node = tree.nodes.new(type='CompositorNodeComposite')  # 创建合成节点（用于将不同的图层合成在一起）

    # 设置文件输出路径
    # output_file_node_masks.base_path = mask_dir  # 替换为实际的掩码保存路径
    output_file_node_image.base_path = root_dir  # 替换为实际的图像保存路径
    output_file_node_image.file_slots[0].path = whole_path  # 设置原图的文件名

    # 连接节点以输出原图
    tree.links.new(render_layers_node.outputs['Image'], composite_node.inputs['Image'])  # 用于Blender内显示
    tree.links.new(render_layers_node.outputs['Image'], output_file_node_image.inputs[0])  # 用于保存原图

    bpy.ops.render.render(write_still=True)

    for i, preobj in enumerate(pre_objects_all):
        if preobj.pass_index in pre_open_idx:
            preobj.hide_render = False


def get_camera_intrinsics(camera, scene):
    # Get camera parameters
    f_in_mm = camera.data.lens  # focal length in mm
    sensor_width_in_mm = camera.data.sensor_width  # sensor width in mm
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100

    # Calculate fx and fy
    fx = (f_in_mm / sensor_width_in_mm) * resolution_x_in_px * scale
    fy = (f_in_mm / sensor_width_in_mm) * resolution_y_in_px * scale

    # Principal point
    cx = resolution_x_in_px * scale / 2
    cy = resolution_y_in_px * scale / 2

    # Intrinsic matrix
    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]
    
    return K

def initialize_scene(scene):
    scene.cycles.device = "GPU"
    # scene.cycles.device = "CPU"
    scene.cycles.samples = 256
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene = bpy.data.scenes[0]


class Cameras:
    def __init__(self, config):

        try:
            self.camera = bpy.data.objects["Camera"]
        except:
            self.camera = None

        self.config = config

        self.root_dir = config["basic"]["root_dir"]
        self.num_layers = config["basic"]["layer_num"] if "layer_num" in config["basic"] else None
        self.view_num = config["basic"]["view_num"] if "view_num" in config["basic"] else None

        self.view_num_all = config["navigation"]["view_num_all"] if "view_num_all" in config["navigation"] else None
        self.pre_camera_name = config["navigation"]["pre_camera_name"] if "pre_camera_name" in config["navigation"] else None
        self.post_camera_name = config["navigation"]["post_camera_name"] if "post_camera_name" in config["navigation"] else None

        self.resolution_x = config["basic"]["resolution_x"]
        self.resolution_y = config["basic"]["resolution_y"]

        self.sequence = "pre"
        self.pre_name = "pre"
        self.post_name = None
        self.post_num = config["basic"]["post_num"]

        self.cameras_json_pre = {}
        self.cameras_json_post = {}

        self.continue_uid = None


    def save_cameras_json(self, path, sequence):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            if sequence == "pre":
                json.dump(self.cameras_json_pre, f)
            else:
                json.dump(self.cameras_json_post, f)


    def clear_cameras_json(self):
        # self.cameras_json = {}
        self.cameras_json_pre = {}
        self.cameras_json_post = {}
        logger.info("\ncameras_json cleared")


    def get_navigation_cameras(self, camera_name, view_number, sequence):
        object_to_export = bpy.data.objects[camera_name]

        if sequence == "pre":
            cameras_json = self.cameras_json_pre
        elif sequence == "post":
            cameras_json = self.cameras_json_post
        else:
            raise ValueError("Invalid sequence")

        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end
        space = (frame_end - frame_start) // view_number 

        for i in range(view_number):
            frame = frame_start + i * space if i * space <= (frame_end - frame_start) else frame_end
            bpy.context.scene.frame_set(frame)

            rotation_euler = object_to_export.rotation_euler
            location = object_to_export.location

            rot = (rotation_euler[0], rotation_euler[1], rotation_euler[2])
            loc = (location[0], location[1], location[2])

            cameras_json[i] = {
                "uid": i,
                "frame": frame,
                "rotation": rot,
                "location": loc
            }

        print("Get rotation and location data")


    def random_sample_sphere(self, center, radius):
        r = radius * (random.random() ** (1/3))  # 立方根确保点均匀分布在球体内部
        
        theta = random.uniform(0, 2 * math.pi) 
        # phi = random.uniform(0, math.pi)  
        phi = math.acos(2 * random.random() - 1) # arccos -1，1 垂直角的余角

        
        x = center[0] + r * math.sin(phi) * math.cos(theta)
        y = center[1] + r * math.sin(phi) * math.sin(theta)
        z = center[2] + r * math.cos(phi)
        
        return (x, y, z)


    def set_camera_location(self, emin, emax, radius_min, radius_max, azimuth, center, look_out):

        elevation = random.uniform(emin, emax)
        distance = random.uniform(radius_min, radius_max)

        x = distance * math.cos(math.radians(elevation)) * math.cos(math.radians(azimuth))
        y = distance * math.cos(math.radians(elevation)) * math.sin(math.radians(azimuth))
        z = distance * math.sin(math.radians(elevation))
        camera = bpy.data.objects["Camera"]
        camera.location = x, y, z

        if look_out:
            direction = camera.location
        else:
            direction = - camera.location 

        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        camera.location = x + center[0], y + center[1], z + center[2]
        print(camera.location)

        logger.info(f"Elevation: {elevation}")
        logger.info(f"Distance: {distance}")
        logger.info(f"Azimuth: {azimuth}")
        logger.info(f"center: {center}")
        logger.info(f"Camera location: {camera.location}")
        # breakpoint()
        return camera


    def random_render_layer(self, layer_index, center, center_radius, radius_min, radius_max, emin, emax, azimuth_min, azimuth_max, look_out=True):    
        for i in range(self.view_num):

            uid = layer_index * self.view_num + i
            print(f"Rendering image {uid}")
            logger.info(f"\nRendering image {uid}")

            if self.continue_uid is not None and uid < self.continue_uid:
                print(f"Skip image {uid}")
                logger.info(f"Skip image {uid}")
                continue

            # azimuth = 360 / self.view_num * i
            # azimuth = random.uniform(azimuth_min, azimuth_max)
            azimuth_indent = (azimuth_max - azimuth_min) / self.view_num
            azimuth = azimuth_min + azimuth_indent * i

            # if azimuth < azimuth_min or azimuth > azimuth_max:
            #     continue

            # TODO
            center = self.random_sample_sphere(center, center_radius)

            camera = self.set_camera_location(emin, emax, radius_min, radius_max, azimuth, center, look_out)
            print(f"Camera location: {camera.location}")
            print(f"Azimuth: {azimuth}")

            # # discard
            # occlusion = self.if_occlusion()
            # if occlusion:
            #     print("Occlusion, skip")
            #     logger.info("Occlusion, skip")
            #     # continue # XXX
            # else:
            #     print("No occlusion")
            #     logger.info("No occlusion")

            if if_occlude(camera, bpy.context.scene):
                print("Occlusion, skip")
                logger.info("Occlusion, skip")
                continue
            else:
                print("No occlusion")
                logger.info("No occlusion")

            root_dir = self.root_dir
            mask_dir = os.path.join(root_dir, f"{uid:03d}", "pre_gt")
            whole_path = f"whole_{uid:03d}_#"
            mask_path = f"gt_{uid:03d}"

            render_mask = self.config["basic"]["render_mask"]

            if self.sequence == "pre":

                if render_mask:
                    render_mask_pre_cd(mask_path, mask_dir)

                render_whole_cd(whole_path, root_dir)
                pre_collection = bpy.data.collections[self.pre_name]
                pre_collection.hide_render = True
                pre_collection.hide_select = True

                if render_mask:
                    for i in range(self.post_num):
                        post_name = f"post-{i+1:03d}"
                        mask_dir = os.path.join(root_dir, f"{uid:03d}", post_name + "_gt")
                        try:
                            post_collection = bpy.data.collections[post_name]
                            post_collection.hide_render = False
                            post_collection.hide_select = False
                        except:
                            print(f"{post_name} not found")
                            continue
                        render_mask_post_cd(mask_path, mask_dir, self.pre_name, post_name)
                        post_collection.hide_render = True
                        post_collection.hide_select = True

                pre_collection.hide_render = False
                pre_collection.hide_select = False

            elif self.sequence == "post":
                # render_mask_post_cd(mask_path, mask_dir, self.pre_name, self.post_name)
                render_whole_post_cd(whole_path, root_dir, self.pre_name, self.post_name)
            else:
                raise ValueError("Sequence not assigned")


    def random_render(self):
        '''
        render n layers of randomized spherical cameras
        '''
        for i in range(self.num_layers):
            layer_name = f"layer_{i}"
            logger.info(f"\nRendering layer {layer_name}")
            try: 
                center = self.config[layer_name]["center"]
            except:
                print(f"Skip layer {layer_name}")
                logger.info(f"Skip layer {layer_name}")
                continue
            
            center_radius = self.config[layer_name]["center_radius"]
            emin = self.config[layer_name]["elevation_min"]
            emax = self.config[layer_name]["elevation_max"]
            dist_scale_min = self.config[layer_name]["disctance_scale_min"]
            dist_scale_max = self.config[layer_name]["disctance_scale_max"]
            look_out = self.config[layer_name]["look_out"]
            azimuth_min = self.config[layer_name]["azimuth_min"] if "azimuth_min" in self.config[layer_name] else 0
            azimuth_max = self.config[layer_name]["azimuth_max"] if "azimuth_max" in self.config[layer_name] else 360

            bbox = scene_bbox()
            length = math.sqrt(((bbox[1][0] - bbox[0][0]) / 2.0) ** 2 + ((bbox[1][1] - bbox[0][1]) / 2.0) ** 2 + ((bbox[1][2] - bbox[0][2]) / 2.0) ** 2)
            length = 10
            radius_min = length * dist_scale_min
            radius_max = length * dist_scale_max

            self.random_render_layer(i, center, center_radius, radius_min, radius_max, emin, emax, azimuth_min, azimuth_max, look_out)


    def navigation_render(self):
        '''
        render cameras from navigation animation keyframes
        '''

        logger.info(f"\nRendering navigation camera")

        # context = bpy.context
        # scene = context.scene
        # render = scene.render

        render_mask = self.config["basic"]["render_mask"]
        root_dir = self.root_dir
        
        sequence = self.sequence
        if sequence == "pre":
            camera_name = self.pre_camera_name
            cameras = self.cameras_json_pre
        elif sequence == "post":
            camera_name = self.post_camera_name
            cameras = self.cameras_json_post
        else:
            raise ValueError("Invalid sequence")

        # camera = bpy.data.objects[camera_name]
        # scene.camera = camera
        # self.camera = camera
        self.get_navigation_cameras(camera_name, self.view_num_all, sequence)
        self.save_cameras_json(os.path.join(root_dir, f"cameras_{sequence}.json"), sequence)

        for uid, data in cameras.items():

            if self.continue_uid is not None and uid < self.continue_uid:
                print(f"Skip image {uid}")
                logger.info(f"Skip image {uid}")
                continue

            rotation_euler = data["rotation"]
            location = data["location"]
            frame = data["frame"]

            camera = bpy.data.objects["Camera"]

            camera.rotation_euler = rotation_euler
            camera.location = location

            logger.info(f"\nRendering image {uid}")
            logger.info(f"Frame: {frame}")
            logger.info(f"Rotation: {rotation_euler}")
            logger.info(f"Location: {location}")

            if if_occlude(camera, bpy.context.scene):
                print("Occlusion, skip")
                logger.info("Occlusion, skip")
                continue
            else:
                print("No occlusion")
                logger.info("No occlusion")

            mask_dir = os.path.join(root_dir, f"{uid:03d}", "pre_gt")
            whole_path = f"whole_{uid:03d}_#"
            mask_path = f"gt_{uid:03d}"
            
            if self.sequence == "pre":

                if render_mask:
                    render_mask_pre_cd(mask_path, mask_dir)

                render_whole_cd(whole_path, root_dir)
                pre_collection = bpy.data.collections[self.pre_name]
                pre_collection.hide_render = True
                pre_collection.hide_select = True

                if render_mask:
                    for i in range(self.post_num):
                        post_name = f"post-{i+1:03d}"
                        mask_dir = os.path.join(root_dir, f"{uid:03d}", post_name + "_gt")
                        try:
                            post_collection = bpy.data.collections[post_name]
                            post_collection.hide_render = False
                            post_collection.hide_select = False
                        except:
                            print(f"{post_name} not found")
                            continue
                        render_mask_post_cd(mask_path, mask_dir, self.pre_name, post_name)
                        post_collection.hide_render = True
                        post_collection.hide_select = True

                pre_collection.hide_render = False
                pre_collection.hide_select = False

            elif self.sequence == "post":
                # render_mask_post_cd(mask_path, mask_dir, self.pre_name, self.post_name)
                render_whole_post_cd(whole_path, root_dir, self.pre_name, self.post_name)
            else:
                raise ValueError("Sequence not assigned")
                




def render_all(config, continue_seq, continue_uid):

    post_num = config["basic"]["post_num"]
    camera_type = config["basic"]["camera"]
    collections = bpy.context.scene.collection.children
    
    for collection in collections:
        collection.hide_render = True
        collection.hide_select = True
        if collection.name == 'pre':
            pre_scene = collection
            collection.hide_render = False
            collection.hide_select = False
        elif collection.name == 'base':
            base_scene = collection
            collection.hide_render = False
            collection.hide_select = False
        elif collection.name == 'unchange':
            unchange_scene = collection
            collection.hide_render = False
            collection.hide_select = False
        # elif collection.name.startswith('post'):
        #     collection.hide_render = False
        #     collection.hide_select = False

    root_dir = config["basic"]["root_dir"]
    if continue_seq == None or continue_seq == "pre":
        cameras = Cameras(config)
        cameras.sequence = "pre"
        cameras.pre_name = "pre"
        cameras.continue_uid = continue_uid
        pre_root_dir = os.path.join(root_dir, 'pre')
        cameras.root_dir = pre_root_dir
        if camera_type == "random":
            cameras.random_render()
        elif camera_type == "navigation":
            cameras.navigation_render()
        else:
            raise ValueError("Invalid camera type")
    
    if continue_seq == None or continue_seq == "pre":
        continue_i = 1
    elif continue_seq.startswith("post"):
        continue_i = int(continue_seq.split('-')[1])
    else:
        continue_i = 1
    
    for i in range(post_num): 
        idx = i+1 # 1~10

        if idx < continue_i:
            logger.info(f"\nSkip post-{idx:03d}")
            continue

        post_scene = None
        for col in collections:
            if col.name == f'post-{idx:03d}':
                post_scene = col
        
        if post_scene is None:
            print(f"post-{idx:03d} not found")
            continue

        if not post_scene.objects:
            print(f"post-{idx:03d} is empty")
            continue
        
        post_scene.hide_render = False
        post_scene.hide_select = False

        cameras = Cameras(config)
        cameras.sequence = "post"
        cameras.post_name = f"post-{idx:03d}"
        post_root_dir = os.path.join(root_dir, f'post-{idx:03d}')
        cameras.root_dir = post_root_dir

        if camera_type == "random":
            cameras.random_render()
        elif camera_type == "navigation":
            cameras.navigation_render()
        else:
            raise ValueError("Invalid camera type")

        post_scene.hide_render = True
        post_scene.hide_select = True



# gpt version
def main(obj_pth, config, continue_seq, continue_uid) -> None:
    """Saves rendered images of the object in the scene."""
    reset_scene()
    bpy.ops.wm.open_mainfile(filepath=obj_pth)
    context = bpy.context
    scene = context.scene
    render = scene.render

    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    
    render.resolution_x = config["basic"]["resolution_x"]
    render.resolution_y = config["basic"]["resolution_y"]
    render.resolution_percentage = 100

    initialize_scene(scene)

    # if config["basic"]["camera"] == "random":
    
    # reset_lighting()
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera

    # bbox = scene_bbox()
    empty = bpy.data.objects.new("Empty", None)
    empty.location = Vector((0.0, 0.0, 0.0))

    scene.collection.objects.link(empty)

    bpy.context.scene.render.use_lock_interface = True
    scene.render.use_lock_interface = True


    render_all(config, continue_seq, continue_uid)


# --- Supplementary Code for Test Set Rendering ---
# You can add the following functions to your script.

def render_special_test_set(config):
    """Renders a specific test set of images using pre-sequence cameras for post-sequence scenes."""
    logger.info("--- Starting Special Test Set Rendering ---")

    # test_ind_dict = { # livingroom
        # 'post-001': ['059', '023', '019', '000', '089'],
        # 'post-002': ['003', '013', '044', '070', '078'], 
        # 'post-003': ['005', '013', '019', '035'],
        # 'post-006': ['004', '022', '036', '099'],
    # }
    # test_ind_dict = {  # desk
    #     'post-001': ['006', '013', '025', '034', '077'],
        # 'post-002': ['003', '052', '082', '091'],
    #     'post-003': ['091', '085', '073', '041'],
    #     # 添加更多的post文件夹和对应的test indexes
    # }
    # test_ind_dict = {  # bookcase
    #     'post-001': ['000', '011', '031', '076', '090', '092'], # in 两本 out 三本
        # 'post-002': ['002', '008', '028', '037', '051'],
    # }
    # test_ind_dict = {  # bedroom
        # 'post-001': ['012', '039', '060', '073'], 
        # 'post-001': ['012', '030', '039', '060', '073', '113', '137', '157', '197'], 
        # 'post-001': ['197'], 
    #     'post-002': ['000', '034', '069', '099'],
    #     'post-003': ['010', '012', '039', '060', '073', '089'],
    # }

    test_ind_dict = { # livingroom_ctrl
        'post-001': ['000', '019', '043', '064', '071'],
        'post-002': ['000', '019', '043', '064', '071'],
        'post-003': ['000', '019', '043', '064', '071'],
        'post-004': ['003', '013', '044', '070', '078'], 
        'post-005': ['003', '011', '035', '070', '078'], 
        'post-006': ['003', '011', '035', '070', '078'], 
        'post-007': ['005', '013', '019', '035', '044'],
        'post-008': ['005', '012', '019', '022', '035'],
        'post-009': ['005', '012', '019', '022', '035'],
        'post-010': ['000', '006', '022', '035', '071'],
        'post-011': ['000', '006', '022', '035', '073'],
    }

    
    original_root_dir = config["basic"]["root_dir"]
    # Create a new directory for the test set output
    # new_root_dir = os.path.join(os.path.dirname(original_root_dir.rstrip('/\\')), "test-post")
    new_root_dir = os.path.join(original_root_dir, "test-post")
    pre_camera_json_path = os.path.join(original_root_dir, 'pre', 'cameras_pre.json')

    logger.info(f"Test set output directory: {new_root_dir}")
    logger.info(f"Loading pre-camera data from: {pre_camera_json_path}")

    if not os.path.exists(pre_camera_json_path):
        logger.error(f"Pre-camera JSON file not found at {pre_camera_json_path}")
        print(f"Error: Pre-camera JSON file not found at {pre_camera_json_path}")
        return

    with open(pre_camera_json_path, 'r') as f:
        pre_cameras_data = json.load(f)

    # Setup rendering environment
    scene = bpy.context.scene
    render = scene.render
    render.resolution_x = config["basic"]["resolution_x"]
    render.resolution_y = config["basic"]["resolution_y"]
    render.resolution_percentage = 100
    
    camera = bpy.data.objects["Camera"]
    collections = bpy.context.scene.collection.children

    # Hide all collections initially
    for collection in collections:
        collection.hide_render = True
    
    # Ensure base components are visible
    if 'base' in collections:
        collections['base'].hide_render = False
    if 'unchange' in collections:
        collections['unchange'].hide_render = False
    if 'pre' in collections:
        collections['pre'].hide_render = False # 'pre' objects should NOT be hidden!!!

    # Render each post scene with specified pre-cameras
    for post_name, uids in test_ind_dict.items():
        logger.info(f"--- Rendering for {post_name} ---")

        if post_name not in collections:
            logger.warning(f"Collection {post_name} not found, skipping.")
            continue
        
        collections[post_name].hide_render = False
        output_dir = os.path.join(new_root_dir, post_name)
        
        for uid_str in uids:
            uid_int = int(uid_str)
            
            if str(uid_int) not in pre_cameras_data:
                logger.warning(f"UID {uid_str} not found in pre_cameras_data, skipping.")
                continue
                
            cam_data = pre_cameras_data[str(uid_int)]
            
            camera.location = cam_data['location']
            camera.rotation_euler = cam_data['rotation']
            
            logger.info(f"  Rendering UID {uid_str} for {post_name}")

            whole_path = f"whole_{uid_int:03d}_#"
            # render_whole_cd(whole_path, output_dir)
            render_whole_post_cd(whole_path, output_dir, "pre", post_name)


        collections[post_name].hide_render = True

    logger.info("--- Finished Special Test Set Rendering ---")


def main_for_test_set(obj_pth, config):
    """Sets up the scene and runs the special test set rendering."""
    reset_scene()
    bpy.ops.wm.open_mainfile(filepath=obj_pth)
    context = bpy.context
    scene = context.scene
    render = scene.render

    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    
    initialize_scene(scene)

    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    scene.camera = bpy.data.objects["Camera"]
    
    render_special_test_set(config)


if __name__ == "__main__":
    start_i = time.time()
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # A simple check for a new flag to run the test set rendering
    # To run, add `--render_test_post` to your command line arguments.
    # if "--render_test_post" in sys.argv:
    if args.render_test_post:
        main_for_test_set(args.object_path, config)
    else:
        main(args.object_path, config, args.continue_sequence, args.continue_uid)
        
    end_i = time.time()
    print("Finished", "in", end_i - start_i, "seconds")






# Running scripts:
# /opt/blender-3.2.1-linux-x64/blender -b -P blender_script.py -- --object_path scene.blend

'''
1. camera info json
2. (whf?) nohup render
3. (optional) pre light and post without light
4. (wtf??????) occlusion automatic recognition
4. (ok) occlusion bbox judge / ray cast judge
5. occlusion post process

Livingroom:

D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\001-livingroom.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\config.yaml --continue_sequence post-001 --continue_uid 13 --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\outputs\all\log.log

D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\001-livingroom.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\config_nav.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\outputs\all_navigation_3\log.log 

# 新的测试版本
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script_v2.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\001-livingroom-v2.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\config_nav_v2.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\outputs\all_navigation_4\log.log 

# for test-post livingroom dataset
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script_v2_supp.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\001-livingroom-v2.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\config_nav_v2.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\outputs\all_navigation_4\log.log --render_test_post

# note: navigation2: focal len = 35 

# for test-post control dataset
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script_v2_supp.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\001-livingroom-v3.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\config_nav_v3.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\001-livingroom\outputs\ctrl_dataset\log.log --render_test_post




Desk:

# navigation
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\002-desk\desk.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\002-desk\config_nav.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\002-desk\outputs\all_navigation\log.log --continue_sequence post-004 --continue_uid 0

# 新的测试版本
# navigation
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script_v2.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\002-desk\desk_v2.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\002-desk\config_nav_v2.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\002-desk\outputs\all_navigation_v2\log.log

# test post 
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script_v2_supp.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\002-desk\desk_v2.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\002-desk\config_nav_v2.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\002-desk\outputs\all_navigation_v2\log.log --render_test_post



Bookcase:
1. 补一个全局的视角
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\003-bookcase\bookcase.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\003-bookcase\config.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\003-bookcase\outputs\all\log.log 

# navigation
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\003-bookcase\bookcase.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\003-bookcase\config_nav.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\003-bookcase\outputs\all_navigation\log.log 

# test post
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script_v2_supp.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\003-bookcase\bookcase.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\003-bookcase\config_nav_v2.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\003-bookcase\outputs\all_navigation\log.log --render_test_post



Bedroom:

1. 是因为设了时间上限吗？为什么会有随机光点？
2. 补充几个全局视角

# XXX explicitly set:
# set the occlusion x to -3.2! have to change back
# length to 10

D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\bedroom.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\config.yaml --continue_sequence pre --continue_uid 37  --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\outputs\all\log.log 

D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\bedroom.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\config.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\outputs\all\log.log 

# navigation
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\bedroom.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\config_nav.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\outputs\all_navigation\log.log 

# test-post
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script_v2_supp.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\bedroom_v2.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\config_nav_v2.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\004-bedroom\outputs\all_navigation_v3\log.log --render_test_post


Sculpture gallery:
D:\Zhou\01Softwares\Blender\blender.exe -b -P D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\blender_script.py -- --object_path D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\005-gallery\gallery.blend --config D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\005-gallery\config.yaml --log D:\Zhou\04Learning\Res\MyProjects\ChangeDet\data\blender_dataset\005-gallery\outputs\all\log.log 



--object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2
'''


# if __name__ == "__main__":
#     start_i = time.time()
#     config_path = args.config
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#     main(args.object_path, config, args.continue_sequence, args.continue_uid)
#     end_i = time.time()
#     print("Finished", "in", end_i - start_i, "seconds")



