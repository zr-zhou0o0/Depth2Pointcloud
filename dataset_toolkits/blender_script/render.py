'''
Render version 1.0
Data folder structure：
renders/
    |-{sha256}/
        |-000.png
        |-000_depth.png
        |-000_mask.png
        |-...
    |-{sha256}/
        |-000.png
        |-000_depth.png
        |-...
'''

import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import glob


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}


def setup_gpu(gpu_ids_str):
    """Configures Blender to use a specific GPU for Cycles rendering."""
    try:
        # gpu_ids = [int(id.strip()) for id in gpu_ids_str.split(',')]
        gpu_ids = [int(x) for x in gpu_ids_str.split(',') if x]
         
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        bpy.context.scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'  # Or 'OPTIX' or 'HIP' depending on your GPU
        
        # De-select all devices first
        for device in prefs.devices:
            device.use = False
            
        enabled_gpus = []
        # Select the specified GPUs
        for gpu_id in gpu_ids:
            if gpu_id < len(prefs.devices):
                prefs.devices[gpu_id].use = True
                enabled_gpus.append(prefs.devices[gpu_id].name)
            else:
                print(f"Warning: GPU ID {gpu_id} is out of range. Found {len(prefs.devices)} devices.")
        
        if enabled_gpus:
            print(f"Successfully configured to use {len(enabled_gpus)} GPUs: {', '.join(enabled_gpus)}")
        else:
            print("Warning: No valid GPUs were enabled. Falling back to default.")

    except Exception as e:
        print(f"Error setting up GPU: {e}. Rendering will proceed with default settings.")


def delete_geometry_by_name(name_pattern):
    """
    Finds and deletes all objects matching a name pattern and their children.
    """
    # Use a list to collect objects to delete to avoid issues with modifying collections while iterating
    objects_to_delete = []
    
    # Find all parent objects matching the name pattern
    parent_objects = [obj for obj in bpy.data.objects if name_pattern in obj.name]
    
    for parent in parent_objects:
        # Recursively find all children
        children_recursive = []
        def get_children(obj):
            for child in obj.children:
                children_recursive.append(child)
                get_children(child)
        
        get_children(parent)
        
        # Add children and the parent itself to the main deletion list
        objects_to_delete.extend(children_recursive)
        objects_to_delete.append(parent)

    if objects_to_delete:
        # Remove duplicates
        objects_to_delete = list(set(objects_to_delete))
        print(f"[INFO] Deleting {len(objects_to_delete)} objects related to '{name_pattern}'.")
        
        # Deselect all, select objects to be deleted, and delete them
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects_to_delete:
            # Object might have been deleted already if it was a child of another deleted object
            if obj.name in bpy.data.objects:
                bpy.data.objects[obj.name].select_set(True)
        bpy.ops.object.delete()


def init_render(engine='CYCLES', resolution=512, geo_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True
        
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    
def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mist=False, save_mask=False):
    if not any([save_depth, save_normal, save_albedo, save_mist, save_mask]):
        return {}, {}, {}
    outputs = {}
    spec_nodes = {}
    
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers['ViewLayer'].use_pass_z = save_depth
    bpy.context.scene.view_layers['ViewLayer'].use_pass_normal = save_normal
    bpy.context.scene.view_layers['ViewLayer'].use_pass_diffuse_color = save_albedo
    bpy.context.scene.view_layers['ViewLayer'].use_pass_mist = save_mist
    bpy.context.scene.view_layers['ViewLayer'].use_pass_object_index = save_mask
    
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'BW'
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0  # (min value you will be getting)
        map.inputs[2].default_value = 10 # (max value you will be getting)
        map.inputs[3].default_value = 0  # (min value you will map to)
        map.inputs[4].default_value = 1  # (max value you will map to)
        
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
        
        outputs['depth'] = depth_file_output
        spec_nodes['depth_map'] = map
    
    if save_normal:
        # normal_node = nodes.new(type='ShaderNodeNormalMap')
        # output_node = nodes.new(type='ShaderNodeOutputMaterial')
        # normal_node.location = (-300, 0)
        # output_node.location = (300, 0)

        # links = mat.node_tree.links
        # links.new(normal_node.outputs['Normal'], output_node.inputs['Surface'])

        # bpy.context.scene.cycles.samples = 256  # 提高样本数量以获得更好的质量
        # bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR' 
        # bpy.context.scene.render.image_settings.color_mode = 'RGBA'  
        # output_path = os.path.join(os.path.dirname(bpy.data.filepath), "normal_map.exr")  
        # bpy.context.scene.render.filepath = output_path

        # bpy.ops.render.render(write_still=True)  
            
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        # links.new(normal_node.outputs['Normal'], normal_file_output.inputs['Surface'])
        
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        # normal_file_output.format.file_format = 'PNG'
        # normal_file_output.format.color_mode = 'RGB'
        normal_file_output.format.color_mode = 'RGBA'
        normal_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        outputs['normal'] = normal_file_output
    
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
        
    if save_mist:
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = ''
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        
        outputs['mist'] = mist_file_output
    
    if save_mask:
        pass
        
    return outputs, spec_nodes, render_layers

def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
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

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light
    }


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)
        
def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
        
def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")
            
def delete_custom_normals():
     for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.scene.view_layers['View Layer'].material_override = new_mat

def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        
def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
        
def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene(additional_scale: float) -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    print(f"[DEBUG] Scene root objects: {[obj.name for obj in scene_root_objects]}")

    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox() # e.g. [2, 3, 4], [5, 6, 7]
    scale = 1 / max(bbox_max - bbox_min) # 所有轴上大减小 取最长的一个轴 作为scale
    scene.scale = scene.scale * scale

    scale = scale * additional_scale
    scene.scale = scene.scale * additional_scale

    # Apply scale to matrix_world.
    # BUG 不应该是整个场景的中心在原点，而应该是只有目标物体的中心在原点。
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2 # 把中心移到原点
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose() # 分解对象的全局变换矩阵，得到位置、旋转和缩放分量
    # matrix_world indeed is C2W
    rt = rt.to_matrix() # 将四元数旋转转换为3x3旋转矩阵
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    
    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode)
    outputs, spec_nodes, render_layers = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist,
        save_mask=arg.save_mask
    )
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
        # delete_custom_normals()
    print('[INFO] Scene initialized.')
    
    # normalize scene
    scale, offset = normalize_scene(args.scale)
    print('[INFO] Scene normalized.')
    
    # Initialize camera and lighting
    cam = init_camera()
    init_lighting()
    print('[INFO] Camera and lighting initialized.')

    # Override material
    if arg.geo_mode:
        override_material()
    
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": []
    }
    views = json.loads(arg.views)
    
    if arg.no_geo:
        delete_geometry_by_name('random_geometry')
        print('[INFO] --no_geo: Removed random geometry objects.')

        

    # Create mask node and outputs
    # Number of nodes and links are determined by the number of objects in the scene
    # So we cannot create nodes in the `init_nodes` function
    if arg.save_mask:
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links   

        mask_objects = []
        max_passidx = 0

        scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        all_mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

        print(f"[DEBUG] Scene root objects: {[obj.name for obj in scene_root_objects]}")
        print(f"[DEBUG] All mesh objects: {[obj.name for obj in all_mesh_objects]}")

        # Filter out objects that are children of 'random_geometry' empty object
        excluded_objects = set()
        
        # Check for random_geometry collection (old method)
        if 'random_geometry' in bpy.data.collections:
            random_geometry_collection = bpy.data.collections['random_geometry']
            # Get all objects in the collection, including nested ones
            def get_collection_objects(collection):
                objects = set(collection.objects)
                for child_collection in collection.children:
                    objects.update(get_collection_objects(child_collection))
                return objects
            excluded_objects.update(get_collection_objects(random_geometry_collection))
        
        # Check for random_geometry empty object (new method)
        random_geometry_objects = [obj for obj in bpy.context.scene.objects if obj.name == 'random_geometry']
        if random_geometry_objects:
            random_geometry_empty = random_geometry_objects[0]
            # Get all children (including nested children) of random_geometry empty object
            def get_children_recursive(obj):
                children = set()
                for child in obj.children:
                    children.add(child)
                    children.update(get_children_recursive(child))
                return children
            excluded_objects.update(get_children_recursive(random_geometry_empty))
            print(f"[DEBUG] Excluding {len(get_children_recursive(random_geometry_empty))} objects under random_geometry empty object")
        
        mask_objects = [obj for obj in all_mesh_objects if obj not in excluded_objects and not obj.hide_render]
        
        for i, obj in enumerate(mask_objects):
            obj.pass_index = i + 1
            max_passidx = max(max_passidx, i + 1)
        
        print(f"[INFO] Using {len(mask_objects)} objects for mask rendering")
        
        if mask_objects:
            # Create mask output file node
            mask_file_output = nodes.new('CompositorNodeOutputFile')
            mask_file_output.base_path = ''
            mask_file_output.file_slots[0].use_node_format = True
            mask_file_output.format.file_format = 'PNG'
            mask_file_output.format.color_mode = 'BW'
            mask_file_output.format.color_depth = '8'
            
            # Ensure we have enough file slots
            while len(mask_file_output.file_slots) < max_passidx + 1:
                mask_file_output.file_slots.new("")
            
            # Create ID mask nodes and combine them
            if len(mask_objects) == 1:
                # Single object case
                id_mask_node = nodes.new(type='CompositorNodeIDMask')
                id_mask_node.index = mask_objects[0].pass_index
                links.new(render_layers.outputs['IndexOB'], id_mask_node.inputs['ID value'])
                links.new(id_mask_node.outputs['Alpha'], mask_file_output.inputs[0])
            else:
                # Multiple objects case - create combined mask
                id_mask_nodes = []
                for obj in mask_objects:
                    id_mask_node = nodes.new(type='CompositorNodeIDMask')
                    id_mask_node.index = obj.pass_index
                    links.new(render_layers.outputs['IndexOB'], id_mask_node.inputs['ID value'])
                    id_mask_nodes.append(id_mask_node)
                
                # Chain Math nodes to combine all masks using MAXIMUM operation
                current_output = id_mask_nodes[0].outputs['Alpha']
                for i in range(1, len(id_mask_nodes)):
                    math_node = nodes.new(type='CompositorNodeMath')
                    math_node.operation = 'MAXIMUM'
                    links.new(current_output, math_node.inputs[0])
                    links.new(id_mask_nodes[i].outputs['Alpha'], math_node.inputs[1])
                    current_output = math_node.outputs['Value']
                
                # Connect final combined mask to output
                links.new(current_output, mask_file_output.inputs[0])
            
            outputs['mask'] = mask_file_output
            spec_nodes['mask_objects'] = mask_objects
            spec_nodes['max_passidx'] = max_passidx


    for i, view in enumerate(views):
        # Set camera parameters
        # 相机约束设置（朝向原点）在 init_camera() 中已经设置
        cam.location = (
            view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['pitch'])
        )
        cam.data.lens = 16 / np.tan(view['fov'] / 2) # 利用 fov 计算焦距
        
        if arg.save_depth: # 深度范围设置
            # √3/2 ≈ 0.866: 这是单位立方体的空间对角线的一半，确保覆盖场景中的物体
            spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(3)
            spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(3)
        
        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png')
        for name, output in outputs.items():
            output.file_slots[0].path = os.path.join(arg.output_folder, f'{i:03d}_{name}')
            
        # Render the scene
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            path = glob.glob(f'{output.file_slots[0].path}*.{ext}')[0]
            os.rename(path, f'{output.file_slots[0].path}.{ext}')
        
        # HERE
        # Save camera parameters
        metadata = {
            "file_path": f'{i:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam) 
        }
        if arg.save_depth:
            metadata['depth'] = {
                'min': view['radius'] - 0.5 * np.sqrt(3),
                'max': view['radius'] + 0.5 * np.sqrt(3)
            }
        to_export["frames"].append(metadata)

    # Save the camera parameters
    with open(os.path.join(arg.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)
        
    if arg.save_mesh:
        # triangulate meshes
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()
        print('[INFO] Meshes triangulated.')
        
        # export ply mesh
        bpy.ops.export_mesh.ply(filepath=os.path.join(arg.output_folder, 'mesh.ply'))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--no_geo', action='store_true', help='Disable geometry mode for rendering.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    parser.add_argument('--save_mask', action='store_true', help='Save the segmentation mask of target object. For partial point cloud generation.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale the object during normalization.')
    parser.add_argument('--gpu_ids', type=str, default=None,
                    help='Comma-separated GPU indices to use.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    
    if args.gpu_ids is not None:
        setup_gpu(args.gpu_ids)

    main(args)
    