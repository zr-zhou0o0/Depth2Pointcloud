'''
Add Geometric Primitives Script for Blender
This script loads a 3D object, adds random geometric primitives as children of an empty object,
and exports the combined scene as a GLB file.
'''

import argparse
import sys
import os
import math
import random
import json
import time
import bpy
from mathutils import Vector, Euler
import bmesh


IMPORT_FUNCTIONS = {
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


def init_scene():
    """Initialize the scene by clearing all objects, materials, and textures."""
    # Delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Delete all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # Delete all textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # Delete all images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str):
    """Load a 3D object into the scene."""
    file_extension = object_path.split(".")[-1].lower()
    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path}")

    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)


def delete_invisible_objects():
    """Delete all invisible objects in the scene."""
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


def scene_bbox():
    """Calculate the bounding box of all mesh objects in the scene."""
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
        raise RuntimeError("No mesh objects in scene to compute bounding box")
    
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene():
    """Normalize the scene to fit in a unit cube centered at origin."""
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    print(f"[DEBUG] Scene root objects: {[obj.name for obj in scene_root_objects]}")

    if len(scene_root_objects) > 1:
        # Create an empty object as parent for all root objects
        scene = bpy.data.objects.new("SceneParent", None)
        bpy.context.scene.collection.objects.link(scene)

        # Parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

def create_random_primitive(idx, primitive_type, empty_parent, bbox_min, bbox_max, scale_range=(0.1, 0.4), placement_offset=0.0):
    """Create a random geometric primitive as a child of the empty parent."""
    print(f"[DEBUG] Bbox min: {bbox_min}, bbox max: {bbox_max}")

    # Calculate position within the bounding box with some margin
    margin = 0.3 # 0.1
    
    # Decide which axis to place the primitive along (x, y, or z)
    axis = random.choice(['x', 'y', 'z'])
    # Decide whether to place on the min or max side of the axis
    side = random.choice([-1, 1])

    if axis == 'x':
        if side == -1:
            pos_x = random.uniform(bbox_min.x - placement_offset - margin, bbox_min.x - placement_offset)
        else:
            pos_x = random.uniform(bbox_max.x + placement_offset, bbox_max.x + placement_offset + margin)
        pos_y = random.uniform(bbox_min.y - margin, bbox_max.y + margin)
        pos_z = random.uniform(bbox_min.z - margin, bbox_max.z + margin)
    elif axis == 'y':
        if side == -1:
            pos_y = random.uniform(bbox_min.y - placement_offset - margin, bbox_min.y - placement_offset)
        else:
            pos_y = random.uniform(bbox_max.y + placement_offset, bbox_max.y + placement_offset + margin)
        pos_x = random.uniform(bbox_min.x - margin, bbox_max.x + margin)
        pos_z = random.uniform(bbox_min.z - margin, bbox_max.z + margin)
    else: # axis == 'z'
        if side == -1:
            pos_z = random.uniform(bbox_min.z - placement_offset - margin, bbox_min.z - placement_offset)
        else:
            pos_z = random.uniform(bbox_max.z + placement_offset, bbox_max.z + placement_offset + margin)
        pos_x = random.uniform(bbox_min.x - margin, bbox_max.x + margin)
        pos_y = random.uniform(bbox_min.y - margin, bbox_max.y + margin)
    
    # Random scale
    scale = random.uniform(scale_range[0], scale_range[1])
    
    # Random rotation
    rot_x = random.uniform(0, 2 * math.pi)
    rot_y = random.uniform(0, 2 * math.pi)
    rot_z = random.uniform(0, 2 * math.pi)
    
    # Create primitive based on type
    if primitive_type == 'cube':
        bpy.ops.mesh.primitive_cube_add(
            size=2,
            location=(pos_x, pos_y, pos_z),
            rotation=(rot_x, rot_y, rot_z)
        )
    elif primitive_type == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(
            radius=1,
            depth=2,
            location=(pos_x, pos_y, pos_z),
            rotation=(rot_x, rot_y, rot_z)
        )
    elif primitive_type == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=1,
            location=(pos_x, pos_y, pos_z),
            rotation=(rot_x, rot_y, rot_z)
        )
    elif primitive_type == 'cone':
        bpy.ops.mesh.primitive_cone_add(
            radius1=1,
            radius2=0,
            depth=2,
            location=(pos_x, pos_y, pos_z),
            rotation=(rot_x, rot_y, rot_z)
        )
    else:
        raise ValueError(f"Unsupported primitive type: {primitive_type}")
    
    # Get the created object and set its properties
    primitive_obj = bpy.context.active_object
    primitive_obj.name = f"{idx}_{axis}"
    primitive_obj.scale = (scale, scale, scale)
    
    # For cube primitives, apply additional random scaling to each axis to create cuboids
    if primitive_type == 'cube':
        # Random scale factors for each axis (0.5 to 2.0 times the base scale)
        scale_x_factor = random.uniform(0.5, 2.0)
        scale_y_factor = random.uniform(0.5, 2.0)
        scale_z_factor = random.uniform(0.5, 2.0)
        
        # Apply the additional scaling to each axis
        primitive_obj.scale = (
            scale * scale_x_factor,
            scale * scale_y_factor, 
            scale * scale_z_factor
        )
        
        print(f"Cube scaled to cuboid: X={scale_x_factor:.2f}, Y={scale_y_factor:.2f}, Z={scale_z_factor:.2f}")
    
    # For cylinder primitives, apply random scaling to radius and height
    elif primitive_type == 'cylinder':
        # Random scale factors for radius (X,Y) and height (Z)
        radius_factor = random.uniform(0.5, 2.0)  # Scale radius
        height_factor = random.uniform(0.5, 3.0)  # Scale height, can be taller
        
        # Apply the scaling: X,Y for radius, Z for height
        primitive_obj.scale = (
            scale * radius_factor,
            scale * radius_factor,  # Keep X and Y the same for circular cross-section
            scale * height_factor
        )
        
        print(f"Cylinder scaled: radius={radius_factor:.2f}, height={height_factor:.2f}")
    
    # For cone primitives, apply random scaling to radius and height
    elif primitive_type == 'cone':
        # Random scale factors for radius (X,Y) and height (Z)
        radius_factor = random.uniform(0.5, 2.0)  # Scale base radius
        height_factor = random.uniform(0.5, 3.0)  # Scale height, can be taller
        
        # Apply the scaling: X,Y for radius, Z for height
        primitive_obj.scale = (
            scale * radius_factor,
            scale * radius_factor,  # Keep X and Y the same for circular base
            scale * height_factor
        )
        
        print(f"Cone scaled: radius={radius_factor:.2f}, height={height_factor:.2f}")
    
    primitive_obj.parent = empty_parent
    
    # Create a simple material for the primitive
    material = bpy.data.materials.new(name=f"PrimitiveMaterial_{primitive_type}")
    material.use_nodes = True
    
    # Set random color
    if material.node_tree:
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Random color
            r = random.uniform(0.2, 0.8)
            g = random.uniform(0.2, 0.8)
            b = random.uniform(0.2, 0.8)
            bsdf.inputs[0].default_value = (r, g, b, 1.0)
    
    primitive_obj.data.materials.append(material)
    
    return primitive_obj


def add_geometric_primitives(num_primitives, primitive_types, use_time_seed):
    """Add random geometric primitives to the scene."""
    # Get original scene bounding box
    bbox_min, bbox_max = scene_bbox()
    
    # Create empty object named "random_geometry"
    empty_obj = bpy.data.objects.new("random_geometry", None)
    bpy.context.scene.collection.objects.link(empty_obj)
    
    print(f"Adding {num_primitives} random primitives...")
    
    # Add random primitives
    for i in range(num_primitives):

        if use_time_seed:
            current_time = time.time()
            seed = int((current_time * 1000000) % (2**32))
            random.seed(seed)
            print(f"[DEBUG] Random seed for primitive {i+1}: {seed}")

        primitive_type = random.choice(primitive_types)
        primitive_obj = create_random_primitive(i, primitive_type, empty_obj, bbox_min, bbox_max)
        
        # Primitives are already parented to empty_obj in create_random_primitive()
        # No need to move to any collection - they stay in the default scene collection
        
        print(f"Added {primitive_type} primitive {i+1}/{num_primitives}")



def post_process_primitives_for_overlap(original_bbox_min, original_bbox_max):
    """
    Check for overlaps between added primitives and the original object's bounding box.
    If an overlap is detected, move the primitive away from the original object.
    """
    print("[INFO] Starting post-processing to prevent overlaps...")
    print(f"[DEBUG] Bbox min: {original_bbox_min}, bbox max: {original_bbox_max}")
    
    # Find the empty object holding the random primitives
    random_geometry_parent = bpy.data.objects.get("random_geometry")
    if not random_geometry_parent:
        print("[WARNING] 'random_geometry' parent object not found. Skipping overlap check.")
        return

    primitives = [obj for obj in random_geometry_parent.children if obj.type == 'MESH']
    
    for prim in primitives:
        # Calculate the world-space bounding box of the primitive
        prim_bbox_min = (math.inf,) * 3
        prim_bbox_max = (-math.inf,) * 3
        for corner in prim.bound_box:
            world_corner = prim.matrix_world @ Vector(corner)
            prim_bbox_min = tuple(min(x, y) for x, y in zip(prim_bbox_min, world_corner))
            prim_bbox_max = tuple(max(x, y) for x, y in zip(prim_bbox_max, world_corner))
        
        prim_bbox_min = Vector(prim_bbox_min)
        prim_bbox_max = Vector(prim_bbox_max)
        prim_center = (prim_bbox_min + prim_bbox_max) / 2
        
        original_center = (original_bbox_min + original_bbox_max) / 2
        
        # Check for overlap on each axis and apply correction
        
        axis = prim.name.split('_')[-1]
        
        
         
        # Z-axis (Up/Down)
        if axis == 'z':
            if prim_center.z > original_center.z: # Primitive is above
                overlap = original_bbox_max.z - prim_bbox_min.z
                if overlap > 0:
                    prim.location.z += overlap
                    print(f"Moved primitive '{prim.name}' up by {overlap:.3f} to resolve Z overlap.")
            else: # Primitive is below
                overlap = prim_bbox_max.z - original_bbox_min.z
                if overlap > 0:
                    prim.location.z -= overlap
                    print(f"Moved primitive '{prim.name}' down by {overlap:.3f} to resolve Z overlap.")

        # Y-axis (Front/Back)
        if axis == "y":
            if prim_center.y > original_center.y: # Primitive is in front
                overlap = original_bbox_max.y - prim_bbox_min.y
                if overlap > 0:
                    prim.location.y += overlap
                    print(f"Moved primitive '{prim.name}' forward by {overlap:.3f} to resolve Y overlap.")
            else: # Primitive is behind
                overlap = prim_bbox_max.y - original_bbox_min.y
                if overlap > 0:
                    prim.location.y -= overlap
                    print(f"Moved primitive '{prim.name}' backward by {overlap:.3f} to resolve Y overlap.")


        # X-axis (Left/Right)
        if axis == "x":
            if prim_center.x > original_center.x: # Primitive is to the right
                overlap = original_bbox_max.x - prim_bbox_min.x
                if overlap > 0:
                    prim.location.x += overlap
                    print(f"Moved primitive '{prim.name}' right by {overlap:.3f} to resolve X overlap.")
            else: # Primitive is to the left
                overlap = prim_bbox_max.x - original_bbox_min.x
                if overlap > 0:
                    prim.location.x -= overlap
                    print(f"Moved primitive '{prim.name}' left by {overlap:.3f} to resolve X overlap.")
                
    # Update the scene to reflect the changes
    bpy.context.view_layer.update()
    print("[INFO] Overlap post-processing finished.")


def export_glb(output_path):
    """Export the scene as a GLB file."""
    # Select all objects
    bpy.ops.object.select_all(action='SELECT')
    
    # Export as GLB
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        use_selection=False,
        export_format='GLB',
        export_materials='EXPORT',
        export_colors=True,
        export_extras=False,
        export_cameras=False,
        export_lights=False
    )
    
    print(f"Exported scene to: {output_path}")


def main(args):
    """Main function to process the object and add geometric primitives."""
    print(f"Processing object: {args.object}")
    print(f"Output file: {args.output_file}")
    print(f"Number of primitives: {args.num_primitives}")
    print(f"Primitive types: {args.primitive_types}")
    
    # Initialize scene
    if args.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(args.object)
    
    print('[INFO] Scene initialized and object loaded.')
    
    # Normalize scene
    scale, offset = normalize_scene()
    original_bbox_min, original_bbox_max = scene_bbox()
    print('[INFO] Scene normalized.')
    
    # Add geometric primitives
    add_geometric_primitives(args.num_primitives, args.primitive_types, args.use_time_seed)
    print('[INFO] Geometric primitives added.')

    # Post-process primitives to avoid overlap
    post_process_primitives_for_overlap(original_bbox_min, original_bbox_max)

    
    # Export the result
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    export_glb(args.output_file)
    print('[INFO] Export completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add geometric primitives to a 3D object')
    parser.add_argument('--object', type=str, required=True, help='Path to the 3D object file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output GLB file')
    parser.add_argument('--num_primitives_min', type=int, default=2, help='Minimum number of primitives to add')
    parser.add_argument('--num_primitives_max', type=int, default=8, help='Maximum number of primitives to add')
    parser.add_argument('--primitive_types', type=str, default='cube,cylinder,sphere,cone', 
                        help='Comma-separated list of primitive types')
    parser.add_argument('--sha256', type=str, default=None, help='SHA256 hash of the object for reproducible randomness')
    parser.add_argument('--use_time_seed', action='store_true', help='Use current time as random seed for true randomness')
    
    # Parse arguments from command line
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    
    # Parse primitive types
    args.primitive_types = [t.strip() for t in args.primitive_types.split(',')]
    
    # Set random seed based on different strategies
    if args.use_time_seed:
        # Use current time with microsecond precision for maximum randomness
        current_time = time.time()
        # Use microseconds part for better randomness distribution
        seed = int((current_time * 1000000) % (2**32))
        random.seed(seed)
        print(f"Using time-based random seed: {seed} (time: {current_time})")
    elif args.sha256:
        # Use first 8 characters of SHA256 hash as seed
        seed = int(args.sha256[:8], 16)
        random.seed(seed)
        print(f"Using random seed based on SHA256: {seed}")
    else:
        # Extract SHA256 from output filename as fallback
        output_basename = os.path.basename(args.output_file)
        if output_basename.endswith('.glb'):
            potential_sha256 = output_basename[:-4]  # Remove .glb extension
            if len(potential_sha256) >= 8:
                try:
                    seed = int(potential_sha256[:8], 16)
                    random.seed(seed)
                    print(f"Using random seed from filename: {seed}")
                except ValueError:
                    # If not valid hex, use filename hash
                    seed = hash(potential_sha256) % (2**32)
                    random.seed(seed)
                    print(f"Using random seed from filename hash: {seed}")
            else:
                random.seed(42)  # Default fallback
                print("Using default random seed: 42")
        else:
            random.seed(42)  # Default fallback
            print("Using default random seed: 42")
    
    # Randomly determine the number of primitives within the specified range
    args.num_primitives = random.randint(args.num_primitives_min, args.num_primitives_max)
    print(f"Randomly selected {args.num_primitives} primitives (range: {args.num_primitives_min}-{args.num_primitives_max})")
    
    try:
        main(args)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
