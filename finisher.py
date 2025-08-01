import bpy
import sys
import argparse
import os
import math
from mathutils import Vector

def enable_gltf_addon():
    """Ensure the GLTF 2.0 add-on is enabled."""
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
        print("GLTF 2.0 add-on enabled successfully.")
    except Exception as e:
        print(f"Error enabling GLTF add-on: {str(e)}")
        sys.exit(1)

def get_args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Path to Trellis output .glb')
    parser.add_argument('-o', '--output', required=True, help='Path for final .glb output')
    return parser.parse_args(argv)

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge(do_recursive=True)

def import_mesh(filepath):
    print(f"Importing mesh: {filepath}")
    if not os.path.exists(filepath):
        print(f"❌ ERROR: Input file not found: {filepath}")
        sys.exit(1)
    enable_gltf_addon()
    try:
        bpy.ops.import_scene.gltf(filepath=filepath)
    except Exception as e:
        print(f"❌ ERROR: Failed to import GLB file: {str(e)}")
        sys.exit(1)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes:
        print("❌ ERROR: No mesh objects found in imported file.")
        sys.exit(1)
    return meshes[0]

def center_and_rescale(obj, target_size=1.0):
    print("Centering and resizing mesh...")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    verts = [v.co for v in obj.data.vertices]
    if not verts:
        print("❌ ERROR: Mesh has no vertices.")
        sys.exit(1)
    
    # Initialize min and max corners
    min_corner = Vector(verts[0])
    max_corner = Vector(verts[0])

    # Compute bounding box
    for v in verts[1:]:
        min_corner = Vector((
            min(min_corner.x, v.x),
            min(min_corner.y, v.y),
            min(min_corner.z, v.z)
        ))
        max_corner = Vector((
            max(max_corner.x, v.x),
            max(max_corner.y, v.y),
            max(max_corner.z, v.z)
        ))

    dimensions = max_corner - min_corner
    max_dim = max(dimensions.x, dimensions.y, dimensions.z)

    if max_dim > 0:
        scale_factor = target_size / max_dim
        obj.scale *= scale_factor
        bpy.ops.object.transform_apply(scale=True)

    obj.location = (0, 0, 0)

def export_glb(obj, output_path):
    print(f"Exporting optimized mesh to: {output_path}")
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            use_selection=True,
            export_apply=False,
            export_materials='EXPORT',
            export_yup=True
        )
    except Exception as e:
        print(f"❌ ERROR: Failed to export GLB file: {str(e)}")
        sys.exit(1)

def main(args):
    clean_scene()
    obj = import_mesh(args.input)
    center_and_rescale(obj)
    export_glb(obj, args.output)

if __name__ == '__main__':
    args = get_args()
    main(args)