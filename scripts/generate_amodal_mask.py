'''
generate 2d bbox and amodal mask
'''

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def visualize_2d_bbox(object_dir: Path) -> None:
    # Load metadata
    metadata_path = object_dir / "meta-data.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load full RGB image
    rgb_path = object_dir / "full_rgb.png"
    rgb_image = Image.open(rgb_path).convert("RGB")

    # Create a white canvas
    white_canvas = Image.new("RGB", rgb_image.size, "black")
    draw = ImageDraw.Draw(white_canvas)

    # Iterate over objects in metadata
    obj_dict = metadata.get("obj_dict", {})
    for obj_key, obj_data in obj_dict.items():
        bbox_2d_from_3d = obj_data.get("bbox_2d_from_3d")
        if not bbox_2d_from_3d:
            continue

        # Convert 8 3D bbox points to 2D projections
        points_2d = [(int(x), int(y)) for x, y in bbox_2d_from_3d]
        
        points_2d_face1 = [
            points_2d[0], points_2d[1], points_2d[3], points_2d[2], points_2d[0],  # Front
            
        ]
        
        points_2d_face2 = [
            points_2d[4], points_2d[5], points_2d[7], points_2d[6], points_2d[4],  # Back
        ]
        
        points_2d_face3 = [
            points_2d[0], points_2d[4], points_2d[5], points_2d[1], points_2d[0],  # top
        ]
        
        points_2d_face4 = [
            points_2d[2], points_2d[6], points_2d[7], points_2d[3], points_2d[2],  # bottom
        ]
        
        points_2d_face5 = [
            points_2d[0], points_2d[2], points_2d[6], points_2d[4], points_2d[0],  # left
        ]
        
        points_2d_face6 = [
            points_2d[1], points_2d[3], points_2d[7], points_2d[5], points_2d[1],  # right
        ]

        # Draw the 2D bbox projection
        draw.polygon(points_2d_face1, fill="white")
        draw.polygon(points_2d_face2, fill="white")
        draw.polygon(points_2d_face3, fill="white")
        draw.polygon(points_2d_face4, fill="white")
        draw.polygon(points_2d_face5, fill="white")
        draw.polygon(points_2d_face6, fill="white")
        

    # Save the visualized image
    output_path = os.path.join(object_dir, "2dbbox.png")
    white_canvas.save(output_path)
    print(f"Saved visualization to {output_path}")


def generate_amodal_mask(object_dir: Path) -> None:
    """Build amodal mask by subtracting visible mask from 2D bbox mask."""

    bbox_mask_path = object_dir / "2dbbox.png"
    vis_mask_path = object_dir / "vis_mask.png"
    output_path = object_dir / "amodal_mask.png"

    if not bbox_mask_path.exists():
        print(f"Skip amodal mask for {object_dir}: 2dbbox.png missing")
        return

    if not vis_mask_path.exists():
        print(f"Skip amodal mask for {object_dir}: vis_mask.png missing")
        return

    bbox_mask = Image.open(bbox_mask_path).convert("L")
    vis_mask = Image.open(vis_mask_path).convert("L")

    if bbox_mask.size != vis_mask.size:
        vis_mask = vis_mask.resize(bbox_mask.size, resample=Image.NEAREST)

    bbox_arr = np.array(bbox_mask) > 0
    vis_arr = np.array(vis_mask) > 0

    amodal_arr = np.logical_and(bbox_arr, np.logical_not(vis_arr)).astype(np.uint8) * 255
    amodal_image = Image.fromarray(amodal_arr, mode="L")
    amodal_image.save(output_path)
    print(f"Saved amodal mask to {output_path}")


def process_dataset(dataset_root: Path) -> None:

    for object_dir in dataset_root.iterdir():
        if object_dir.is_dir():
            visualize_2d_bbox(object_dir)
            generate_amodal_mask(object_dir)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Visualize 2D BBox from metadata.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the dataset containing object folders.",
    )
    args = parser.parse_args()

    process_dataset(args.dataset_root)


if __name__ == "__main__":
    main()