"""Generate amodal mask and amodal image assets for each sample."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def create_amodal_mask(object_dir: Path) -> None:
	"""Compose an amodal mask combining occluded and visible regions."""

	occ_mask_path = object_dir / "occ_mask.png"
	vis_mask_path = object_dir / "vis_mask.png"
	output_path = object_dir / "amodal_mask.png"

	if not occ_mask_path.exists():
		print(f"Skip amodal mask for {object_dir}: occ_mask.png missing")
		return

	if not vis_mask_path.exists():
		print(f"Skip amodal mask for {object_dir}: vis_mask.png missing")
		return

	occ_mask = Image.open(occ_mask_path).convert("L")
	vis_mask = Image.open(vis_mask_path).convert("L")

	if occ_mask.size != vis_mask.size:
		vis_mask = vis_mask.resize(occ_mask.size, resample=Image.NEAREST)

	occ_arr = np.array(occ_mask) > 0
	vis_arr = np.array(vis_mask) > 0

	amodal_arr = np.full(occ_arr.shape, 255, dtype=np.uint8)
	amodal_arr[occ_arr] = 0
	amodal_arr[vis_arr] = 200

	Image.fromarray(amodal_arr, mode="L").save(output_path)
	print(f"Saved amodal mask to {output_path}")


def create_amodal_image(object_dir: Path) -> None:
	"""Overlay visible pixels on black background to build the amodal image."""

	rgb_path = object_dir / "full_rgb.png"
	vis_mask_path = object_dir / "vis_mask.png"
	output_path = object_dir / "amodal_image.png"

	if not rgb_path.exists():
		print(f"Skip amodal image for {object_dir}: full_rgb.png missing")
		return

	if not vis_mask_path.exists():
		print(f"Skip amodal image for {object_dir}: vis_mask.png missing")
		return

	rgb_image = Image.open(rgb_path).convert("RGB")
	vis_mask = Image.open(vis_mask_path).convert("L")

	if rgb_image.size != vis_mask.size:
		vis_mask = vis_mask.resize(rgb_image.size, resample=Image.NEAREST)

	rgb_arr = np.array(rgb_image)
	vis_arr = np.array(vis_mask) > 0

	amodal_arr = np.zeros_like(rgb_arr)
	amodal_arr[vis_arr] = rgb_arr[vis_arr]

	Image.fromarray(amodal_arr, mode="RGB").save(output_path)
	print(f"Saved amodal image to {output_path}")


def process_dataset(dataset_root: Path) -> None:
	for object_dir in dataset_root.iterdir():
		if object_dir.is_dir():
			create_amodal_mask(object_dir)
			create_amodal_image(object_dir)


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate amodal assets from masks and RGB images.")
	parser.add_argument(
		"--dataset-root",
		type=Path,
		required=True,
		help="Root directory containing per-sample folders.",
	)
	args = parser.parse_args()

	process_dataset(args.dataset_root)


if __name__ == "__main__":
	main()
