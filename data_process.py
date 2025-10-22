from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image


RAW_SUBDIRS = {
	"depth": "depth",
	"rgb": "rgb",
	"annotation": "annotation",
	"segm": "segm",
}

DEPTH_SUFFIXES = [".npy.gz", ".npy"]
RGB_SUFFIXES = [".jpeg", ".jpg", ".png"]
ANNOTATION_SUFFIXES = [".json"]
SEGMENTATION_SUFFIXES = [".npy.gz", ".npy"]
DEPTH_ERROR_THRESHOLD = 1000.0


def parse_filename_triplet(filename: str) -> tuple[str, str, str]:
	"""Split a raw asset filename into ``uid1``, ``middle``, ``uid2`` parts."""

	stem = filename
	for suffix in (
		*DEPTH_SUFFIXES,
		*RGB_SUFFIXES,
		*ANNOTATION_SUFFIXES,
		*SEGMENTATION_SUFFIXES,
	):
		if stem.endswith(suffix):
			stem = stem[: -len(suffix)]
			break
	uid1, middle, uid2 = stem.split("_", maxsplit=2)
	return uid1, middle, uid2


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def load_npy_array(file_path: Path) -> np.ndarray:
	if file_path.suffix == ".gz":
		with gzip.open(file_path, "rb") as gz_file:
			return np.load(gz_file, allow_pickle=False)
	return np.load(file_path, allow_pickle=False)


def load_depth_array(depth_path: Path) -> np.ndarray:
	return load_npy_array(depth_path)


def load_segmentation_array(segm_path: Path) -> np.ndarray:
	return load_npy_array(segm_path)


def load_annotation(annotation_path: Path) -> dict[str, Any]:
	with annotation_path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def depth_to_png(depth_array: np.ndarray) -> Image.Image:
	depth = np.asarray(depth_array, dtype=np.float32)
	finite_mask = np.isfinite(depth)
	valid_mask = finite_mask & (depth <= DEPTH_ERROR_THRESHOLD)

	luminance = np.zeros(depth.shape, dtype=np.uint8)
	alpha = np.zeros(depth.shape, dtype=np.uint8)

	if valid_mask.any():
		valid_values = depth[valid_mask]
		min_val = float(valid_values.min())
		max_val = float(valid_values.max())

		if np.isclose(max_val, min_val):
			luminance[valid_mask] = 255
		else:
			normalized = (valid_values - min_val) / (max_val - min_val)
			luminance[valid_mask] = np.round(normalized * 255).astype(np.uint8)

		alpha[valid_mask] = 255

	la_array = np.stack([luminance, alpha], axis=-1)
	image = Image.fromarray(la_array)
	if image.mode != "LA":
		image = image.convert("LA")
	return image


def load_rgb_image(rgb_path: Path) -> Image.Image:
	with Image.open(rgb_path) as img:
		return img.convert("RGB")


def iter_depth_files(depth_root: Path) -> Iterable[Path]:
	candidates: dict[tuple[str, str], Path] = {}
	for suffix in DEPTH_SUFFIXES:
		for path in depth_root.glob(f"*{suffix}"):
			try:
				uid1, _, uid2 = parse_filename_triplet(path.name)
			except ValueError:
				logging.debug("Skipping depth file with unexpected format: %s", path.name)
				continue
			key = (uid1, uid2)
			if key not in candidates:
				candidates[key] = path
	for path in sorted(candidates.values(), key=lambda item: item.name):
		yield path


def resolve_rgb_path(uid1: str, uid2: str, root: Path) -> Optional[Path]:
	for suffix in RGB_SUFFIXES:
		candidate = root / f"{uid1}_rgb_{uid2}{suffix}"
		if candidate.exists():
			return candidate
	return None


def resolve_annotation_path(uid1: str, uid2: str, root: Path) -> Optional[Path]:
	for suffix in ANNOTATION_SUFFIXES:
		candidate = root / f"{uid1}_annotation_{uid2}{suffix}"
		if candidate.exists():
			return candidate
	return None


def resolve_segmentation_path(uid1: str, uid2: str, root: Path) -> Optional[Path]:
	for suffix in SEGMENTATION_SUFFIXES:
		candidate = root / f"{uid1}_segm_{uid2}{suffix}"
		if candidate.exists():
			return candidate
	return None


def save_vis_mask(mask_array: np.ndarray, output_path: Path) -> None:
	if mask_array.ndim == 3:
		mask_array = np.any(mask_array, axis=-1)
	binary_mask = (mask_array.astype(np.uint8)) * 255
	mask_image = Image.fromarray(binary_mask, mode="L")
	mask_image.save(output_path)


def export_objects(
	scene_uid: str,
	annotation: dict[str, Any],
	depth_array: np.ndarray,
	depth_png: Image.Image,
	rgb_image: Image.Image,
	segm_array: Optional[np.ndarray],
	output_root: Path,
) -> int:
	obj_dict = annotation.get("obj_dict")
	if not isinstance(obj_dict, dict):
		return 0

	segm_available = segm_array is not None
	if not segm_available:
		logging.warning("Segmentation missing for scene %s; skipping vis_mask generation", scene_uid)

	def sort_object_key(value: str) -> tuple[int, int | str]:
		try:
			return (0, int(value))
		except ValueError:
			return (1, value)

	object_count = 0
	for obj_key in sorted(obj_dict.keys(), key=sort_object_key):
		obj_entry = obj_dict.get(obj_key, {})
		obj_id_list = obj_entry.get("obj_id")
		if not obj_id_list:
			continue

		try:
			obj_id_value = int(obj_id_list[0])
		except (TypeError, ValueError, IndexError):
			logging.debug(
				"Skipping object %s in scene %s due to invalid obj_id payload", obj_key, scene_uid
			)
			continue

		object_uid = f"{scene_uid}-{obj_id_value:06d}"
		object_dir = output_root / object_uid
		if object_dir.exists():
			shutil.rmtree(object_dir)
		ensure_dir(object_dir)

		np.save(object_dir / "full_depth.npy", depth_array)
		depth_png.save(object_dir / "full_depth.png")
		rgb_image.save(object_dir / "full_rgb.png")
		metadata = deepcopy(annotation)
		metadata["dataset_name"] = object_uid
		obj_dict_meta = metadata.get("obj_dict")
		if isinstance(obj_dict_meta, dict):
			metadata["obj_dict"] = {obj_key: deepcopy(obj_dict_meta.get(obj_key, {}))}
		with (object_dir / "meta-data.json").open("w", encoding="utf-8") as handle:
			json.dump(metadata, handle, ensure_ascii=False)

		if segm_available:
			mask = (segm_array == obj_id_value)
			save_vis_mask(mask, object_dir / "vis_mask.png")

		object_count += 1

	return object_count


def process_scene(
	depth_path: Path,
	rgb_root: Path,
	annotation_root: Path,
	segm_root: Path,
	output_root: Path,
) -> bool:
	uid1, _, uid2 = parse_filename_triplet(depth_path.name)
	scene_uid = f"{uid1}-{uid2}"
	rgb_path = resolve_rgb_path(uid1, uid2, rgb_root)
	annotation_path = resolve_annotation_path(uid1, uid2, annotation_root)
	segm_path = resolve_segmentation_path(uid1, uid2, segm_root)

	missing = [
		label
		for label, path in (
			("rgb", rgb_path),
			("annotation", annotation_path),
		)
		if path is None
	]
	if missing:
		logging.warning("Skipping %s (missing: %s)", depth_path.name, ", ".join(missing))
		return None

	depth_array = load_depth_array(depth_path)
	depth_png = depth_to_png(depth_array)

	assert rgb_path is not None
	assert annotation_path is not None
	annotation_data = load_annotation(annotation_path)
	rgb_image = load_rgb_image(rgb_path)

	segm_array: Optional[np.ndarray] = None
	if segm_path is None:
		logging.warning("Segmentation file missing for scene %s", scene_uid)
	else:
		try:
			segm_array = load_segmentation_array(segm_path)
		except Exception as exc:
			logging.error("Failed to load segmentation for scene %s: %s", scene_uid, exc)

	object_count = export_objects(
		scene_uid,
		annotation_data,
		depth_array,
		depth_png,
		rgb_image,
		segm_array,
		output_root,
	)

	logging.info("Processed scene %s into %d objects", scene_uid, object_count)
	return object_count > 0


def build_dataset(
	raw_root: Path,
	output_root: Path,
	limit: Optional[int] = None,
) -> None:
	depth_root = raw_root / RAW_SUBDIRS["depth"]
	rgb_root = raw_root / RAW_SUBDIRS["rgb"]
	annotation_root = raw_root / RAW_SUBDIRS["annotation"]
	segm_root = raw_root / RAW_SUBDIRS["segm"]

	ensure_dir(output_root)

	processed = 0
	for depth_path in iter_depth_files(depth_root):
		result = process_scene(
			depth_path,
			rgb_root,
			annotation_root,
			segm_root,
			output_root,
		)
		if result:
			processed += 1
		if limit is not None and processed >= limit:
			break

	logging.info("Finished processing %d scene assets", processed)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--raw-root",
		type=Path,
		default=Path("data/raw"),
		help="Root directory containing the raw modality folders.",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("data/outputs/dataset"),
		help="Destination directory for the reorganized dataset.",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Process at most this many samples (useful for smoke tests).",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		help="Python logging level (e.g., DEBUG, INFO, WARNING).",
	)

	args = parser.parse_args()

	logging.basicConfig(
		level=getattr(logging, args.log_level.upper(), logging.INFO),
		format="[%(levelname)s] %(message)s",
	)

	build_dataset(args.raw_root, args.output_root, args.limit)


if __name__ == "__main__":
	main()
