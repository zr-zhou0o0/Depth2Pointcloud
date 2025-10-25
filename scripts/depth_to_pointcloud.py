from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


DEFAULT_DEPTH_FILENAME = "full_depth.npy"
DEFAULT_MASK_FILENAME = "vis_mask.png"
DEFAULT_METADATA_FILENAME = "meta-data.json"
DEFAULT_OUTPUT_FILENAME = "vis_pnts.ply"
DEFAULT_CAMERA_FRAME = "blender"
DEPTH_MAX_DEFAULT = 1000.0
DEPTH_MIN_DEFAULT = 1e-6

BLENDER_TO_CV = np.array(
	[
		[1.0, 0.0, 0.0, 0.0],
		[0.0, -1.0, 0.0, 0.0],
		[0.0, 0.0, -1.0, 0.0],
		[0.0, 0.0, 0.0, 1.0],
	],
	dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Generate per-object point clouds from depth maps and instance masks "
			"stored in the dataset created by data_process.py."
		)
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=Path("data/outputs/dataset"),
		help="Directory containing per-object folders (e.g. {uid1}-{uid2}-{obj_id:06d}).",
	)
	parser.add_argument(
		"--output-name",
		type=str,
		default=DEFAULT_OUTPUT_FILENAME,
		help="Filename for the generated point cloud inside each object folder.",
	)
	parser.add_argument(
		"--depth-max",
		type=float,
		default=DEPTH_MAX_DEFAULT,
		help="Depth values above this threshold are treated as invalid.",
	)
	parser.add_argument(
		"--depth-min",
		type=float,
		default=DEPTH_MIN_DEFAULT,
		help="Depth values below or equal to this threshold are treated as invalid.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing point clouds instead of skipping them.",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Process at most this many object folders (useful for smoke tests).",
	)
	parser.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		help="Logging level (DEBUG, INFO, WARNING, ERROR).",
	)
	parser.add_argument(
		"--camera-frame",
		choices=("opencv", "blender"),
		default=DEFAULT_CAMERA_FRAME,
		help="Coordinate frame used in metadata; blender poses are converted to OpenCV.",
	)
	return parser.parse_args()


def configure_logging(level: str) -> None:
	logging.basicConfig(
		level=getattr(logging, level.upper(), logging.INFO),
		format="[%(levelname)s] %(message)s",
	)


def iter_object_dirs(dataset_root: Path) -> Iterable[Path]:
	if not dataset_root.exists():
		raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
	for entry in sorted(dataset_root.iterdir()):
		if entry.is_dir():
			yield entry


def load_metadata(meta_path: Path) -> dict:
	with meta_path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def load_depth(depth_path: Path) -> np.ndarray:
	depth = np.load(depth_path, allow_pickle=False)
	if depth.ndim != 2:
		raise ValueError(f"Expected 2D depth array, got shape {depth.shape} for {depth_path}")
	return depth.astype(np.float32)


def load_mask(mask_path: Path) -> np.ndarray:
	with Image.open(mask_path) as img:
		mask = np.array(img.convert("L"), dtype=np.uint8)
	return mask > 0


def _convert_pose_if_needed(pose: np.ndarray, camera_frame: str) -> np.ndarray:
	if camera_frame == "blender":
		return pose @ BLENDER_TO_CV
	return pose


def ensure_intrinsics(metadata: dict) -> np.ndarray:
	intrinsics = metadata.get("camera_intrinsics")
	if intrinsics is None:
		raise KeyError("camera_intrinsics missing from metadata")
	matrix = np.asarray(intrinsics, dtype=np.float64)
	if matrix.size != 9:
		raise ValueError(f"camera_intrinsics has unexpected shape {matrix.shape}")
	return matrix.reshape(3, 3)


def ensure_camera_pose(metadata: dict, camera_frame: str) -> np.ndarray:
	rot = metadata.get("camera_pose_rot")
	tran = metadata.get("camera_pose_tran")
	if rot is not None and tran is not None:
		rot_m = np.asarray(rot, dtype=np.float64)
		tran_v = np.asarray(tran, dtype=np.float64)
		if rot_m.size != 9 or tran_v.size != 3:
			raise ValueError("camera_pose_rot or camera_pose_tran has unexpected shape")
		pose = np.eye(4, dtype=np.float64)
		pose[:3, :3] = rot_m.reshape(3, 3)
		pose[:3, 3] = tran_v.reshape(3)
		return _convert_pose_if_needed(pose, camera_frame)

	extrinsics = metadata.get("camera_extrinsics")
	if extrinsics is None:
		raise KeyError("No camera pose information available in metadata")

	extr = np.asarray(extrinsics, dtype=np.float64)
	if extr.shape != (3, 4):
		raise ValueError(f"camera_extrinsics has unexpected shape {extr.shape}")

	extr4 = np.eye(4, dtype=np.float64)
	extr4[:3, :4] = extr
	pose = np.linalg.inv(extr4)
	return _convert_pose_if_needed(pose, camera_frame)


def depth_to_points(
	depth_map: np.ndarray,
	mask: Optional[np.ndarray],
	intrinsics: np.ndarray,
	pose: np.ndarray,
	depth_min: float,
	depth_max: float,
) -> np.ndarray:
	if mask is not None and mask.shape != depth_map.shape:
		raise ValueError(
			f"Mask shape {mask.shape} does not match depth map shape {depth_map.shape}"
		)

	valid = np.isfinite(depth_map)
	valid &= depth_map > depth_min
	valid &= depth_map <= depth_max
	if mask is not None:
		valid &= mask

	if not np.any(valid):
		return np.empty((0, 3), dtype=np.float32)

	v_coords, u_coords = np.nonzero(valid)
	depth_values = depth_map[v_coords, u_coords]

	fx = intrinsics[0, 0]
	fy = intrinsics[1, 1]
	cx = intrinsics[0, 2]
	cy = intrinsics[1, 2]

	x_cam = (u_coords - cx) / fx * depth_values
	y_cam = (v_coords - cy) / fy * depth_values
	z_cam = depth_values

	camera_points = np.stack((x_cam, y_cam, z_cam), axis=1).astype(np.float64)

	rotation = pose[:3, :3]
	translation = pose[:3, 3]
	world_points = camera_points @ rotation.T + translation
	return world_points.astype(np.float32)


def save_pointcloud(points: np.ndarray, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		handle.write("ply\n")
		handle.write("format ascii 1.0\n")
		handle.write(f"element vertex {len(points)}\n")
		handle.write("property float x\n")
		handle.write("property float y\n")
		handle.write("property float z\n")
		handle.write("end_header\n")
		for point in points:
			handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

def save_pointcloud_npy(points: np.ndarray, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(output_path, points)


def process_object(
	object_dir: Path,
	output_name: str,
	depth_min: float,
	depth_max: float,
	overwrite: bool,
	camera_frame: str,
) -> bool:
	meta_path = object_dir / DEFAULT_METADATA_FILENAME
	depth_path = object_dir / DEFAULT_DEPTH_FILENAME
	mask_path = object_dir / DEFAULT_MASK_FILENAME
	output_path = object_dir / output_name

	if not meta_path.exists() or not depth_path.exists():
		logging.debug("Skipping %s (missing depth or metadata)", object_dir.name)
		return False

	if output_path.exists() and not overwrite:
		logging.debug("Skipping %s (point cloud already exists)", object_dir.name)
		return False

	if not mask_path.exists():
		logging.warning("Mask %s not found for %s; skipping", mask_path.name, object_dir.name)
		return False

	metadata = load_metadata(meta_path)
	depth_map = load_depth(depth_path)
	mask = load_mask(mask_path)
	intrinsics = ensure_intrinsics(metadata)
	pose = ensure_camera_pose(metadata, camera_frame)

	# 1. 先生成scene坐标系下的点云
	scene_points = depth_to_points(depth_map, mask, intrinsics, pose, depth_min, depth_max)
	if scene_points.size == 0:
		logging.warning("No valid points generated for %s; skipping output", object_dir.name)
		return False

	# 保存scene坐标点云
	scene_ply_path = object_dir / (output_name.replace('.ply', '_scene.ply'))
	save_pointcloud(scene_points, scene_ply_path)

	# 2. 获取obj2world变换
	# 物体序号：取obj_dict的第一个key
	obj_dict = metadata.get("obj_dict", {})
	if not obj_dict:
		logging.warning("No obj_dict in metadata for %s", object_dir.name)
		return False
	obj_idx = next(iter(obj_dict.keys()))
	obj_info = obj_dict[obj_idx]
	obj_rot = np.asarray(obj_info["obj_rot"], dtype=np.float64)
	obj_tran = np.asarray(obj_info["obj_tran"], dtype=np.float64)
	obj_scale = np.asarray(obj_info["obj_scale"], dtype=np.float64)

	# 构造obj2world矩阵
	obj2world = np.eye(4, dtype=np.float64)
	obj2world[:3, :3] = obj_rot
	obj2world[:3, 3] = obj_tran

	# 3. world->obj: 先减去tran, 再旋转, 再除以scale
	# 先将scene_points扩展为齐次
	num_pts = scene_points.shape[0]
	scene_points_h = np.concatenate([scene_points, np.ones((num_pts, 1), dtype=np.float64)], axis=1)
	# world->obj: x_obj = np.linalg.inv(obj2world) @ x_world
	obj2world_inv = np.linalg.inv(obj2world)
	obj_points_h = (obj2world_inv @ scene_points_h.T).T
	obj_points = obj_points_h[:, :3] / obj_scale

	# 保存obj坐标点云
	obj_ply_path = object_dir / output_name
	save_pointcloud(obj_points, obj_ply_path)

	# 保存npy
	obj_npy_path = object_dir / output_name.replace('.ply', '.npy')
	save_pointcloud_npy(obj_points, obj_npy_path)

	logging.info("Saved %d points to %s (scene), %s (obj), %s (npy)", len(scene_points), scene_ply_path, obj_ply_path, obj_npy_path)
	return True


def main() -> None:
	args = parse_args()
	configure_logging(args.log_level)

	processed = 0
	for idx, object_dir in enumerate(iter_object_dirs(args.dataset_root), start=1):
		if args.limit is not None and processed >= args.limit:
			break

		try:
			result = process_object(
				object_dir,
				args.output_name,
				depth_min=args.depth_min,
				depth_max=args.depth_max,
				overwrite=args.overwrite,
				camera_frame=args.camera_frame,
			)
		except Exception as exc:
			logging.error("Failed to process %s: %s", object_dir.name, exc, exc_info=True)
			continue

		if result:
			processed += 1

	logging.info("Finished generating point clouds for %d objects", processed)


if __name__ == "__main__":
	main()
