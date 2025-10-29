"""Align predicted meshes to the input world coordinate system.

Usage example:
	python amodel3r_align_output.py \
		--input-root data/outputs/test \
		--output-root path/to/method_outputs \
		--aligned-root path/to/aligned_outputs

The script expects each scene directory under ``output-root`` to contain a
``mesh.ply``. The corresponding ``input-root`` scene directory must provide a
``meta-data.json`` describing the camera pose, matching the format consumed by
``Front3D_Recon_Dataset``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
	import trimesh
except ImportError as exc:  # pragma: no cover - import guard
	raise SystemExit(
		"Failed to import trimesh. Install it with `pip install trimesh`."
	) from exc


LOGGER = logging.getLogger("amodel3r_align_output")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input-root",
		type=Path,
		required=True,
		help="Root directory containing reference scene folders with meta-data.json.",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		required=True,
		help="Root directory containing predicted meshes (mesh.ply per scene).",
	)
	parser.add_argument(
		"--aligned-root",
		type=Path,
		default=None,
		help=(
			"Destination root for aligned meshes. Defaults to --output-root in-place "
			"subdirectories."
		),
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite aligned meshes if they already exist.",
	)
	return parser.parse_args(argv)


def configure_logging() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="[%(levelname)s] %(message)s",
	)


def load_camera_pose(meta_path: Path) -> np.ndarray:
	with meta_path.open("r", encoding="utf-8") as fp:
		meta = json.load(fp)

	rotation = np.array(meta["camera_pose_rot"], dtype=np.float64)
	translation = np.array(meta["camera_pose_tran"], dtype=np.float64)

	pose = np.eye(4, dtype=np.float64)
	pose[:3, :3] = rotation
	pose[:3, 3] = translation
	return pose


def align_mesh(mesh_path: Path, pose: np.ndarray) -> trimesh.Trimesh:
	mesh = trimesh.load(mesh_path, process=False)
	if not isinstance(mesh, trimesh.Trimesh):
		raise ValueError(f"Expected a trimesh.Trimesh from {mesh_path}, got {type(mesh)}")

	vertices = mesh.vertices.copy()
	aligned_vertices = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
	mesh.vertices = aligned_vertices
	return mesh


def main(argv: Iterable[str] | None = None) -> int:
	args = parse_args(argv)
	configure_logging()

	input_root: Path = args.input_root.resolve()
	output_root: Path = args.output_root.resolve()
	aligned_root: Path = (
		args.aligned_root.resolve() if args.aligned_root else output_root
	)

	if not input_root.is_dir():
		LOGGER.error("Input root %s does not exist or is not a directory", input_root)
		return 1

	if not output_root.is_dir():
		LOGGER.error("Output root %s does not exist or is not a directory", output_root)
		return 1

	scene_dirs = sorted([p for p in output_root.iterdir() if p.is_dir()])
	if not scene_dirs:
		LOGGER.error("No scene directories found under %s", output_root)
		return 1

	processed = 0
	skipped = 0

	for scene_dir in scene_dirs:
		scene_id = scene_dir.name
		mesh_path = scene_dir / "mesh.ply"
		if not mesh_path.is_file():
			LOGGER.warning("Skip %s: mesh.ply not found", scene_id)
			skipped += 1
			continue

		meta_path = input_root / scene_id / "meta-data.json"
		if not meta_path.is_file():
			LOGGER.warning("Skip %s: meta-data.json not found", scene_id)
			skipped += 1
			continue

		try:
			pose = load_camera_pose(meta_path)
		except (json.JSONDecodeError, KeyError) as exc:
			LOGGER.warning("Skip %s: failed to read pose (%s)", scene_id, exc)
			skipped += 1
			continue

		try:
			aligned_mesh = align_mesh(mesh_path, pose)
		except Exception as exc:  # pragma: no cover - depends on mesh format
			LOGGER.warning("Skip %s: failed to align mesh (%s)", scene_id, exc)
			skipped += 1
			continue

		dest_dir = aligned_root / scene_id
		dest_dir.mkdir(parents=True, exist_ok=True)
		dest_path = dest_dir / "mesh_world.ply"

		if dest_path.exists() and not args.overwrite:
			LOGGER.info("Skip %s: %s exists (use --overwrite)", scene_id, dest_path)
			skipped += 1
			continue

		aligned_mesh.export(dest_path)
		LOGGER.info("Aligned %s -> %s", scene_id, dest_path)
		processed += 1

	LOGGER.info("Done. %d aligned, %d skipped.", processed, skipped)
	return 0 if processed else 1


if __name__ == "__main__":
	sys.exit(main())

