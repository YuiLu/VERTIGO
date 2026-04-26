#!/usr/bin/env python3
"""Trajectory visualization for camera transforms JSON.

This utility is Blender-free and supports common trajectory intrinsics:
- top-level intrinsics: fl_x/fl_y/cx/cy/w/h
- per-frame intrinsics: frames[i].fl_x/fl_y/cx/cy/w/h

It produces static PNG visualizations (3D trajectory + sampled camera frustums).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def calculate_fov_from_intrinsics(fx: float, fy: float, width: float, height: float) -> Tuple[float, float]:
    h_fov = 2.0 * np.arctan(width / (2.0 * fx)) * 180.0 / np.pi
    v_fov = 2.0 * np.arctan(height / (2.0 * fy)) * 180.0 / np.pi
    return float(h_fov), float(v_fov)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize camera trajectories without Blender")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input transforms json file OR directory containing *_transforms_cleaning.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for PNG files. Defaults to <input>/traj_viz for directory input, or file parent for file input.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_transforms_cleaning.json",
        help="Glob pattern used when input is a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search json files when input is a directory.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If > 0, only process first N files.",
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="first_frame",
        choices=["none", "first_frame", "origin_center"],
        help="Trajectory normalization for visualization only.",
    )
    parser.add_argument(
        "--draw-frustums",
        action="store_true",
        help="Draw sampled camera frustums using per-frame intrinsics.",
    )
    parser.add_argument(
        "--frustum-count",
        type=int,
        default=12,
        help="Approximate number of frustums to draw along trajectory.",
    )
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=0.3,
        help="Relative frustum size factor with respect to scene size.",
    )
    parser.add_argument(
        "--default-fov",
        type=float,
        default=60.0,
        help="Fallback vertical FOV if intrinsics are missing.",
    )
    parser.add_argument(
        "--title-from-caption",
        action="store_true",
        help="If enabled, read sibling *_caption.json and use Movement as title.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    return parser.parse_args()


def _get_intrinsic(frame: dict, top: dict, key: str) -> Optional[float]:
    v = frame.get(key, top.get(key))
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_transforms(json_path: Path, default_vfov: float) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("Missing frames[] in transforms json")

    poses = []
    vfovs = []
    for frame in frames:
        mat = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"Invalid transform_matrix shape: {mat.shape}")
        poses.append(mat)

        fy = _get_intrinsic(frame, data, "fl_y")
        h = _get_intrinsic(frame, data, "h")
        fx = _get_intrinsic(frame, data, "fl_x")
        w = _get_intrinsic(frame, data, "w")
        if fy is not None and h is not None and fy > 0 and h > 0:
            if fx is None:
                fx = fy
            if w is None:
                w = h
            _, vfov = calculate_fov_from_intrinsics(fx, fy, w, h)
        else:
            vfov = float(default_vfov)
        vfovs.append(vfov)

    return np.stack(poses, axis=0), np.asarray(vfovs, dtype=np.float64)


def normalize_poses(c2ws: np.ndarray, mode: str) -> np.ndarray:
    out = c2ws.copy()
    if mode == "none":
        return out
    if mode == "first_frame":
        ref_w2c = np.linalg.inv(out[:1])
        out = np.matmul(np.repeat(ref_w2c, out.shape[0], axis=0), out)
        return out
    if mode == "origin_center":
        t0 = out[0, :3, 3].copy()
        out[:, :3, 3] -= t0[None, :]
        return out
    raise ValueError(f"Unknown normalize mode: {mode}")


def create_frustum(pose_matrix: np.ndarray, fov: float, scale: float) -> np.ndarray:
    fov_rad = math.radians(float(fov))
    far = float(scale)
    height = 2.0 * far * math.tan(fov_rad / 2.0)
    width = height * (16.0 / 9.0)

    vertices_cam = np.array(
        [
            [0.0, 0.0, 0.0],
            [width / 2.0, height / 2.0, far],
            [-width / 2.0, height / 2.0, far],
            [-width / 2.0, -height / 2.0, far],
            [width / 2.0, -height / 2.0, far],
        ],
        dtype=np.float64,
    )

    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]
    vertices_world = vertices_cam @ R.T + t[None, :]
    return vertices_world


def draw_frustum_on_axis(ax, pose_matrix: np.ndarray, fov: float, scale: float, color, alpha: float = 0.6) -> None:
    v = create_frustum(pose_matrix, fov=fov, scale=scale)

    # plot with same axis mapping used for trajectory: (x, z, -y)
    for j in range(1, 5):
        ax.plot(
            [v[0, 0], v[j, 0]],
            [v[0, 2], v[j, 2]],
            [-v[0, 1], -v[j, 1]],
            color=color,
            alpha=alpha,
            linewidth=1.2,
        )

    face_indices = [1, 2, 3, 4, 1]
    ax.plot(
        v[face_indices, 0],
        v[face_indices, 2],
        -v[face_indices, 1],
        color=color,
        alpha=alpha,
        linewidth=1.2,
    )

    verts = [[(v[i, 0], v[i, 2], -v[i, 1]) for i in [1, 2, 3, 4]]]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color, alpha=0.12, linewidths=0.0))


def maybe_get_title(json_path: Path, title_from_caption: bool) -> str:
    stem = json_path.name.replace("_transforms_cleaning.json", "")
    if not title_from_caption:
        return stem

    caption_path = json_path.with_name(f"{stem}_caption.json")
    if not caption_path.exists():
        return stem
    try:
        data = json.loads(caption_path.read_text(encoding="utf-8"))
        return str(data.get("Movement", stem))
    except Exception:
        return stem


def visualize_trajectory(c2ws: np.ndarray, vfovs: np.ndarray, vis_path: Path, title: str, draw_frustums: bool, frustum_count: int, frustum_scale: float) -> None:
    positions = c2ws[:, :3, 3]
    num_frames = len(c2ws)

    if num_frames < 2:
        raise ValueError("Need at least 2 frames to visualize trajectory")

    colors = plt.cm.rainbow(np.linspace(0, 1, num_frames))

    bounds_points = [positions, np.zeros((1, 3), dtype=positions.dtype)]
    base_points = np.concatenate(bounds_points, axis=0)
    base_min = np.min(base_points, axis=0)
    base_max = np.max(base_points, axis=0)
    base_span = np.maximum(base_max - base_min, 1e-6)
    base_range = max(float(np.max(base_span)) * 0.6, 0.3)
    scene_size = max(float(np.linalg.norm(base_span)), base_range * 2.0, 0.6)

    frustum_indices: List[int] = []
    fr_scale = scene_size * float(frustum_scale)
    if draw_frustums:
        step = max(1, num_frames // max(1, frustum_count))
        frustum_indices = list(range(0, num_frames, step))
        if frustum_indices[-1] != num_frames - 1:
            frustum_indices.append(num_frames - 1)
        frustum_vertices = [
            create_frustum(c2ws[i], fov=float(vfovs[i]), scale=fr_scale)
            for i in frustum_indices
        ]
        if frustum_vertices:
            bounds_points.extend(frustum_vertices)

    all_bounds = np.concatenate(bounds_points, axis=0)
    min_vals = np.min(all_bounds, axis=0)
    max_vals = np.max(all_bounds, axis=0)
    center = (min_vals + max_vals) / 2.0
    span = np.maximum(max_vals - min_vals, 1e-6)
    max_range = max(float(np.max(span)) * 0.6, 0.3)

    fig = plt.figure(figsize=(12, 10), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    for i in range(num_frames - 1):
        ax.plot(
            positions[i : i + 2, 0],
            positions[i : i + 2, 2],
            -positions[i : i + 2, 1],
            color=colors[i],
            linewidth=1.8,
        )

    start_end_same = np.linalg.norm(positions[-1] - positions[0]) < 1e-6
    if start_end_same:
        ax.scatter(positions[0, 0], positions[0, 2], -positions[0, 1], facecolors="none", edgecolors="green", linewidths=2.5, s=170, marker="o", label="Start")
    else:
        ax.scatter(positions[0, 0], positions[0, 2], -positions[0, 1], c="green", s=80, marker="o", label="Start")
    ax.scatter(positions[-1, 0], positions[-1, 2], -positions[-1, 1], c="red", s=80, marker="o", label="End")
    ax.scatter([0], [0], [0], color="black", s=90, marker="x", label="Origin")

    if draw_frustums:
        for i in frustum_indices:
            draw_frustum_on_axis(ax, c2ws[i], fov=float(vfovs[i]), scale=fr_scale, color=colors[i], alpha=0.65)

    # Axis helper
    axis_len = max_range * 0.25
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="red", arrow_length_ratio=0.12, linewidth=2)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="blue", arrow_length_ratio=0.12, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="green", arrow_length_ratio=0.12, linewidth=2)

    # Set view bounds with mapped axes
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[2] - max_range, center[2] + max_range)
    ax.set_zlim(-center[1] - max_range, -center[1] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")
    ax.set_title(title, fontsize=11, pad=16)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.05)
    cbar.set_label("Trajectory Progress")

    ax.legend(loc="upper right", fontsize=9)

    vis_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(vis_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def collect_json_files(input_path: Path, pattern: str, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if recursive:
        return sorted(input_path.rglob(pattern))
    return sorted(input_path.glob(pattern))


def run(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = collect_json_files(input_path, pattern=args.glob, recursive=args.recursive)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise RuntimeError(f"No files matched under {input_path} with pattern {args.glob}")

    if args.output_dir is None:
        if input_path.is_file():
            output_dir = input_path.parent
        else:
            output_dir = input_path / "traj_viz"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    bad = 0
    for json_path in files:
        stem = json_path.name.replace("_transforms_cleaning.json", "")
        vis_path = output_dir / f"{stem}_traj.png"

        if vis_path.exists() and not args.overwrite:
            continue

        try:
            c2ws, vfovs = load_transforms(json_path, default_vfov=args.default_fov)
            c2ws = normalize_poses(c2ws, mode=args.normalize_mode)
            title = maybe_get_title(json_path, title_from_caption=args.title_from_caption)
            visualize_trajectory(
                c2ws=c2ws,
                vfovs=vfovs,
                vis_path=vis_path,
                title=title,
                draw_frustums=args.draw_frustums,
                frustum_count=args.frustum_count,
                frustum_scale=args.frustum_scale,
            )
            ok += 1
        except Exception as e:
            bad += 1
            print(f"[WARN] Failed: {json_path} | {e}")

    print(f"[DONE] output_dir={output_dir} ok={ok} failed={bad}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
