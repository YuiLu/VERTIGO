import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.options import Options
from core.utils import camera_to_token_single, project_world_origin_to_screen, standardize_translation


def _as_path(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "transforms", "transforms_path", "trajectory", "trajectory_path"):
            if key in value and isinstance(value[key], str):
                return value[key]
    return None


def _resolve_path(path_value: str, base_dir: Path, data_root: Optional[str]) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidates = []
    if data_root:
        candidates.append(Path(data_root) / path)
    candidates.append(base_dir / path)
    candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            record["_source_file"] = str(path)
            records.append(record)
    return records


def _collect_records(preference_path: str) -> List[Dict[str, Any]]:
    path = Path(preference_path)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        records = []
        for file in files:
            records.extend(_read_jsonl(file))
        return records
    return _read_jsonl(path)


def _get_first(record: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return default


def _get_intrinsic(frame: Dict[str, Any], top: Dict[str, Any], key: str) -> Optional[float]:
    value = frame.get(key, top.get(key))
    if value is None:
        return None
    return float(value)


def _standard_image(rgb_path: str, target_height: int, target_width: int) -> torch.Tensor:
    image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {rgb_path}")
    image = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image[..., [2, 1, 0]]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    height, width = image_tensor.shape[1], image_tensor.shape[2]

    if height > target_height:
        start_y = (height - target_height) // 2
        image_tensor = image_tensor[:, start_y : start_y + target_height, :]
    if width > target_width:
        start_x = (width - target_width) // 2
        image_tensor = image_tensor[:, :, start_x : start_x + target_width]

    if image_tensor.shape[1] < target_height or image_tensor.shape[2] < target_width:
        padded_image = torch.zeros((3, target_height, target_width), dtype=torch.float32)
        top_padding = (target_height - image_tensor.shape[1]) // 2
        left_padding = (target_width - image_tensor.shape[2]) // 2
        padded_image[
            :, top_padding : top_padding + image_tensor.shape[1], left_padding : left_padding + image_tensor.shape[2]
        ] = image_tensor
        image_tensor = padded_image
    return image_tensor


def _standard_depth(depth_path: str, target_height: int, target_width: int) -> torch.Tensor:
    depth_image = np.load(depth_path).astype(np.float32)
    depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).float()
    height, width = depth_tensor.shape[1], depth_tensor.shape[2]

    if height > target_height:
        start_y = (height - target_height) // 2
        depth_tensor = depth_tensor[:, start_y : start_y + target_height, :]
    if width > target_width:
        start_x = (width - target_width) // 2
        depth_tensor = depth_tensor[:, :, start_x : start_x + target_width]

    if depth_tensor.shape[1] < target_height or depth_tensor.shape[2] < target_width:
        padded_depth = torch.zeros((1, target_height, target_width), dtype=torch.float32)
        top_padding = (target_height - depth_tensor.shape[1]) // 2
        left_padding = (target_width - depth_tensor.shape[2]) // 2
        padded_depth[
            :, top_padding : top_padding + depth_tensor.shape[1], left_padding : left_padding + depth_tensor.shape[2]
        ] = depth_tensor
        depth_tensor = padded_depth
    return depth_tensor


class TrajectoryDpoDataset(Dataset):
    """Preference pairs for DPO post-training of the trajectory generator.

    Expected JSONL fields:
    - prompt/text/caption: natural-language condition.
    - chosen/chosen_path/preferred: path to preferred transforms JSON.
    - rejected/rejected_path/dispreferred: path to rejected transforms JSON.
    """

    def __init__(self, opt: Options):
        if opt.dpo_preference_path is None:
            raise ValueError("--dpo-preference-path is required for DPO training")
        self.opt = opt
        self.records = _collect_records(opt.dpo_preference_path)
        if opt.dpo_score_margin > 0:
            self.records = [record for record in self.records if self._passes_margin(record)]
        if opt.dpo_max_pairs > 0:
            self.records = self.records[: opt.dpo_max_pairs]
        if not self.records:
            raise RuntimeError(f"No DPO preference pairs loaded from {opt.dpo_preference_path}")

    def _passes_margin(self, record: Dict[str, Any]) -> bool:
        chosen_score = _get_first(record, ["chosen_score", "positive_score", "winner_score"])
        rejected_score = _get_first(record, ["rejected_score", "negative_score", "loser_score"])
        if chosen_score is None or rejected_score is None:
            return True
        return float(chosen_score) - float(rejected_score) >= float(self.opt.dpo_score_margin)

    def __len__(self) -> int:
        return len(self.records)

    def _load_transforms(self, traj_ref: Any, base_dir: Path) -> Dict[str, Any]:
        if isinstance(traj_ref, dict) and "frames" in traj_ref:
            return traj_ref
        traj_path = _as_path(traj_ref)
        if traj_path is None:
            raise ValueError(f"Trajectory entry must be a path or transforms dict, got: {type(traj_ref)}")
        resolved = _resolve_path(traj_path, base_dir=base_dir, data_root=self.opt.dpo_data_root)
        with resolved.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _transforms_to_coords(self, transforms_json: Dict[str, Any]) -> torch.Tensor:
        frames = transforms_json.get("frames", [])
        if not isinstance(frames, list) or not frames:
            raise ValueError("Missing frames[] in transforms JSON")

        if len(frames) >= self.opt.pose_length:
            input_view_indices = np.linspace(0, len(frames) - 1, self.opt.pose_length).round().astype(int)
        else:
            input_view_indices = np.arange(len(frames))

        top_h = transforms_json.get("h")
        top_w = transforms_json.get("w")
        top_fx = transforms_json.get("fl_x")
        top_fy = transforms_json.get("fl_y")
        top_cx = transforms_json.get("cx")
        top_cy = transforms_json.get("cy")

        c2ws = []
        intrinsics = []
        for index in input_view_indices:
            frame = frames[int(index)]
            matrix = np.asarray(frame["transform_matrix"], dtype=np.float32)
            if matrix.shape != (4, 4):
                raise ValueError(f"Invalid transform_matrix shape: {matrix.shape}")

            w = _get_intrinsic(frame, transforms_json, "w") or top_w
            h = _get_intrinsic(frame, transforms_json, "h") or top_h
            fx = _get_intrinsic(frame, transforms_json, "fl_x") or top_fx
            fy = _get_intrinsic(frame, transforms_json, "fl_y") or top_fy
            cx = _get_intrinsic(frame, transforms_json, "cx") or top_cx
            cy = _get_intrinsic(frame, transforms_json, "cy") or top_cy
            if None in (w, h, fx, fy, cx, cy):
                raise ValueError("Missing intrinsics in transforms JSON")

            c2ws.append(matrix[:3, :])
            intrinsics.append(
                np.array(
                    [
                        float(fx) * self.opt.target_width / float(w),
                        float(fy) * self.opt.target_height / float(h),
                        float(cx) * self.opt.target_width / float(w),
                        float(cy) * self.opt.target_height / float(h),
                        self.opt.target_width,
                        self.opt.target_height,
                    ],
                    dtype=np.float32,
                )
            )

        c2ws = torch.from_numpy(np.stack(c2ws, axis=0)).float()
        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()

        def matrix_to_square(mat: torch.Tensor) -> torch.Tensor:
            bottom = torch.tensor([0, 0, 0, 1], device=mat.device, dtype=mat.dtype)
            return torch.cat([mat, bottom.repeat(mat.shape[0], 1, 1)], dim=1)

        scale = 1.0
        if self.opt.camera_norm_mode == "first_frame":
            ref_w2c = torch.inverse(matrix_to_square(c2ws[:1]))
            c2ws = (ref_w2c.repeat(c2ws.shape[0], 1, 1) @ matrix_to_square(c2ws))[:, :3, :]
            t_norm = c2ws[:, :3, 3].norm(dim=-1).max()
            scale = float(t_norm + 1e-5)
            c2ws[:, :3, 3] = c2ws[:, :3, 3] / scale
        elif self.opt.camera_norm_mode == "origin_global":
            c2ws[:, :3, 3] = c2ws[:, :3, 3] - c2ws[:1, :3, 3]
            if self.opt.camera_translation_norm == "dataset_p99":
                c2ws[:, :3, 3] = standardize_translation(c2ws[:, :3, 3], self.opt, anchor=False)
            else:
                scale = float(self.opt.camera_global_scale)
                c2ws[:, :3, 3] = c2ws[:, :3, 3] / scale
        elif self.opt.camera_norm_mode == "none":
            scale = 1.0
        else:
            raise ValueError(f"Unknown camera_norm_mode: {self.opt.camera_norm_mode}")

        cameras = torch.cat([c2ws.flatten(1, 2), intrinsics], dim=1)
        camera_tokens = camera_to_token_single(cameras)
        coords_traj = ((camera_tokens[:, :7] + 1) * 0.5 * self.opt.discrete_bins).clip(0, self.opt.discrete_bins).long()
        coords_instri = (camera_tokens[:, 7:] / 10 * self.opt.discrete_bins).clip(0, self.opt.discrete_bins).long()
        coords_scale = (
            torch.tensor((math.log10(scale) + 2) / 4 * self.opt.discrete_bins)
            .expand(coords_instri.shape[0], 1)
            .clip(0, self.opt.discrete_bins)
            .long()
        )
        return torch.cat([coords_traj, coords_instri, coords_scale], dim=1).flatten() + 3

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        base_dir = Path(record.get("_source_file", self.opt.dpo_preference_path)).parent

        prompt = _get_first(record, [self.opt.dpo_prompt_key, "prompt", "text", "caption"])
        if isinstance(prompt, dict):
            prompt = _get_first(prompt, [self.opt.text_key, "Movement", "Concise Interaction", "text"])
        if prompt is None:
            raise ValueError(f"Missing prompt/text field in DPO record {idx}")

        chosen_ref = _get_first(record, [self.opt.dpo_chosen_key, "chosen", "chosen_path", "preferred", "positive"])
        rejected_ref = _get_first(record, [self.opt.dpo_rejected_key, "rejected", "rejected_path", "dispreferred", "negative"])
        if chosen_ref is None or rejected_ref is None:
            raise ValueError(f"Missing chosen/rejected trajectory in DPO record {idx}")

        chosen_json = self._load_transforms(chosen_ref, base_dir=base_dir)
        rejected_json = self._load_transforms(rejected_ref, base_dir=base_dir)

        item = {
            "text": str(prompt),
            "chosen_coords": self._transforms_to_coords(chosen_json),
            "rejected_coords": self._transforms_to_coords(rejected_json),
            "meta": record,
        }

        if self.opt.cond_mode == "text":
            item["rgb"] = torch.zeros((3, self.opt.target_height, self.opt.target_width), dtype=torch.float32)
            item["depth"] = torch.zeros((1, self.opt.target_height, self.opt.target_width), dtype=torch.float32)
        else:
            image_ref = _get_first(record, ["image", "image_path", "rgb", "rgb_path"])
            depth_ref = _get_first(record, ["depth", "depth_path"])
            if self.opt.cond_mode in {"image", "image+text", "image+depth", "depth+image+text"} and image_ref is None:
                raise ValueError("Non-text DPO modes require image_path/rgb_path in each preference record")
            if self.opt.cond_mode in {"image+depth", "depth+image+text"} and depth_ref is None:
                raise ValueError("Depth-conditioned DPO modes require depth_path in each preference record")
            image_path_value = _as_path(image_ref)
            if image_path_value is None:
                raise ValueError("image_path/rgb_path must be a string path or an object with a path field")
            image_path = _resolve_path(image_path_value, base_dir=base_dir, data_root=self.opt.dpo_data_root)
            item["rgb"] = _standard_image(str(image_path), self.opt.target_height, self.opt.target_width)
            if depth_ref is None:
                item["depth"] = torch.zeros((1, self.opt.target_height, self.opt.target_width), dtype=torch.float32)
            else:
                depth_path_value = _as_path(depth_ref)
                if depth_path_value is None:
                    raise ValueError("depth_path must be a string path or an object with a path field")
                depth_path = _resolve_path(depth_path_value, base_dir=base_dir, data_root=self.opt.dpo_data_root)
                item["depth"] = _standard_depth(str(depth_path), self.opt.target_height, self.opt.target_width)

        return item


def _build_sequence(coords: torch.Tensor, max_len: int, opt: Options) -> Dict[str, np.ndarray]:
    coords_np = coords.detach().cpu().numpy().astype(np.int64)
    coords_len = min(coords_np.shape[0], max_len)
    pad_len = max_len - coords_len
    include_eos = coords_np.shape[0] <= max_len
    tail_token = opt.eos_token_id if include_eos else opt.pad_token_id
    tail_label = opt.eos_token_id if include_eos else -100
    tail_mask = 1 if include_eos else 0

    tokens = np.concatenate(
        [
            np.full((1,), opt.bos_token_id),
            coords_np[:coords_len],
            np.full((1,), tail_token),
            np.full((pad_len,), opt.pad_token_id),
        ],
        axis=0,
    )
    labels = np.concatenate(
        [
            np.full((opt.num_cond_tokens + 1,), -100),
            coords_np[:coords_len],
            np.full((1,), tail_label),
            np.full((pad_len,), -100),
        ],
        axis=0,
    )
    mask = np.concatenate(
        [
            np.ones(opt.num_cond_tokens + 1 + coords_len),
            np.full((1,), tail_mask),
            np.zeros(pad_len),
        ],
        axis=0,
    )
    num_tokens = opt.num_cond_tokens + 1 + coords_len + tail_mask

    return {"tokens": tokens, "labels": labels, "masks": mask, "num_tokens": np.array(num_tokens, dtype=np.int64)}


def dpo_collate_fn(batch: List[Dict[str, Any]], opt: Options) -> Dict[str, Any]:
    max_len = min(
        max(max(item["chosen_coords"].shape[0], item["rejected_coords"].shape[0]) for item in batch),
        opt.max_seq_length,
    )

    def make_side(side: str) -> Dict[str, Any]:
        seqs = [_build_sequence(item[f"{side}_coords"], max_len=max_len, opt=opt) for item in batch]
        return {
            "tokens": torch.from_numpy(np.stack([seq["tokens"] for seq in seqs], axis=0)).long(),
            "labels": torch.from_numpy(np.stack([seq["labels"] for seq in seqs], axis=0)).long(),
            "masks": torch.from_numpy(np.stack([seq["masks"] for seq in seqs], axis=0)).bool(),
            "num_tokens": torch.from_numpy(np.stack([seq["num_tokens"] for seq in seqs], axis=0)).long(),
            "text": [item["text"] for item in batch],
            "rgb": torch.stack([item["rgb"] for item in batch], dim=0).float(),
            "depth": torch.stack([item["depth"] for item in batch], dim=0).float(),
        }

    return {
        "chosen": make_side("chosen"),
        "rejected": make_side("rejected"),
        "texts": [item["text"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
