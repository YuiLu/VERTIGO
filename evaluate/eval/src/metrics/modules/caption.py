from itertools import product
from typing import List, Tuple

from evo.core import lie_algebra as lie
import numpy as np
import torch
from scipy.stats import mode
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import torchmetrics.functional as F
from torchtyping import TensorType

# ------------------------------------------------------------------------------------- #

num_samples, num_cams, num_total_cams, num_classes = None, None, None, None
width, height = None, None

# ------------------------------------------------------------------------------------- #


ANG_INDEX_TO_PATTERN = {
    0: "static",  # No rotation
    1: "pitch up",  
    2: "pitch down", 
    3: "yaw left",
    4: "yaw right",
    5: "roll left",
    6: "roll right",
}

CAM_INDEX_TO_PATTERN = {
    0: "static",
    1: "move backward",  # keep "move forward" as it is
    2: "move forward",  # keep "move backward" as it is
    3: "move up",
    6: "move down",
    18: "move left",
    9: "move right",
    # ----- #
    12: "move right and up",
    15: "move right and down",
    21: "move left and up",
    24: "move left and down",
    10: "move right and backward",
    11: "move right and forward",
    19: "move left and backward",
    20: "move left and forward",
    4: "move up and backward",
    5: "move up and forward",
    7: "move down and backward",
    8: "move down and forward",
    # ----- #
    13: "move right, up, and backward",
    14: "move right, up, and forward",
    16: "move right, down, and backward",
    17: "move right, down, and forward",
    22: "move left, up, and backward",
    23: "move left, up, and forward",
    25: "move left, down, and backward",
    26: "move left, down, and forward"
}

# ------------------------------------------------------------------------------------- #


def to_euler_angles(
    rotation_mat: TensorType["num_samples", 3, 3]
) -> TensorType["num_samples", 3]:
    rotation_vec = torch.from_numpy(
        np.stack(
            [lie.sst_rotation_from_matrix(r).as_rotvec() for r in rotation_mat.numpy()]
        )
    )
    return rotation_vec


# def compute_relative(f_t: TensorType["num_samples", 3]):
#     max_value = np.max(np.stack([abs(f_t[:, 0]), abs(f_t[:, 1])]), axis=0)
#     xy_f_t = np.divide(
#         (abs(f_t[:, 0]) - abs(f_t[:, 1])),
#         max_value,
#         out=np.zeros_like(max_value),
#         where=max_value != 0,
#     )
#     max_value = np.max(np.stack([abs(f_t[:, 0]), abs(f_t[:, 2])]), axis=0)
#     xz_f_t = np.divide(
#         abs(f_t[:, 0]) - abs(f_t[:, 2]),
#         max_value,
#         out=np.zeros_like(max_value),
#         where=max_value != 0,
#     )
#     max_value = np.max(np.stack([abs(f_t[:, 1]), abs(f_t[:, 2])]), axis=0)
#     yz_f_t = np.divide(
#         abs(f_t[:, 1]) - abs(f_t[:, 2]),
#         max_value,
#         out=np.zeros_like(max_value),
#         where=max_value != 0,
#     )
#     return xy_f_t, xz_f_t, yz_f_t


def compute_relative(f_t: np.ndarray):
    # 计算每个轴的绝对值，只计算一次
    abs_x = np.abs(f_t[:, 0])
    abs_y = np.abs(f_t[:, 1])
    abs_z = np.abs(f_t[:, 2])
    
    # 计算每对轴的最大值
    max_xy = np.maximum(abs_x, abs_y)
    max_xz = np.maximum(abs_x, abs_z)
    max_yz = np.maximum(abs_y, abs_z)
    
    # 计算相对差异，并处理除零错误
    xy_f_t = np.divide(abs_x - abs_y, max_xy, out=np.zeros_like(max_xy), where=max_xy != 0)
    xz_f_t = np.divide(abs_x - abs_z, max_xz, out=np.zeros_like(max_xz), where=max_xz != 0)
    yz_f_t = np.divide(abs_y - abs_z, max_yz, out=np.zeros_like(max_yz), where=max_yz != 0)

    return xy_f_t, xz_f_t, yz_f_t

def compute_camera_dynamics(w2c_poses: TensorType["num_samples", 4, 4], fps: float):
    w2c_poses_inv = torch.from_numpy(
        np.array([lie.se3_inverse(t) for t in w2c_poses.numpy()])
    )
    velocities = w2c_poses_inv[:-1].to(float) @ w2c_poses[1:].to(float)

    # --------------------------------------------------------------------------------- #
    # Translation velocity
    t_velocities = fps * velocities[:, :3, 3]
    t_xy_velocity, t_xz_velocity, t_yz_velocity = compute_relative(t_velocities)
    t_vels = (t_velocities, t_xy_velocity, t_xz_velocity, t_yz_velocity)
    # --------------------------------------------------------------------------------- #
    # # Rotation velocity
    a_velocities = to_euler_angles(velocities[:, :3, :3])
    a_xy_velocity, a_xz_velocity, a_yz_velocity = compute_relative(a_velocities)
    a_vels = (a_velocities, a_xy_velocity, a_xz_velocity, a_yz_velocity)

    return velocities, t_vels, a_vels


# ------------------------------------------------------------------------------------- #


def perform_segmentation(
    velocities: TensorType["num_frames-1", 3],
    xy_velocity: TensorType["num_frames-1", 3],
    xz_velocity: TensorType["num_frames-1", 3],
    yz_velocity: TensorType["num_frames-1", 3],
    static_threshold: float,
    diff_threshold: float,
    smoothing_window_size
) -> List[int]:
    segments = torch.zeros(velocities.shape[0])
    segment_patterns = [torch.tensor(x) for x in product([0, 1, -1], repeat=3)]
    pattern_to_index = {
        tuple(pattern.numpy()): index for index, pattern in enumerate(segment_patterns)
    }

    segmentation_list = []
    for sample_index, sample_velocity in enumerate(velocities):
        sample_pattern = abs(sample_velocity) > static_threshold
        # diff_abs = abs(torch.max(velocities[sample_index])) * diff_threshold
        # print("diff_abs", diff_abs)
        # XY
        if (sample_pattern == torch.tensor([1, 1, 0])).all():
            if xy_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([1, 0, 0])
            elif xy_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 1, 0])

        # XZ
        elif (sample_pattern == torch.tensor([1, 0, 1])).all():
            if xz_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([1, 0, 0])
            elif xz_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 0, 1])

        # YZ
        elif (sample_pattern == torch.tensor([0, 1, 1])).all():
            if yz_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([0, 1, 0])
            elif yz_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 0, 1])

        # XYZ
        elif (sample_pattern == torch.tensor([1, 1, 1])).all():
            if xy_velocity[sample_index] > diff_threshold:
                sample_pattern[1] = 0
            elif xy_velocity[sample_index] < -diff_threshold:
                sample_pattern[0] = 0

            if xz_velocity[sample_index] > diff_threshold:
                sample_pattern[2] = 0
            elif xz_velocity[sample_index] < -diff_threshold:
                sample_pattern[0] = 0

            if yz_velocity[sample_index] > diff_threshold:
                sample_pattern[2] = 0
            elif yz_velocity[sample_index] < -diff_threshold:
                sample_pattern[1] = 0

        sample_pattern = torch.sign(sample_velocity) * sample_pattern
        # print("Sample pattern", sample_pattern)
        # print(velocities[sample_index])
        # print(velocities[sample_index])
        
        segments[sample_index] = pattern_to_index[tuple(sample_pattern.numpy())]
        segmentation_list.append(sample_pattern.numpy())
    
    segmentation_list = np.array(segmentation_list, dtype=int)
    # print("Segmentation list", segmentation_list.shape)
    # xy_segments = segmentation_list[:, 0]
    # xz_segments = segmentation_list[:, 1]
    # yz_segments = segmentation_list[:, 2]
    # # print("XY segments", xy_segments)
    # xy_segments = smooth_segments(xy_segments, smoothing_window_size)
    # xz_segments = smooth_segments(xz_segments, smoothing_window_size)
    # yz_segments = smooth_segments(yz_segments, smoothing_window_size)
    # # print("XY segments", xy_segments)
    # segmentation_list = np.column_stack((xy_segments, xz_segments, yz_segments))
    for sample_index, sample_velocity in enumerate(velocities):
        segments[sample_index] = pattern_to_index[tuple(segmentation_list[sample_index])]
    
    # print("Segmentation list", segmentation_list)
    return np.array(segments, dtype=int)

def perform_angular_segmentation(
    angular_velocities: TensorType["num_frames-1", 3],
    xy_angular_velocity: TensorType["num_frames-1", 3],
    xz_angular_velocity: TensorType["num_frames-1", 3],
    yz_angular_velocity: TensorType["num_frames-1", 3],
    static_threshold: float,
) -> List[int]:
    segments = torch.zeros(angular_velocities.shape[0])
    segment_patterns = [torch.tensor(x) for x in product([0, 1, -1], repeat=3)]
    segment_patterns = [torch.tensor(x) for x in [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]]
    pattern_to_index = {
        tuple(pattern.numpy()): index for index, pattern in enumerate(segment_patterns)
    }

    segmentation_list = []
    for sample_index, sample_angular_velocity in enumerate(angular_velocities):
        sample_pattern = torch.zeros(3)  # Start with all zeros
        
        # 如果旋转速度大于静止阈值，考虑分割
        if torch.abs(sample_angular_velocity).max() > static_threshold:
            
            # 判断旋转方向，并分配唯一的 1 到某一方向
            if torch.abs(sample_angular_velocity[0]) > max(torch.abs(sample_angular_velocity[1]), torch.abs(sample_angular_velocity[2])):  # X 轴最强
                sample_pattern[0] = 1  # X轴旋转
            elif torch.abs(sample_angular_velocity[1]) > max(torch.abs(sample_angular_velocity[0]), torch.abs(sample_angular_velocity[2])):  # Y 轴最强
                sample_pattern[1] = 1  # Y轴旋转
            elif torch.abs(sample_angular_velocity[2]) > max(torch.abs(sample_angular_velocity[0]), torch.abs(sample_angular_velocity[1])):  # Z 轴最强
                sample_pattern[2] = 1  # Z轴旋转
        
        # 使用符号调整旋转方向（正负方向）
        sample_pattern = torch.sign(sample_angular_velocity) * sample_pattern

        # 通过 pattern_to_index 来为每个 sample_pattern 分配标签
        segments[sample_index] = pattern_to_index[tuple(sample_pattern.numpy())]
        segmentation_list.append(sample_pattern.numpy())

    # 将分段列表转换为numpy数组
    segmentation_list = np.array(segmentation_list, dtype=int)

    # # 对各个轴的分段进行平滑处理
    # xy_angular_segments = segmentation_list[:, 0]
    # xz_angular_segments = segmentation_list[:, 1]
    # yz_angular_segments = segmentation_list[:, 2]

    # # 使用窗口大小进行平滑处理
    # xy_angular_segments = smooth_segments(xy_angular_segments, smoothing_window_size)
    # xz_angular_segments = smooth_segments(xz_angular_segments, smoothing_window_size)
    # yz_angular_segments = smooth_segments(yz_angular_segments, smoothing_window_size)

    # # 重新组合平滑后的分段结果
    # segmentation_list = np.column_stack((xy_angular_segments, xz_angular_segments, yz_angular_segments))

    # 再次为每个样本分配标签
    for sample_index, sample_angular_velocity in enumerate(angular_velocities):
        segments[sample_index] = pattern_to_index[tuple(segmentation_list[sample_index])]

    return np.array(segments, dtype=int)


def smooth_segments(arr: List[int], window_size: int) -> List[int]:
    smoothed_arr = arr.copy()

    if len(arr) < window_size:
        return smoothed_arr

    half_window = window_size // 2
    # Handle the first half_window elements
    for i in range(half_window):
        window = arr[: i + half_window + 1]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    for i in range(half_window, len(arr) - half_window):
        window = arr[i - half_window : i + half_window + 1]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    # Handle the last half_window elements
    for i in range(len(arr) - half_window, len(arr)):
        window = arr[i - half_window :]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    return smoothed_arr


def remove_short_chunks(arr: List[int], min_chunk_size: int) -> List[int]:
    def remove_chunk(chunks):
        if len(chunks) == 1:
            return False, chunks

        chunk_lenghts = [(end - start) + 1 for _, start, end in chunks]
        chunk_index = np.argmin(chunk_lenghts)
        chunk_length = chunk_lenghts[chunk_index]
        if chunk_length < min_chunk_size:
            _, start, end = chunks[chunk_index]

            # Check if the chunk is at the beginning
            if chunk_index == 0:
                segment_r, start_r, end_r = chunks[chunk_index + 1]
                chunks[chunk_index + 1] = (segment_r, start_r - chunk_length, end_r)

            elif chunk_index == len(chunks) - 1:
                segment_l, start_l, end_l = chunks[chunk_index - 1]
                chunks[chunk_index - 1] = (segment_l, start_l, end_l + chunk_length)

            else:
                if chunk_length % 2 == 0:
                    half_length_l = chunk_length // 2
                    half_length_r = chunk_length // 2
                else:
                    half_length_l = (chunk_length // 2) + 1
                    half_length_r = chunk_length // 2

                segment_l, start_l, end_l = chunks[chunk_index - 1]
                segment_r, start_r, end_r = chunks[chunk_index + 1]
                chunks[chunk_index - 1] = (segment_l, start_l, end_l + half_length_l)
                chunks[chunk_index + 1] = (segment_r, start_r - half_length_r, end_r)

            chunks.pop(chunk_index)

        return chunk_length < min_chunk_size, chunks

    chunks = find_consecutive_chunks(arr)
    keep_removing, chunks = remove_chunk(chunks)
    while keep_removing:
        keep_removing, chunks = remove_chunk(chunks)

    merged_chunks = []
    for segment, start, end in chunks:
        merged_chunks.extend([segment] * ((end - start) + 1))

    return merged_chunks


# ------------------------------------------------------------------------------------- #


def find_consecutive_chunks(arr: List[int]) -> List[Tuple[int, int, int]]:
    chunks = []
    start_index = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            end_index = i - 1
            if end_index >= start_index:
                chunks.append((arr[start_index], start_index, end_index))
            start_index = i

    # Add the last chunk if the array ends with consecutive similar digits
    if start_index < len(arr):
        chunks.append((arr[start_index], start_index, len(arr) - 1))

    return chunks

def count_segments(lst):
    if not lst:
        return 0  # 如果列表为空，返回 0 段

    num_segments = 1  # 至少有一段，初始化为 1
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:  # 如果当前元素与前一个元素不同，表示新的一段开始
            num_segments += 1

    return num_segments

# ------------------------------------------------------------------------------------- #


class CaptionMetrics(Metric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.metric_kwargs = dict(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
            zero_division=0,
        )

        self.fps = 30
        self.cam_static_threshold = 0.02
        self.cam_diff_threshold = 0.4
        self.smoothing_window_size = 18
        self.angular_static_threshold = 0.005
        self.min_chunk_size = 10

        self.add_state("pred_segments", default=[], dist_reduce_fx="cat")
        self.add_state("target_segments", default=[], dist_reduce_fx="cat")

    def segment_camera_trajectories(
        self, w2c_poses: TensorType["num_samples", 4, 4]
    ) -> TensorType["num_samples"]:
        device = w2c_poses.device

        velocities, t_vels, a_vels = compute_camera_dynamics(w2c_poses.cpu(), fps=self.fps)
        t_velocities, t_xy_velocity, t_xz_velocity, t_yz_velocity = t_vels
        a_velocities, a_xy_velocity, a_xz_velocity, a_yz_velocity = a_vels

        cam_segments = perform_segmentation(t_velocities, t_xy_velocity, t_xz_velocity, t_yz_velocity, static_threshold = self.cam_static_threshold, diff_threshold = self.cam_diff_threshold, smoothing_window_size = self.smoothing_window_size)

        angular_segments = perform_angular_segmentation(a_velocities, a_xy_velocity, a_xz_velocity, a_yz_velocity, static_threshold = self.angular_static_threshold)

        combine_segments_origin = [cam_segments[i]*7+angular_segments[i] for i in range(len(cam_segments))]
        # print("Combine segments", combine_segments_origin)
        idx_len = 100
        smoothing_window_size = 15
        min_chunk_size = 10
        while idx_len > 4:
            combine_segments = smooth_segments(combine_segments_origin, smoothing_window_size)
            # print("Smoothed angular segments", combine_segments)
            combine_segments = remove_short_chunks(combine_segments, min_chunk_size)
            # print("Final angular segments", combine_segments)
            idx_len = count_segments(combine_segments)
            smoothing_window_size += 5
            min_chunk_size += 5
            
            # segment_patterns = [[cam_segments[i], angular_segments[i]] for i in range(len(cam_segments))]

        cam_segments = torch.tensor(combine_segments, device=device)
        return cam_segments

    # --------------------------------------------------------------------------------- #

    def update(
        self,
        pred_trajectories: TensorType["num_samples", "num_cams", 4, 4],
        ref_trajectories: TensorType["num_samples", "num_cams", 4, 4],
        mask: TensorType["num_samples", "num_cams"],
    ) -> Tuple[float, float, float]:
        """Update the state with extracted features."""
        for sample_index in range(pred_trajectories.shape[0]):
            pred = pred_trajectories[sample_index][mask[sample_index].to(bool)]
            ref = ref_trajectories[sample_index][mask[sample_index].to(bool)]
            if pred.shape[0] < 2:
                continue

            self.pred_segments.append(self.segment_camera_trajectories(pred))
            self.target_segments.append(self.segment_camera_trajectories(ref))

    def compute(self) -> Tuple[float, float, float]:
        """ """
        target_segments = dim_zero_cat(self.target_segments)
        pred_segments = dim_zero_cat(self.pred_segments)

        # print(target_segments)
        # print(pred_segments)
        precision = F.precision(pred_segments, target_segments, **self.metric_kwargs)
        recall = F.recall(pred_segments, target_segments, **self.metric_kwargs)
        fscore = F.f1_score(pred_segments, target_segments, **self.metric_kwargs)

        return precision, recall, fscore
