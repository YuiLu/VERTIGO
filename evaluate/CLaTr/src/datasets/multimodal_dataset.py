import random
from copy import deepcopy as dp
from pathlib import Path

from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    def __init__(
        self,
        name,
        dataset_name,
        dataset_dir,
        trajectory,
        feature_type,
        num_rawfeats,
        num_feats,
        num_cams,
        num_cond_feats,
        num_traj_feats,
        standardization,
        **modalities,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.name = name
        self.dataset_name = dataset_name
        self.feature_type = feature_type
        self.num_rawfeats = num_rawfeats
        self.num_feats = num_feats
        self.num_cams = num_cams
        self.trajectory_dataset = trajectory
        self.standardization = standardization
        self.modality_datasets = modalities

    # --------------------------------------------------------------------------------- #

    def set_split(self, split: str, train_rate: float = 1.0):
        self.split = split

        # Get trajectory split
        self.trajectory_dataset = dp(self.trajectory_dataset).set_split(
            split, train_rate
        )
        self.root_filenames = self.trajectory_dataset.filenames

        # Get modality split
        for modality_name in self.modality_datasets.keys():
            self.modality_datasets[modality_name].filenames = self.root_filenames

        self.get_feature = self.trajectory_dataset.get_feature
        self.get_matrix = self.trajectory_dataset.get_matrix

        return self

    # --------------------------------------------------------------------------------- #

    def __getitem__(self, index):
        output = self.trajectory_dataset[index]
        if len(output) == 3:
            # print("Start training ...")
            trajectory_filename, trajectory_feature, padding_mask = self.trajectory_dataset[index]

            out = {
                "traj_filename": trajectory_filename,
                "traj_feat": trajectory_feature.permute(1, 0),
                "padding_mask": padding_mask.to(bool),
            }

            # print(self.modality_datasets)
            for modality_name, modality_dataset in self.modality_datasets.items():
                modality_filename, modality_feature, modality_raw = modality_dataset[index]
                assert trajectory_filename.split(".")[0].replace('_transforms_cleaning', '') == modality_filename.split(".")[0]
                out[f"{modality_name}_filename"] = modality_filename
                out[f"{modality_name}_feat"] = modality_feature
                out[f"{modality_name}_raw"] = modality_raw
                out[f"{modality_name}_padding_mask"] = padding_mask

            return out
        
        elif len(output) == 6:
            # print("Start extraction ...")
            traj_ref_filename, traj_ref_feature, padding_mask_ref, traj_pred_filename, traj_pred_feature, padding_mask_pred = self.trajectory_dataset[index]
            out = {
                "traj_ref_filename": traj_ref_filename,
                "traj_ref_feat": traj_ref_feature.permute(1, 0),
                "padding_mask_ref": padding_mask_ref.to(bool),
                "traj_pred_filename": traj_pred_filename,
                "traj_pred_feat": traj_pred_feature.permute(1, 0),
                "padding_mask_pred": padding_mask_pred.to(bool),
            }

            for modality_name, modality_dataset in self.modality_datasets.items():
                modality_filename, modality_feature, modality_raw = modality_dataset[index]
                assert traj_ref_filename.split(".")[0].replace('_transforms_ref', '') == modality_filename.split(".")[0]
                out[f"{modality_name}_filename"] = modality_filename
                out[f"{modality_name}_feat"] = modality_feature
                out[f"{modality_name}_raw"] = modality_raw
                out[f"{modality_name}_padding_mask"] = padding_mask_ref

            return out

    def __len__(self):
        return len(self.trajectory_dataset)
