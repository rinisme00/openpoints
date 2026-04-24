"""
BreakingBadCls — PointNeXt dataset loader for binary classification of
broken vs. complete 3D objects from the Breaking Bad dataset.

Supports dynamic feature dimensions (e.g. 3D baseline or 9D enriched).
"""

from __future__ import annotations

import os
import glob
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..build import DATASETS


def _load_h5_files(
    data_dirs: list[str] | str,
    split: str,
    enriched: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load HDF5 files for a given split.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    filename = f"{split}_data_enriched.h5" if enriched else f"{split}_data.h5"

    all_points, all_labels = [], []
    for ddir in data_dirs:
        pattern  = os.path.join(ddir, filename)
        h5_files = sorted(glob.glob(pattern))

        if not h5_files:
            logging.warning(f"No HDF5 files found matching '{pattern}'. Skipping...")
            continue

        for fpath in h5_files:
            with h5py.File(fpath, "r") as f:
                # data shape: [B, N, D]
                all_points.append(f["data"][:].astype(np.float32))
                all_labels.append(f["label"][:].astype(np.int64).squeeze(-1))  # [B]

    if not all_points:
        raise FileNotFoundError(
            f"No HDF5 files found for split='{split}' "
            f"(enriched={enriched}) in: {data_dirs}"
        )

    points = np.concatenate(all_points, axis=0)  # [N_total, N, D]
    labels = np.concatenate(all_labels, axis=0)  # [N_total]
    return points, labels


@DATASETS.register_module()
class BreakingBadCls(Dataset):
    """
    Binary classification dataset for the Breaking Bad collection.

    Classes:
        0 → complete
        1 → broken
    """

    classes = ["complete", "broken"]

    def __init__(
        self,
        data_dir: str | list[str] = "data/bb_classification",
        num_points: int = 2048,
        split: str = "train",
        transform=None,
        enriched: bool = False,
        **kwargs,
    ):
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = [d.strip() for d in data_dir.split(',')]

        self.data_dirs = []
        for d in data_dir:
            if not os.path.isabs(d):
                d = os.path.join(os.getcwd(), d)
            self.data_dirs.append(d)

        self.num_points = num_points
        self.split      = split.lower()
        self.transform  = transform
        self.enriched   = enriched

        self.points, self.labels = _load_h5_files(
            self.data_dirs, self.split, enriched=self.enriched
        )

        self.feat_dim = self.points.shape[-1]
        
        # For weighted sampling
        self.class_counts = np.bincount(self.labels, minlength=len(self.classes))
        
        logging.info(
            f"[BreakingBadCls] Loaded {self.split}: "
            f"{len(self.points)} samples | "
            f"class_counts={self.class_counts.tolist()} | "
            f"feat_dim={self.feat_dim} ({'enriched' if enriched else 'baseline'})"
        )

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> dict:
        pts   = self.points[idx]           # [N, D]
        label = np.int64(self.labels[idx])

        # Subsampling
        n = pts.shape[0]
        if n > self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
            pts = pts[choice]
        elif n < self.num_points:
            choice = np.random.choice(n, self.num_points, replace=True)
            pts = pts[choice]

        if self.split == "train":
            np.random.shuffle(pts)

        # Extract features (cols 3+)
        geom_feats = pts[:, 3:].copy() if self.feat_dim > 3 else None

        data = {
            "pos": pts[:, :3].copy(),   # [num_points, 3]
            "y": label,
        }

        if self.transform is not None:
            data = self.transform(data)

        if geom_feats is not None:
            geom_tensor = torch.FloatTensor(geom_feats)
            data["x"] = torch.cat([data["pos"], geom_tensor], dim=-1)
        else:
            data["x"] = data["pos"]

        # Ensure everything is float32 to avoid "Input type (double) and bias type (float)" errors
        data["pos"] = data["pos"].float()
        data["x"] = data["x"].float()

        return data

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def get_sample_weights(self):
        """Returns weights for WeightedRandomSampler."""
        weights = 1.0 / (self.class_counts + 1e-6)
        sample_weights = weights[self.labels]
        return sample_weights
