"""
FantasticBreaksCls — PointNeXt dataset loader for binary classification of
broken vs. complete 3D objects from the Fantastic Breaks v1 dataset.

Data format (HDF5, produced by scripts/prepare_classification_data.py):
    - data:  float32  [B, N, 3]   — normalized point cloud (XYZ)
    - label: int64    [B, 1]      — class index: 0 = complete, 1 = broken

PointNeXt convention (mirrors ModelNet40Ply2048):
    __getitem__ returns a dict with keys:
        'pos' : FloatTensor [num_points, 3] — 3D coordinates
        'x'   : FloatTensor [num_points, C] — input features (here = XYZ copy)
        'y'   : int                         — class label (scalar)

Usage — YAML config snippet:
    dataset:
      common:
        NAME: FantasticBreaksCls
        data_dir: data/classification
        num_points: 2048
      train:
        split: train
      val:
        split: test
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


def _load_h5_files(data_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all HDF5 files matching `<split>_data.h5` inside `data_dir`.

    Returns:
        points : float32 array of shape [N_total, max_points, 3]
        labels : int64 array of shape [N_total]
    """
    pattern = os.path.join(data_dir, f"{split}_data.h5")
    h5_files = sorted(glob.glob(pattern))

    if not h5_files:
        raise FileNotFoundError(
            f"No HDF5 files found matching '{pattern}'.\n"
            f"Run scripts/prepare_classification_data.py first."
        )

    all_points, all_labels = [], []
    for fpath in h5_files:
        with h5py.File(fpath, "r") as f:
            all_points.append(f["data"][:].astype(np.float32))   # [B, N, 3]
            all_labels.append(f["label"][:].astype(np.int64).squeeze(-1))  # [B]

    points = np.concatenate(all_points, axis=0)   # [N_total, N, 3]
    labels = np.concatenate(all_labels, axis=0)   # [N_total]
    return points, labels


@DATASETS.register_module()
class FantasticBreaksCls(Dataset):
    """
    Binary classification dataset for the Fantastic Breaks v1 collection.

    Classes:
        0 → complete  (model_c.ply)
        1 → broken    (model_b_0.ply)

    Args:
        data_dir   : Path to directory containing train_data.h5 / test_data.h5.
        num_points : Number of points to use per sample (subset of the stored N).
        split      : 'train' or 'test'.
        transform  : Optional OpenPoints data-transform pipeline.
    """

    classes = ["complete", "broken"]

    def __init__(
        self,
        data_dir: str = "data/classification",
        num_points: int = 2048,
        split: str = "train",
        transform=None,
        **kwargs,          # absorb extra keys from YAML (e.g. 'NAME')
    ):
        super().__init__()
        # Resolve relative paths against current working directory
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(os.getcwd(), data_dir)

        self.data_dir = data_dir
        self.num_points = num_points
        self.split = "train" if split.lower() == "train" else "test"
        self.transform = transform

        self.points, self.labels = _load_h5_files(self.data_dir, self.split)

        logging.info(
            f"[FantasticBreaksCls] Loaded {self.split}: "
            f"{len(self.points)} samples | "
            f"complete={int((self.labels == 0).sum())}, "
            f"broken={int((self.labels == 1).sum())}"
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict compatible with the PointNeXt training loop:
            'pos' : FloatTensor [num_points, 3]
            'x'   : FloatTensor [num_points, 3]  (same as pos; XYZ as input feature)
            'y'   : int scalar label (0 = complete, 1 = broken)
        """
        pts = self.points[idx]                   # [N, 3]
        label = np.int64(self.labels[idx])        # numpy scalar (has .dtype)

        # ── Point subsampling ────────────────────────────────────────────
        # The stored H5 may have more points than requested (e.g. 8192 stored,
        # 2048 requested).  Randomly sample without replacement each epoch.
        n = pts.shape[0]
        if n > self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
            pts = pts[choice]
        elif n < self.num_points:
            # Upsample with replacement if fewer points are stored
            choice = np.random.choice(n, self.num_points, replace=True)
            pts = pts[choice]
        # else: n == num_points, use as-is

        # Shuffle within a sample during training (helps with FPS layers)
        if self.split == "train":
            np.random.shuffle(pts)

        # ── Assemble return dict ─────────────────────────────────────────
        data = {
            "pos": pts,          # [num_points, 3]  (numpy)
            "y": label,
        }

        # Apply optional transform pipeline (e.g. PointsToTensor, scale, rotate)
        # After transforms, data['pos'] is a torch.Tensor.
        if self.transform is not None:
            data = self.transform(data)

        # 'x' is the per-point feature tensor fed to the backbone.
        # After transforms, pos is already a Tensor — mirror ModelNet40 pattern.
        if "heights" in data:
            data["x"] = torch.cat((data["pos"], data["heights"]), dim=1)
        else:
            data["x"] = data["pos"]

        return data

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self.classes)
