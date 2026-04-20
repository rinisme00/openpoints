"""
FantasticBreaksCls — PointNeXt dataset loader for binary classification of
broken vs. complete 3D objects from the Fantastic Breaks v1 dataset.

Supports two data modes:

  Baseline (enriched=False, default):
    Loads `{split}_data.h5`  —  data shape [B, N, 3]  (XYZ only)
    data['x'] = data['pos'] after transforms  —  shape [N, 3]

  Enriched (enriched=True):
    Loads `{split}_data_enriched.h5`  —  data shape [B, N, 10]
        cols 0-2 : x, y, z  (XYZ, unit-sphere normalized)
        col  3   : k1        principal curvature maximum
        col  4   : k2        principal curvature minimum
        col  5   : H         mean curvature
        col  6   : K         Gaussian curvature
        col  7   : sa_v_ratio surface area / volume (global, broadcast)
        col  8   : dist_centroid distance from centroid
        col  9   : local_density mean dist to 16-NN
    data['x'] = cat(data['pos'], geom_features) after transforms  —  shape [N, 10]

Usage — YAML config snippet:
    dataset:
      common:
        NAME: FantasticBreaksCls
        data_dir: data/fantastic-breaks-classification
        num_points: 8192
        enriched: True     # set True to use geometric-feature-enriched HDF5
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


def _load_h5_files(
    data_dirs: list[str] | str,
    split: str,
    enriched: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load HDF5 files for a given split.

    Args:
        data_dirs : One or more directories to search.
        split     : 'train' or 'test'.
        enriched  : If True, load `{split}_data_enriched.h5` (shape [B, N, 10])
                    instead of the baseline `{split}_data.h5` (shape [B, N, 3]).
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
                # data shape: [B, N, 3] (baseline) or [B, N, 10] (enriched)
                all_points.append(f["data"][:].astype(np.float32))
                all_labels.append(f["label"][:].astype(np.int64).squeeze(-1))  # [B]

    if not all_points:
        raise FileNotFoundError(
            f"No HDF5 files found for split='{split}' "
            f"(enriched={enriched}) in: {data_dirs}"
        )

    points = np.concatenate(all_points, axis=0)  # [N_total, N, 3 or 10]
    labels = np.concatenate(all_labels, axis=0)  # [N_total]
    return points, labels


@DATASETS.register_module()
class FantasticBreaksCls(Dataset):
    """
    Binary classification dataset for the Fantastic Breaks v1 collection.

    Classes:
        0 → complete  (model_c.ply)
        1 → broken    (model_b_0.ply)

    Args:
        data_dir   : Path to directory containing HDF5 files.
        num_points : Number of points to use per sample (subset of stored N).
        split      : 'train' or 'test'.
        transform  : Optional OpenPoints data-transform pipeline.
        enriched   : If True, load `{split}_data_enriched.h5` (10 feature dims)
                     instead of the baseline `{split}_data.h5` (3 dims / XYZ only).
    """

    classes = ["complete", "broken"]

    def __init__(
        self,
        data_dir: str | list[str] = "data/classification",
        num_points: int = 2048,
        split: str = "train",
        transform=None,
        enriched: bool = False,
        **kwargs,          # absorb extra keys from YAML (e.g. 'NAME')
    ):
        super().__init__()
        # Normalise data_dir to a list of absolute paths
        if isinstance(data_dir, str):
            data_dir = [d.strip() for d in data_dir.split(',')]

        self.data_dirs = []
        for d in data_dir:
            if not os.path.isabs(d):
                d = os.path.join(os.getcwd(), d)
            self.data_dirs.append(d)

        self.num_points = num_points
        self.split      = "train" if split.lower() == "train" else "test"
        self.transform  = transform
        self.enriched   = enriched   # True → load *_data_enriched.h5 ([B,N,10])

        self.points, self.labels = _load_h5_files(
            self.data_dirs, self.split, enriched=self.enriched
        )

        feat_dim = self.points.shape[-1]  # 3 (baseline) or 10 (enriched)
        logging.info(
            f"[FantasticBreaksCls] Loaded {self.split}: "
            f"{len(self.points)} samples | "
            f"complete={int((self.labels == 0).sum())}, "
            f"broken={int((self.labels == 1).sum())} | "
            f"feat_dim={feat_dim} ({'enriched' if enriched else 'baseline'})"
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict compatible with the PointNeXt training loop:

            Baseline  (enriched=False):
                'pos' : FloatTensor [num_points, 3]
                'x'   : FloatTensor [num_points, 3]   (= pos, XYZ as feature)
                'y'   : int label

            Enriched  (enriched=True):
                'pos' : FloatTensor [num_points, 3]    (XYZ, augmented by transforms)
                'x'   : FloatTensor [num_points, 11]   (augmented XYZ + 8 geom features)
                'y'   : int label
        """
        pts   = self.points[idx]           # [N, 3] or [N, 10]
        label = np.int64(self.labels[idx])

        # ── Point subsampling ─────────────────────────────────────────────
        # Subsample/upsample rows uniformly across ALL columns so xyz and
        # geometric features stay perfectly aligned after selection.
        n = pts.shape[0]
        if n > self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
            pts = pts[choice]
        elif n < self.num_points:
            choice = np.random.choice(n, self.num_points, replace=True)
            pts = pts[choice]

        # Shuffle rows during training (consistent across all columns)
        if self.split == "train":
            np.random.shuffle(pts)

        # ── Save geometric features before transforms ──────────────────────
        # Transforms (rotate, scale, center) modify data['pos'] only.
        # Curvature and scalar features are rotation-invariant and pre-normalised,
        # so we detach them here and re-attach after augmentation.
        geom_feats = pts[:, 3:].copy() if self.enriched else None  # [N, 7] or None

        # ── Build data dict and apply transforms ───────────────────────────
        data = {
            "pos": pts[:, :3].copy(),   # [num_points, 3]  xyz only
            "y": label,
        }

        if self.transform is not None:
            data = self.transform(data)

        # ── Build data['x'] ────────────────────────────────────────────────
        if geom_feats is not None:
            # Enriched: x = [pos (augmented, 3) | geom_feats (pre-stored, 7)] → [N, 10]
            geom_tensor = torch.FloatTensor(geom_feats)
            data["x"]   = torch.cat([data["pos"], geom_tensor], dim=-1)
        elif "heights" in data:
            data["x"] = torch.cat((data["pos"], data["heights"]), dim=1)
        else:
            # Baseline: x = pos (XYZ after augmentation)
            data["x"] = data["pos"]

        return data

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self.classes)
