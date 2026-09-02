"""
MultiMOCHA unlabeled manifest dataset.

This dataset reads a JSONL manifest where each line points to one point cloud
from one LiDAR domain. It does not use semantic labels. It returns coordinates,
a one-channel strength/intensity feature, and dummy ignore labels so that the
standard Pointcept transform and batching path can be reused safely.

Supported manifest records:

{"domain": "nuscenes", "format": "bin5", "path": "...pcd.bin"}
{"domain": "semantic_kitti", "format": "bin4", "path": "...velodyne/000000.bin"}
{"domain": "semantic_usl", "format": "bin4", "path": "...velodyne/000000.bin"}
{"domain": "rellis3d", "format": "pointcept_npy", "root": ".../train/00000_000307"}
{"domain": "toronto3d", "format": "pointcept_npy", "root": ".../train/L001_x0001_y0010"}
"""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class MultiMOCHAUnlabeledDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split="train",
        transform=None,
        test_mode=False,
        loop=1,
        ignore_index=-1,
        max_points=None,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.test_mode = test_mode
        self.loop = int(loop)
        self.ignore_index = int(ignore_index)
        self.max_points = max_points

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.items = []
        with open(self.manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

        if len(self.items) == 0:
            raise RuntimeError(f"Empty manifest: {self.manifest_path}")

        self.transform = Compose(transform) if transform is not None else None

    def __len__(self):
        return len(self.items) * self.loop

    def _read_bin4(self, path):
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size % 4 != 0:
            raise RuntimeError(f"bin4 file is not divisible by 4: {path}")
        arr = arr.reshape(-1, 4)
        coord = arr[:, :3].astype(np.float32)
        strength = arr[:, 3:4].astype(np.float32)
        return coord, strength

    def _read_bin5(self, path):
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size % 5 != 0:
            raise RuntimeError(f"bin5 file is not divisible by 5: {path}")
        arr = arr.reshape(-1, 5)
        coord = arr[:, :3].astype(np.float32)
        strength = arr[:, 3:4].astype(np.float32)
        return coord, strength

    def _read_pointcept_npy(self, root):
        root = Path(root)
        coord_path = root / "coord.npy"
        strength_path = root / "strength.npy"

        if not coord_path.exists():
            raise FileNotFoundError(f"Missing coord.npy: {coord_path}")
        if not strength_path.exists():
            raise FileNotFoundError(f"Missing strength.npy: {strength_path}")

        coord = np.load(coord_path).astype(np.float32)
        strength = np.load(strength_path).astype(np.float32)

        if strength.ndim == 1:
            strength = strength[:, None]

        if coord.shape[0] != strength.shape[0]:
            raise RuntimeError(
                f"coord/strength length mismatch in {root}: "
                f"{coord.shape[0]} vs {strength.shape[0]}"
            )

        return coord, strength[:, :1]

    def _load_item(self, item):
        fmt = item["format"]

        if fmt == "bin4":
            coord, strength = self._read_bin4(item["path"])
            name = Path(item["path"]).stem
        elif fmt == "bin5":
            coord, strength = self._read_bin5(item["path"])
            name = Path(item["path"]).stem
        elif fmt == "pointcept_npy":
            coord, strength = self._read_pointcept_npy(item["root"])
            name = Path(item["root"]).name
        else:
            raise NotImplementedError(f"Unsupported MultiMOCHA format: {fmt}")

        if self.max_points is not None and coord.shape[0] > self.max_points:
            idx = np.random.choice(coord.shape[0], int(self.max_points), replace=False)
            coord = coord[idx]
            strength = strength[idx]

        segment = np.full((coord.shape[0],), self.ignore_index, dtype=np.int64)

        # Build compatibility features for existing Utonia/MOCHA configs.
        # Existing configs expect feat_keys=("coord", "color", "normal"), i.e. 9 input channels.
        # For LiDAR-only multi-domain SSL, color is represented by repeated intensity/strength,
        # and normal is a simple radial unit vector from the sensor origin.
        color = np.repeat(strength[:, :1], 3, axis=1).astype(np.float32)

        normal = coord.astype(np.float32).copy()
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.maximum(norm, 1e-6)
        normal[~np.isfinite(normal)] = 0

        data_dict = dict(
            coord=coord,
            strength=strength,
            color=color,
            normal=normal.astype(np.float32),
            segment=segment,
            domain=item.get("domain", "unknown"),
            name=f"{item.get('domain', 'unknown')}_{name}",
        )
        return data_dict

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        data_dict = self._load_item(item)

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict
