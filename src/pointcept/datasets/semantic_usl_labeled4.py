"""
SemanticUSL labeled four-sequence dataset.
Separate loader. Does not modify SemanticKITTI behavior.
"""

import os

from .builder import DATASETS
from .semantic_kitti import SemanticKITTIDataset, SemanticKITTIColorNormalDataset

def _semanticusl_labeled4_data_list(data_root, split):
    split2seq = dict(
        train=[3, 12],
        val=[21],
        test=[32],
        trainval=[3, 12, 21],
    )

    if isinstance(split, str):
        seq_list = split2seq[split]
    elif isinstance(split, list):
        seq_list = []
        for item in split:
            if isinstance(item, str) and item in split2seq:
                seq_list += split2seq[item]
            else:
                seq_list.append(int(item))
    else:
        raise NotImplementedError

    data_list = []
    for seq in seq_list:
        seq = str(seq).zfill(2)
        seq_folder = os.path.join(data_root, "dataset", "sequences", seq)
        velodyne_folder = os.path.join(seq_folder, "velodyne")
        seq_files = sorted(os.listdir(velodyne_folder))
        data_list += [
            os.path.join(velodyne_folder, file)
            for file in seq_files
            if file.endswith(".bin")
        ]

    return data_list


@DATASETS.register_module()
class SemanticUSLLabeled4Dataset(SemanticKITTIDataset):
    def get_data_list(self):
        return _semanticusl_labeled4_data_list(self.data_root, self.split)


@DATASETS.register_module()
class SemanticUSLLabeled4ColorNormalDataset(SemanticKITTIColorNormalDataset):
    @staticmethod
    def estimate_normals(points, center=None):
        import numpy as np
        if center is None:
            center = np.array([0, 0, 0], dtype=points.dtype)
        normals = points - center[None, :]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-6)
        normals[~np.isfinite(normals)] = 0
        return normals

    def get_data_list(self):
        return _semanticusl_labeled4_data_list(self.data_root, self.split)
