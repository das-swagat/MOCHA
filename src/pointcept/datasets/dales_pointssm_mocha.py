import os
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class DALESPointSSMMOCHADataset(DefaultDataset):
    """Adapter for the released PointSSM DALES representation.

    Expected assets per sample:
      coord.npy      float coordinates
      intensity.npy  scalar LiDAR strength/intensity
      segment.npy    labels already encoded 0..7
      instance.npy   optional

    No semantic-label remapping is applied.
    """

    VALID_ASSETS = ("coord", "intensity", "segment", "instance")

    class_names = [
        "ground",
        "vegetation",
        "cars",
        "trucks",
        "powerlines",
        "fences",
        "poles",
        "buildings",
    ]

    def get_data_name(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        return os.path.basename(data_path)

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)

        data_dict = {}
        for asset in os.listdir(data_path):
            if not asset.endswith(".npy"):
                continue
            key = asset[:-4]
            if key not in self.VALID_ASSETS:
                continue
            data_dict[key] = np.load(os.path.join(data_path, asset))

        if "coord" not in data_dict:
            raise FileNotFoundError(f"Missing coord.npy in {data_path}")
        if "intensity" not in data_dict:
            raise FileNotFoundError(f"Missing intensity.npy in {data_path}")

        coord = np.asarray(data_dict["coord"], dtype=np.float32)
        strength = np.asarray(data_dict["intensity"], dtype=np.float32)
        if strength.ndim == 1:
            strength = strength[:, None]

        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"Unexpected coord shape {coord.shape} in {data_path}")
        if strength.shape[0] != coord.shape[0]:
            raise ValueError(
                f"coord/intensity length mismatch in {data_path}: "
                f"{coord.shape[0]} vs {strength.shape[0]}"
            )

        data_dict["coord"] = coord
        data_dict["strength"] = strength
        del data_dict["intensity"]

        if "segment" in data_dict:
            segment = np.asarray(data_dict.pop("segment")).reshape(-1).astype(np.int32)
            if segment.shape[0] != coord.shape[0]:
                raise ValueError(
                    f"coord/segment length mismatch in {data_path}: "
                    f"{coord.shape[0]} vs {segment.shape[0]}"
                )
            bad = (segment < 0) | (segment > 7)
            if bad.any():
                vals = np.unique(segment[bad]).tolist()
                raise ValueError(f"Unexpected DALES labels {vals} in {data_path}")
            data_dict["segment"] = segment
        else:
            data_dict["segment"] = np.full(
                coord.shape[0], self.ignore_index, dtype=np.int32
            )

        if "instance" in data_dict:
            data_dict["instance"] = (
                np.asarray(data_dict["instance"]).reshape(-1).astype(np.int32)
            )
        else:
            data_dict["instance"] = np.full(coord.shape[0], -1, dtype=np.int32)

        data_dict["name"] = name
        data_dict["split"] = self.get_split_name(idx)
        return data_dict
