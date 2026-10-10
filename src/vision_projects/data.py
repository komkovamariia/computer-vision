from pathlib import Path
import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from .training import DataSplit


def signs_data(directory: Path) -> DataSplit:
    paths = [directory / "train_signs.h5", directory / "test_signs.h5"]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("Download SIGNS first: python scripts/download_data.py signs")
    datasets = []
    for path, prefix in zip(paths, ("train", "test")):
        with h5py.File(path) as handle:
            images = (
                torch.tensor(np.asarray(handle[f"{prefix}_set_x"]), dtype=torch.float32).permute(0, 3, 1, 2)
                / 255
            )
            labels = torch.tensor(np.asarray(handle[f"{prefix}_set_y"]).reshape(-1), dtype=torch.long)
        datasets.append((images, labels))
    images, labels = datasets[0]
    train, validation = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=1031
    )
    return DataSplit(
        TensorDataset(images[train], labels[train]),
        TensorDataset(images[validation], labels[validation]),
        TensorDataset(*datasets[1]),
    )
