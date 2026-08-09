"""PyTorch Dataset over pre-extracted CT image/mask PNG slice pairs."""

from collections.abc import Callable
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

REQUIRED_COLUMNS: tuple[str, ...] = ("image_path", "mask_path")


class SegichDataset(Dataset):
    def __init__(self, csv_path: Path, transform: Callable | None = None) -> None:
        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Manifest {csv_path} is missing required columns: {missing}")

        self.csv_path = csv_path
        self.transform = transform
        self.pairs: list[tuple[str, str]] = list(
            zip(df["image_path"], df["mask_path"], strict=False)
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]

        img_arr = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        if img_arr is None:
            raise FileNotFoundError(f"Failed to read image at index {idx}: {image_path}")

        msk_arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if msk_arr is None:
            raise FileNotFoundError(f"Failed to read mask at index {idx}: {mask_path}")

        image = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(msk_arr).long() / 255.0

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


class SyntheticSegmentationDataset(Dataset):
    """Randomly generated image/mask pairs for fast, disk-free training-loop verification."""

    def __init__(self, num_samples: int = 16, image_size: tuple[int, int] = (64, 64)) -> None:
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = self.image_size
        image = torch.rand(3, h, w)
        mask = (torch.rand(h, w) > 0.5).float()
        return image, mask
