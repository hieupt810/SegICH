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

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at index {idx}: {image_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask at index {idx}: {mask_path}")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
