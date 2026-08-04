"""PyTorch Dataset over pre-extracted CT image/mask PNG slice pairs."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REQUIRED_COLUMNS: tuple[str, ...] = ("image_path", "mask_path", "foreground_pixels")

JointTransform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class SegichDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Loads 512x512 CT image/mask PNG pairs listed in an extract_data.py processing_log.csv.

    Trusts extract_data.py's guarantees: images and masks are 512x512 single-channel uint8
    PNGs, and mask pixel values are strictly {0, 255}. Shape/dtype are not re-validated here.
    """

    def __init__(self, csv_path: Path, transform: JointTransform | None = None) -> None:
        """Load csv_path's manifest and index it for O(1) random access.

        Raises ValueError if csv_path is missing any of REQUIRED_COLUMNS.
        """
        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Manifest {csv_path} is missing required columns: {missing}")

        self.csv_path = csv_path
        self.transform = transform
        self.image_paths: list[str] = df["image_path"].tolist()
        self.mask_paths: list[str] = df["mask_path"].tolist()
        self.foreground_pixels: list[int] = df["foreground_pixels"].tolist()

    def __len__(self) -> int:
        """Return the number of image/mask pairs listed in the manifest."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image, mask) as float32 (1, H, W) tensors, image in [0,1], mask in {0,1}.

        Raises FileNotFoundError if either PNG at idx cannot be read by OpenCV.
        """
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image_arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_arr is None:
            raise FileNotFoundError(f"Failed to read image at index {idx}: {image_path}")

        mask_arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_arr is None:
            raise FileNotFoundError(f"Failed to read mask at index {idx}: {mask_path}")

        image = torch.from_numpy(image_arr.astype(np.float32) / 255.0).unsqueeze(0)
        mask = torch.from_numpy(mask_arr.astype(np.float32) / 255.0).unsqueeze(0)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
