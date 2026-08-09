"""Extract 2D PNG slices from 3D NIfTI volumes paired with validated NRRD masks."""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import cv2
import nibabel as nib
import nrrd
import numpy as np

TARGET_SIZE: tuple[int, int] = (512, 512)
WINDOW_LEVEL: int = 40
WINDOW_WIDTH: int = 400
MASK_MIN_VALID: int = 0
MASK_MAX_VALID: int = 8


def parse_args() -> argparse.Namespace:
    """Parse the three required CLI directory arguments."""
    parser = argparse.ArgumentParser(
        description="Extract 2D image/mask PNG slices from NIfTI volumes and NRRD masks."
    )
    parser.add_argument("--nifti_dir", type=Path, required=True, help="Directory of NIfTI files.")
    parser.add_argument("--nrrd_dir", type=Path, required=True, help="Directory of NRRD files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for outputs.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output_dir if it exists."
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure stderr logging so progress/warnings don't pollute stdout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def is_nifti_file(path: Path) -> bool:
    """Return True if path is a file with a .nii or .nii.gz extension."""
    return path.is_file() and (path.name.endswith(".nii.gz") or path.name.endswith(".nii"))


def list_nifti_files(nifti_dir: Path) -> list[Path]:
    """Return a sorted, non-recursive list of NIfTI files in nifti_dir."""
    return sorted(p for p in nifti_dir.iterdir() if is_nifti_file(p))


def extract_id(nifti_filename: str) -> str | None:
    """Strip the NIfTI extension, split on '_', and return the second element as the ID.

    Returns None if the filename yields fewer than two underscore-separated parts.
    """
    if nifti_filename.endswith(".nii.gz"):
        stem = nifti_filename[: -len(".nii.gz")]
    elif nifti_filename.endswith(".nii"):
        stem = nifti_filename[: -len(".nii")]
    else:
        stem = nifti_filename

    parts = stem.split("_")
    if len(parts) < 2:
        return None
    return parts[1]


def is_valid_mask_array(data: np.ndarray) -> bool:
    """Check that data holds only integer values ranging exactly from 0 to 8."""
    return bool(
        np.all(np.mod(data, 1) == 0)
        and data.min() == MASK_MIN_VALID
        and data.max() <= MASK_MAX_VALID
    )


def find_valid_mask(nrrd_dir: Path, id_: str) -> tuple[Path, np.ndarray] | None:
    """Find the first NRRD file matching id_ whose data passes is_valid_mask_array.

    Candidate files are matched by substring on id_ and tried in sorted order.
    Unreadable candidates are logged and skipped in favor of the next one.
    """
    candidates = sorted(nrrd_dir.glob(f"*{id_}*.nrrd"))
    for candidate in candidates:
        try:
            data, _header = nrrd.read(str(candidate))
        except Exception:
            logging.warning(
                "Failed to read NRRD candidate %s; trying next.", candidate, exc_info=True
            )
            continue
        if is_valid_mask_array(data):
            return candidate, data
    return None


def window_normalize(slice_: np.ndarray, wl: int, ww: int) -> np.ndarray:
    """Apply CT windowing (level wl, width ww) and rescale to uint8 [0, 255]."""
    low = wl - ww / 2
    high = wl + ww / 2
    clipped = np.clip(slice_, low, high)
    scaled = (clipped - low) / (high - low) * 255.0
    return scaled.astype(np.uint8)


def binarize_mask(slice_: np.ndarray) -> np.ndarray:
    """Binarize a mask slice: values > 0 become 255 (foreground), else 0."""
    return np.where(slice_ > 0, 255, 0).astype(np.uint8)


def process_volume(
    nifti_path: Path,
    mask_path: Path,
    mask_data: np.ndarray,
    id_: str,
    image_dir: Path,
    mask_dir: Path,
    writer: csv.DictWriter,
) -> None:
    """Slice a NIfTI/NRRD pair along z, resize to 512x512, and write PNGs + CSV rows.

    Skips (logs a warning and returns) if the NIfTI and NRRD volume shapes differ.
    """
    img = nib.load(nifti_path)
    img_data = img.get_fdata(dtype=np.float32)

    if img_data.shape != mask_data.shape:
        logging.warning(
            "Shape mismatch for id=%s: nifti=%s (%s) vs nrrd=%s (%s); skipping.",
            id_,
            nifti_path,
            img_data.shape,
            mask_path,
            mask_data.shape,
        )
        return

    for z in range(img_data.shape[2]):
        image_slice = img_data[:, :, z]
        mask_slice = mask_data[:, :, z]

        foreground_pixels = int(np.count_nonzero(mask_slice > 0))

        mask_bin = binarize_mask(mask_slice)
        mask_resized = cv2.resize(mask_bin, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        img_u8 = window_normalize(image_slice, WINDOW_LEVEL, WINDOW_WIDTH)
        img_resized = cv2.resize(img_u8, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        image_out = image_dir / f"{id_}_{z:04d}.png"
        mask_out = mask_dir / f"{id_}_{z:04d}.png"
        cv2.imwrite(str(image_out), img_resized)
        cv2.imwrite(str(mask_out), mask_resized)

        writer.writerow(
            {
                "image_path": str(image_out),
                "mask_path": str(mask_out),
                "foreground_pixels": foreground_pixels,
            }
        )


def main() -> None:
    """Run the full extraction pipeline over --nifti_dir, pairing masks from --nrrd_dir."""
    args = parse_args()
    configure_logging()

    if not args.nifti_dir.is_dir():
        logging.error("--nifti_dir does not exist or is not a directory: %s", args.nifti_dir)
        sys.exit(1)
    if not args.nrrd_dir.is_dir():
        logging.error("--nrrd_dir does not exist or is not a directory: %s", args.nrrd_dir)
        sys.exit(1)

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    image_dir = args.output_dir / "image"
    mask_dir = args.output_dir / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = list_nifti_files(args.nifti_dir)
    if not nifti_files:
        logging.warning("No NIfTI files found in %s", args.nifti_dir)

    csv_path = args.output_dir / "processing_log.csv"
    fieldnames = ["image_path", "mask_path", "foreground_pixels"]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for nifti_path in nifti_files:
            try:
                id_ = extract_id(nifti_path.name)
                if id_ is None:
                    logging.warning(
                        "Skipping %s: cannot extract ID from filename.", nifti_path.name
                    )
                    continue

                candidate = find_valid_mask(args.nrrd_dir, id_)
                if candidate is None:
                    logging.warning(
                        "Skipping %s: no valid NRRD mask found for id=%s.",
                        nifti_path.name,
                        id_,
                    )
                    continue

                mask_path, mask_data = candidate
                process_volume(nifti_path, mask_path, mask_data, id_, image_dir, mask_dir, writer)

            except Exception:
                logging.exception("Unexpected error processing %s; skipping.", nifti_path.name)
                continue


if __name__ == "__main__":
    main()
