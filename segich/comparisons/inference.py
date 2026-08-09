"""Run a trained smp checkpoint on one image, optionally score Dice, and save a visualization."""

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from matplotlib.figure import Figure

from segich.transform import TestTransform
from segich.utils import configure_logging

DEFAULT_THRESHOLD: float = 0.5
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-image inference: checkpoint dir, image/mask paths, output."""
    parser = argparse.ArgumentParser(
        description="Run a trained segmentation checkpoint on one image and save a prediction."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a save_pretrained checkpoint directory.",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional path to a ground-truth mask, for Dice scoring.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the output visualization PNG."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold for binarizing the prediction (default: 0.5).",
    )
    return parser.parse_args()


def load_model(checkpoint_dir: str, device: torch.device) -> torch.nn.Module:
    """Load an smp save_pretrained checkpoint onto device, skipping the ImageNet encoder init."""
    model = smp.from_pretrained(checkpoint_dir, encoder_weights=None, map_location=device.type)
    return model.to(device).eval()


def load_image_and_mask(
    image_path: str, mask_path: str | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read an RGB image and optional grayscale mask; raise FileNotFoundError if missing."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    mask = None
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    return image, mask


def preprocess(image: np.ndarray, mask: np.ndarray | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply TestTransform to image and mask, using a zero placeholder mask when none is given."""
    mask_input = mask if mask is not None else np.zeros(image.shape[:2], dtype=np.uint8)
    return TestTransform()(image, mask_input)


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Invert TestTransform's ImageNet normalization to an HWC float array in [0, 1] for display."""
    image = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def overlay_prediction(
    image_rgb: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Alpha-blend a semi-transparent red prediction mask onto an RGB [0, 1] image."""
    overlay = image_rgb.copy()
    overlay[pred_mask] = (1 - alpha) * overlay[pred_mask] + alpha * np.array([1.0, 0.0, 0.0])
    return overlay


def build_figure(
    image_rgb: np.ndarray,
    ground_truth: np.ndarray | None,
    pred_overlay: np.ndarray,
    dice: float | None,
) -> Figure:
    """Build a matplotlib panel of input | [ground truth] | prediction, Dice in the title."""
    panels = [("Input", image_rgb, None)]
    if ground_truth is not None:
        panels.append(("Ground Truth", ground_truth, "gray"))
    panels.append(("Prediction", pred_overlay, None))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for ax, (title, data, cmap) in zip(axes, panels, strict=True):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    title = f"Dice: {dice:.4f}" if dice is not None else "Dice: N/A (no ground truth provided)"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    """Run inference on one image and save an input/ground-truth/prediction visualization panel."""
    args = parse_args()
    configure_logging()
    plt.switch_backend("Agg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    image, mask = load_image_and_mask(args.image, args.mask)
    image_t, mask_t = preprocess(image, mask)

    with torch.no_grad():
        probs = torch.sigmoid(model(image_t.unsqueeze(0).to(device)))
    pred_mask = (probs > args.threshold).squeeze().cpu().numpy()

    dice = None
    if mask is not None:
        target = (mask_t > 0).long().unsqueeze(0).unsqueeze(0).to(device)
        tp, fp, fn, tn = smp.metrics.get_stats(
            probs, target, mode="binary", threshold=args.threshold
        )
        dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()

    image_rgb = denormalize(image_t)
    ground_truth = (mask_t > 0).float().numpy() if mask is not None else None
    pred_overlay = overlay_prediction(image_rgb, pred_mask)

    fig = build_figure(image_rgb, ground_truth, pred_overlay, dice)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    logging.info(
        "Dice: %s - saved visualization to %s",
        f"{dice:.4f}" if dice is not None else "N/A (no ground truth provided)",
        output_path,
    )


if __name__ == "__main__":
    main()
