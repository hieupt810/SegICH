import argparse
import logging
import math
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from segich.dataset import SegichDataset, SyntheticSegmentationDataset
from segich.losses import BCEDiceLoss
from segich.transform import TestTransform, TrainTransform
from segich.utils import configure_logging, set_seed

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training/evaluation: data CSVs, model choice, and hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Train a segmentation model on 2D image/mask PNG slices."
    )
    parser.add_argument(
        "--train-csv", type=str, default=None, help="Path to the training CSV file."
    )
    parser.add_argument(
        "--val-csv", type=str, default=None, help="Path to the validation CSV file."
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="Unet",
        help="Segmentation model architecture (default: Unet).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        help="Segmentation model encoder (default: resnet34).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training and validation."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of DataLoader worker subprocesses."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Probability thresholds for validation Dice/PPV/NPV (>=4 values).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/best_model",
        help="Directory to save the best validation-Dice checkpoint.",
    )
    parser.add_argument(
        "--bce-weight", type=float, default=0.5, help="Weight for the BCE loss term."
    )
    parser.add_argument(
        "--dice-weight", type=float, default=0.5, help="Weight for the Dice loss term."
    )
    parser.add_argument(
        "--log-interval", type=int, default=50, help="Log training progress every N steps."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a few training steps on synthetic data to verify the loop end-to-end.",
    )
    parser.add_argument(
        "--smoke-samples", type=int, default=16, help="Number of synthetic samples."
    )
    parser.add_argument(
        "--smoke-size", type=int, default=64, help="Synthetic square image size (HxW)."
    )
    parser.add_argument(
        "--smoke-batch-size", type=int, default=4, help="Batch size for --smoke-test."
    )
    parser.add_argument(
        "--smoke-epochs", type=int, default=3, help="Number of epochs for --smoke-test."
    )

    args = parser.parse_args()
    if len(args.thresholds) < 4:
        parser.error("--thresholds requires at least 4 values.")
    if not args.smoke_test and (args.train_csv is None or args.val_csv is None):
        parser.error("--train-csv and --val-csv are required unless --smoke-test is set.")
    return args


def build_model(arch: str, encoder: str) -> torch.nn.Module:
    """Create an ImageNet-pretrained smp segmentation model with 3 input channels, 1 output class"""
    return smp.create_model(
        arch=arch, encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=1
    )


def build_loss_function(bce_weight: float = 0.5, dice_weight: float = 0.5) -> BCEDiceLoss:
    """Build a combined BCE+Dice loss over raw logits, matching the model's output channel."""
    return BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)


def build_dataloaders(
    train_csv: str, val_csv: str, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    """Build streaming train/val DataLoaders over SegichDataset with Train/TestTransform."""
    train_ds = SegichDataset(train_csv, transform=TrainTransform())
    val_ds = SegichDataset(val_csv, transform=TestTransform())
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def prepare_target(mask: torch.Tensor) -> torch.Tensor:
    """Binarize a raw uint8 {0,255} mask batch (N,H,W) into a float {0,1} target (N,1,H,W)."""
    return (mask > 0).unsqueeze(1).float()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
) -> tuple[float, float, float]:
    """Run one training epoch under AMP autocast + GradScaler; return mean (loss, bce, dice)."""
    model.train()
    running_loss = running_bce = running_dice = 0.0
    num_steps = len(loader)
    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = prepare_target(masks).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss, bce_loss, dice_loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        running_bce += bce_loss.item() * images.size(0)
        running_dice += dice_loss.item() * images.size(0)

        if step % log_interval == 0 or step == num_steps:
            logging.info(
                "Epoch %d, step %d/%d - loss=%.4f bce=%.4f dice=%.4f",
                epoch,
                step,
                num_steps,
                loss.item(),
                bce_loss.item(),
                dice_loss.item(),
            )

    n = len(loader.dataset)
    return running_loss / n, running_bce / n, running_dice / n


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
) -> dict[float, dict[str, float]]:
    """Run one inference pass and return per-threshold micro Dice/PPV/NPV metrics."""
    model.eval()
    accum = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in thresholds}

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        targets = prepare_target(masks).to(device, non_blocking=True).long()

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
        probs = torch.sigmoid(logits.float())

        for t in thresholds:
            tp, fp, fn, tn = smp.metrics.get_stats(probs, targets, mode="binary", threshold=t)
            accum[t]["tp"] += tp.sum()
            accum[t]["fp"] += fp.sum()
            accum[t]["fn"] += fn.sum()
            accum[t]["tn"] += tn.sum()

    results: dict[float, dict[str, float]] = {}
    for t in thresholds:
        tp, fp, fn, tn = accum[t]["tp"], accum[t]["fp"], accum[t]["fn"], accum[t]["tn"]
        results[t] = {
            "dice": smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item(),
            "ppv": smp.metrics.positive_predictive_value(tp, fp, fn, tn, reduction="micro").item(),
            "npv": smp.metrics.negative_predictive_value(tp, fp, fn, tn, reduction="micro").item(),
        }
    return results


def format_metrics_table(metrics: dict[float, dict[str, float]]) -> str:
    """Render per-threshold Dice/PPV/NPV metrics as a fixed-width plain-text table."""
    header = f"{'Threshold':>10} | {'Dice':>8} | {'PPV':>8} | {'NPV':>8}"
    rows = [header, "-" * len(header)]
    for t in sorted(metrics):
        m = metrics[t]
        rows.append(f"{t:>10.2f} | {m['dice']:>8.4f} | {m['ppv']:>8.4f} | {m['npv']:>8.4f}")
    return "\n".join(rows)


def run_smoke_test(args: argparse.Namespace) -> None:
    """Run a few training steps on synthetic data and verify the loop completes with finite loss."""
    configure_logging()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.create_model(
        arch=args.arch, encoder_name=args.encoder, encoder_weights=None, in_channels=3, classes=1
    ).to(device)
    criterion = build_loss_function(args.bce_weight, args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    loader = DataLoader(
        SyntheticSegmentationDataset(
            num_samples=args.smoke_samples, image_size=(args.smoke_size, args.smoke_size)
        ),
        batch_size=args.smoke_batch_size,
        shuffle=True,
    )

    logging.info(
        "Smoke test: %s/%s, %d samples, %dx%d, %d epochs",
        args.arch,
        args.encoder,
        args.smoke_samples,
        args.smoke_size,
        args.smoke_size,
        args.smoke_epochs,
    )

    losses = []
    for epoch in range(1, args.smoke_epochs + 1):
        train_loss, train_bce, train_dice = train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        logging.info(
            "Smoke epoch %d/%d - loss=%.4f (bce=%.4f, dice=%.4f)",
            epoch,
            args.smoke_epochs,
            train_loss,
            train_bce,
            train_dice,
        )
        losses.append(train_loss)

    assert all(math.isfinite(loss) for loss in losses), f"Non-finite loss encountered: {losses}"
    if losses[-1] > losses[0]:
        logging.warning(
            "Smoke test: loss rose from %.4f to %.4f (may be noise)", losses[0], losses[-1]
        )
    logging.info("Smoke test passed: losses=%s", [f"{loss:.4f}" for loss in losses])


def main() -> None:
    """Train a binary segmentation model, then evaluate Dice/PPV/NPV on the validation set."""
    args = parse_args()
    if args.smoke_test:
        run_smoke_test(args)
        return

    configure_logging()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch=args.arch, encoder=args.encoder).to(device)
    criterion = build_loss_function(args.bce_weight, args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    train_loader, val_loader = build_dataloaders(
        args.train_csv, args.val_csv, args.batch_size, args.num_workers
    )
    logging.info(
        "Training %s/%s for %d epochs on %s (train=%d, val=%d)",
        args.arch,
        args.encoder,
        args.epochs,
        device,
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    best_dice = -1.0
    best_metrics: dict[float, dict[str, float]] = {}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_bce, train_dice = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        scheduler.step()

        metrics = evaluate(model, val_loader, device, args.thresholds)
        epoch_dice = max(m["dice"] for m in metrics.values())
        logging.info(
            "Epoch %d/%d - loss=%.4f (bce=%.4f, dice=%.4f) - lr=%.2e - val_dice=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_bce,
            train_dice,
            scheduler.get_last_lr()[0],
            epoch_dice,
        )

        if epoch_dice > best_dice:
            best_dice = epoch_dice
            best_metrics = metrics
            model.save_pretrained(checkpoint_dir)
            logging.info(
                "Epoch %d: new best val_dice=%.4f, saved checkpoint to %s",
                epoch,
                best_dice,
                checkpoint_dir,
            )

    print(format_metrics_table(best_metrics))


if __name__ == "__main__":
    main()
