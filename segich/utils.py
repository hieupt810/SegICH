import logging
import random
from pathlib import Path

import torch


def configure_logging(log_file: Path | str | None = None) -> None:
    """Configure stderr logging, optionally also duplicating INFO+ records to a log file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", handlers=handlers)


def set_seed(seed: int) -> None:
    """Seed python's random, numpy, and torch (CPU + all CUDA devices) RNGs for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
