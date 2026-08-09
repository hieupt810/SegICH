import logging
import random

import torch


def configure_logging() -> None:
    """Configure stderr logging so progress/warnings don't pollute stdout."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def set_seed(seed: int) -> None:
    """Seed python's random, numpy, and torch (CPU + all CUDA devices) RNGs for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
