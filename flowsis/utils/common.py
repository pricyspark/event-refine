import torch
import random
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_rng(
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Pass either 'rng' or 'seed', not both.")

    if rng is None:
        rng = np.random.default_rng(seed)

    return rng


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
