"""Cross-cutting helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def use_truststore() -> None:
    """Route SSL through the OS cert store (helpful on Windows / corporate networks)."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
