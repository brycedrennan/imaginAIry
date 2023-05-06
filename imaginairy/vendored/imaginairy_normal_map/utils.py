from functools import lru_cache

import torch


@lru_cache()
def get_device() -> str:
    """Return the best torch backend available."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps:0"

    return "cpu"
