import os

# tells pytorch to allow MPS usage (for Mac M1 compatibility)
os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# use more memory than we should
os.putenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import sys  # noqa

from .api import imagine, imagine_image_files  # noqa
from .schema import (  # noqa
    ImaginePrompt,
    ImagineResult,
    LazyLoadingImage,
    WeightedPrompt,
)

# if python version is 3.11 or higher, throw an exception
if sys.version_info >= (3, 11):
    msg = (
        "Imaginairy is not compatible with Python 3.11 or higher. Please use Python 3.8 - 3.10.\n"
        "This is due to torch 1.13 not supporting Python 3.11 and this library not having yet switched "
        "to torch 2.0"
    )
    raise RuntimeError(msg)
