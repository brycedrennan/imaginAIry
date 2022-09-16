import os

# tells pytorch to allow MPS usage (for Mac M1 compatibility)
os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .api import imagine, imagine_image_files  # noqa
from .schema import (  # noqa
    ImaginePrompt,
    ImagineResult,
    LazyLoadingImage,
    WeightedPrompt,
)
