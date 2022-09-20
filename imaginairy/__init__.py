import os.path

# tells pytorch to allow MPS usage (for Mac M1 compatibility)
os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .api import imagine, imagine_image_files  # noqa
from .enhancers.describe_image_blip import generate_caption  # noqa
from .schema import (  # noqa
    ImaginePrompt,
    ImagineResult,
    LazyLoadingImage,
    WeightedPrompt,
)

PKG_ROOT = os.path.dirname(__file__)
