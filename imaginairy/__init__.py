import os.path

# tells pytorch to allow MPS usage (for Mac M1 compatibility)
os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import PIL.Image  # noqa

from .api import imagine, imagine_image_files  # noqa
from .enhancers.describe_image_blip import generate_caption  # noqa
from .schema import (  # noqa
    ImaginePrompt,
    ImagineResult,
    LazyLoadingImage,
    WeightedPrompt,
)

# https://stackoverflow.com/questions/71738218/module-pil-has-not-attribute-resampling
if not hasattr(PIL.Image, "Resampling"):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image

PKG_ROOT = os.path.dirname(__file__)
