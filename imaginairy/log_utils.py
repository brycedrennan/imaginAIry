import logging
import logging.config
import re
import warnings

import torch
from pytorch_lightning import _logger as pytorch_logger
from torchvision.transforms import ToPILImage
from transformers.modeling_utils import logger as modeling_logger
from transformers.utils.logging import _configure_library_root_logger

_CURRENT_LOGGING_CONTEXT = None

logger = logging.getLogger(__name__)


def log_conditioning(conditioning, description):
    if _CURRENT_LOGGING_CONTEXT is None:
        return

    _CURRENT_LOGGING_CONTEXT.log_conditioning(conditioning, description)


def log_latent(latents, description):
    if _CURRENT_LOGGING_CONTEXT is None:
        return

    _CURRENT_LOGGING_CONTEXT.log_latents(latents, description)


def log_img(img, description):
    if _CURRENT_LOGGING_CONTEXT is None:
        return
    _CURRENT_LOGGING_CONTEXT.log_img(img, description)


def log_tensor(t, description=""):
    if _CURRENT_LOGGING_CONTEXT is None:
        return
    _CURRENT_LOGGING_CONTEXT.log_img(t, description)


class ImageLoggingContext:
    def __init__(self, prompt, model, img_callback=None, img_outdir=None):
        self.prompt = prompt
        self.model = model
        self.step_count = 0
        self.img_callback = img_callback
        self.img_outdir = img_outdir

    def __enter__(self):
        global _CURRENT_LOGGING_CONTEXT  # noqa
        _CURRENT_LOGGING_CONTEXT = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_LOGGING_CONTEXT  # noqa
        _CURRENT_LOGGING_CONTEXT = None

    def log_conditioning(self, conditioning, description):
        if not self.img_callback:
            return
        img = conditioning_to_img(conditioning)

        self.img_callback(img, description, self.step_count, self.prompt)

    def log_latents(self, latents, description):
        from imaginairy.img_utils import model_latents_to_pillow_imgs  # noqa

        if not self.img_callback:
            return
        if latents.shape[1] != 4:
            # logger.info(f"Didn't save tensor of shape {samples.shape} for {description}")
            return
        self.step_count += 1
        try:
            shape_str = ",".join(tuple(latents.shape))
        except TypeError:
            shape_str = str(latents.shape)
        description = f"{description}-{shape_str}"
        for img in model_latents_to_pillow_imgs(latents):
            self.img_callback(img, description, self.step_count, self.prompt)

    def log_img(self, img, description):
        if not self.img_callback:
            return
        self.step_count += 1
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img.squeeze().cpu().detach())
        img = img.copy()
        self.img_callback(img, description, self.step_count, self.prompt)

    def log_tensor(self, t, description=""):
        if not self.img_callback:
            return

        if len(t.shape) == 2:
            self.log_img(t, description)

    def log_indexed_graph_of_tensor(self):
        pass

    # def img_callback(self, img, description, step_count, prompt):
    #     steps_path = os.path.join(self.img_outdir, "steps", f"{self.file_num:08}_S{prompt.seed}")
    #     os.makedirs(steps_path, exist_ok=True)
    #     filename = f"{self.file_num:08}_S{prompt.seed}_step{step_count:04}_{filesafe_text(description)[:40]}.jpg"
    #     destination = os.path.join(steps_path, filename)
    #     draw = ImageDraw.Draw(img)
    #     draw.text((10, 10), str(description))
    #     img.save(destination)


def filesafe_text(t):
    return re.sub(r"[^a-zA-Z0-9.,\[\]() -]+", "_", t)[:130]


def conditioning_to_img(conditioning):
    return ToPILImage()(conditioning)


class IndentingFormatter(logging.Formatter):
    def format(self, record):
        s = super().format(record)
        if _CURRENT_LOGGING_CONTEXT is not None:
            s = f"    {s}"
        return s


def configure_logging(level="INFO"):
    fmt = "%(message)s"
    if level == "DEBUG":
        fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {
                "format": fmt,
                "class": "imaginairy.log_utils.IndentingFormatter",
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",  # Default is stderr
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["default"],
                "level": "WARNING",
                "propagate": False,
            },
            "imaginairy": {"handlers": ["default"], "level": level, "propagate": False},
            "transformers.modeling_utils": {
                "handlers": ["default"],
                "level": "ERROR",
                "propagate": False,
            },
        },
    }
    suppress_annoying_logs_and_warnings()
    logging.config.dictConfig(LOGGING_CONFIG)


def disable_transformers_custom_logging():
    _configure_library_root_logger()
    _logger = modeling_logger.parent
    _logger.handlers = []
    _logger.propagate = True
    _logger.setLevel(logging.NOTSET)
    modeling_logger.handlers = []
    modeling_logger.propagate = True
    modeling_logger.setLevel(logging.ERROR)


def disable_pytorch_lighting_custom_logging():
    pytorch_logger.setLevel(logging.NOTSET)


def disable_common_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"The operator .*?is not currently supported.*",
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=r"The parameter 'pretrained' is.*"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=r"Arguments other than a weight.*"
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def suppress_annoying_logs_and_warnings():
    disable_transformers_custom_logging()
    disable_pytorch_lighting_custom_logging()
    disable_common_warnings()
