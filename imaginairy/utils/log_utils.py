"""Utilities for image generation logging"""

import logging
import logging.config
import re
import time
import warnings
from contextlib import contextmanager
from functools import lru_cache
from logging import Logger
from typing import Callable

import torch.cuda

from imaginairy.utils.memory_tracker import TorchRAMTracker

_CURRENT_LOGGING_CONTEXT = None

logger = logging.getLogger(__name__)


def log_conditioning(conditioning, description):
    if _CURRENT_LOGGING_CONTEXT is None:
        return

    _CURRENT_LOGGING_CONTEXT.log_conditioning(conditioning, description)


def log_latent(latents, description):
    if _CURRENT_LOGGING_CONTEXT is None:
        return

    if latents is None:
        return

    _CURRENT_LOGGING_CONTEXT.log_latents(latents, description)


def log_img(img, description):
    if _CURRENT_LOGGING_CONTEXT is None:
        return
    _CURRENT_LOGGING_CONTEXT.log_img(img, description)


def log_progress_latent(latent):
    if _CURRENT_LOGGING_CONTEXT is None:
        return
    _CURRENT_LOGGING_CONTEXT.log_progress_latent(latent)


def log_tensor(t, description=""):
    if _CURRENT_LOGGING_CONTEXT is None:
        return
    _CURRENT_LOGGING_CONTEXT.log_img(t, description)


def increment_step():
    if _CURRENT_LOGGING_CONTEXT is None:
        return
    _CURRENT_LOGGING_CONTEXT.step_count += 1


@contextmanager
def timed_log_method(logger, level, msg, *args, hide_below_ms=0, **kwargs):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        if elapsed_ms < hide_below_ms:
            return
        full_msg = f"{msg} (in {elapsed_ms:.1f}ms)"
        logger.log(level, full_msg, *args, **kwargs, stacklevel=3)


@lru_cache
def add_timed_methods_to_logger():
    """Monkey patches the default python logger to have timed logs"""

    def create_timed_method(level):
        def timed_method(self, msg, *args, hide_below_ms=0, **kwargs):
            return timed_log_method(
                self, level, msg, *args, hide_below_ms=hide_below_ms, **kwargs
            )

        return timed_method

    logging.Logger.timed_debug = create_timed_method(logging.DEBUG)
    logging.Logger.timed_info = create_timed_method(logging.INFO)
    logging.Logger.timed_warning = create_timed_method(logging.WARNING)
    logging.Logger.timed_error = create_timed_method(logging.ERROR)
    logging.Logger.timed_critical = create_timed_method(logging.CRITICAL)


add_timed_methods_to_logger()


class TimedLogger(Logger):
    def timed_debug(self, msg, *args, hide_below_ms=0, **kwargs):
        return timed_log_method(
            self, logging.DEBUG, msg, *args, hide_below_ms=hide_below_ms, **kwargs
        )

    def timed_info(self, msg, *args, hide_below_ms=0, **kwargs):
        return timed_log_method(
            self, logging.INFO, msg, *args, hide_below_ms=hide_below_ms, **kwargs
        )

    def timed_warning(self, msg, *args, hide_below_ms=0, **kwargs):
        return timed_log_method(
            self, logging.WARNING, msg, *args, hide_below_ms=hide_below_ms, **kwargs
        )

    def timed_error(self, msg, *args, hide_below_ms=0, **kwargs):
        return timed_log_method(
            self, logging.ERROR, msg, *args, hide_below_ms=hide_below_ms, **kwargs
        )

    def timed_critical(self, msg, *args, hide_below_ms=0, **kwargs):
        return timed_log_method(
            self, logging.CRITICAL, msg, *args, hide_below_ms=hide_below_ms, **kwargs
        )


def getLogger(name) -> TimedLogger:
    return logging.getLogger(name)  # type: ignore


class TimingContext:
    """Tracks time and memory usage of a block of code"""

    def __init__(
        self,
        description: str,
        device: str | None = None,
        callback_fn: Callable | None = None,
    ):
        from imaginairy.utils import get_device

        self.description = description
        self._device = device or get_device()
        self.callback_fn = callback_fn

        self.start_time = None
        self.end_time = None
        self.duration = 0

        self.memory_context = None
        self.memory_start = None
        self.memory_end = 0
        self.memory_peak = 0
        self.memory_peak_delta = 0

    def start(self):
        # supports repeated calls to start/stop
        if self._device == "cuda":
            self.memory_context = TorchRAMTracker(self.description)
            self.memory_context.start()
            if self.memory_start is None:
                self.memory_start = self.memory_context.start_memory
            self.end_time = None
        self.start_time = time.time()

    def stop(self):
        # supports repeated calls to start/stop
        self.end_time = time.time()
        self.duration += self.end_time - self.start_time

        if self._device == "cuda":
            self.memory_context.stop()

            self.memory_end = self.memory_context.end_memory
            self.memory_peak = max(self.memory_context.peak_memory, self.memory_peak)
            self.memory_peak_delta = max(
                self.memory_context.peak_memory_delta, self.memory_peak_delta
            )

        if self.callback_fn is not None:
            self.callback_fn(self)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class ImageLoggingContext:
    def __init__(
        self,
        prompt,
        model=None,
        debug_img_callback=None,
        img_outdir=None,
        progress_img_callback=None,
        progress_img_interval_steps=3,
        progress_img_interval_min_s=0.1,
        progress_latent_callback=None,
    ):
        self.prompt = prompt
        self.model = model
        self.step_count = 0
        self.image_count = 0
        self.debug_img_callback = debug_img_callback
        self.img_outdir = img_outdir
        self.progress_img_callback = progress_img_callback
        self.progress_img_interval_steps = progress_img_interval_steps
        self.progress_img_interval_min_s = progress_img_interval_min_s
        self.progress_latent_callback = progress_latent_callback

        self.summary_context = TimingContext("total")
        self.summary_context.start()
        self.timing_contexts = {}
        self.last_progress_img_ts = 0
        self.last_progress_img_step = -1000

        self._prev_log_context = None

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        global _CURRENT_LOGGING_CONTEXT
        self._prev_log_context = _CURRENT_LOGGING_CONTEXT
        _CURRENT_LOGGING_CONTEXT = self
        return self

    def stop(self):
        global _CURRENT_LOGGING_CONTEXT
        _CURRENT_LOGGING_CONTEXT = self._prev_log_context

    def timing(self, description):
        if description not in self.timing_contexts:

            def cb(context):
                self.timing_contexts[description] = context

            tc = TimingContext(description, callback_fn=cb)
            self.timing_contexts[description] = tc
        return self.timing_contexts[description]

    def get_performance_stats(self) -> dict[str, dict[str, float]]:
        # calculate max peak seen in any timing context
        self.summary_context.stop()
        self.timing_contexts["total"] = self.summary_context

        # move total to the end
        self.timing_contexts["total"] = self.timing_contexts.pop("total")

        if torch.cuda.is_available():
            self.summary_context.memory_peak = max(
                max(context.memory_peak, context.memory_start, context.memory_end)
                for context in self.timing_contexts.values()
            )

        performance_stats = {}
        for context in self.timing_contexts.values():
            performance_stats[context.description] = {
                "duration": context.duration,
                "memory_start": context.memory_start,
                "memory_end": context.memory_end,
                "memory_peak": context.memory_peak,
                "memory_peak_delta": context.memory_peak_delta,
            }
        return performance_stats

    def log_conditioning(self, conditioning, description):
        if not self.debug_img_callback:
            return
        img = conditioning_to_img(conditioning)

        self.debug_img_callback(
            img, description, self.image_count, self.step_count, self.prompt
        )

    def log_latents(self, latents, description):
        if "predicted_latent" in description:
            if self.progress_latent_callback is not None:
                self.progress_latent_callback(latents)
            if (
                self.step_count - self.last_progress_img_step
            ) > self.progress_img_interval_steps and (
                time.perf_counter() - self.last_progress_img_ts
                > self.progress_img_interval_min_s
            ):
                self.log_progress_latent(latents)
                self.last_progress_img_step = self.step_count
                self.last_progress_img_ts = time.perf_counter()

        if not self.debug_img_callback:
            return
        if latents.shape[1] != 4:
            # logger.info(f"Didn't save tensor of shape {samples.shape} for {description}")
            return

        try:
            shape_str = ",".join(tuple(latents.shape))
        except TypeError:
            shape_str = str(latents.shape)
        description = f"{description}-{shape_str}"
        for latent in latents:
            self.image_count += 1
            latent = latent.unsqueeze(0)
            img = latent_to_raw_image(latent)
            self.debug_img_callback(
                img, description, self.image_count, self.step_count, self.prompt
            )
        # for img in model_latents_to_pillow_imgs(latents):
        #     self.image_count += 1
        #     self.debug_img_callback(
        #         img, description, self.image_count, self.step_count, self.prompt
        #     )

    def log_img(self, img, description):
        if not self.debug_img_callback:
            return
        import torch
        from torchvision.transforms import ToPILImage

        self.image_count += 1
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img.squeeze().cpu().detach())
        img = img.copy()
        self.debug_img_callback(
            img, description, self.image_count, self.step_count, self.prompt
        )

    def log_progress_latent(self, latent):
        from imaginairy.utils.img_utils import model_latents_to_pillow_imgs

        if not self.progress_img_callback:
            return
        for img in model_latents_to_pillow_imgs(latent):
            self.progress_img_callback(img)

    def log_tensor(self, t, description=""):
        if not self.debug_img_callback:
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
    from torchvision.transforms import ToPILImage

    return ToPILImage()(conditioning)


class ColorIndentingFormatter(logging.Formatter):
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    def format(self, record):
        s = super().format(record)
        color = ""
        reset = ""
        if record.levelno >= logging.ERROR:
            color = self.RED
        elif record.levelno >= logging.WARNING:
            color = self.YELLOW

        if _CURRENT_LOGGING_CONTEXT is not None:
            s = f"    {s}"

        if color is None and not s.startswith("    "):
            color = self.GREEN

        if color:
            reset = self.RESET
        s = f"{color}{s}{reset}"
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
                "class": "imaginairy.utils.log_utils.ColorIndentingFormatter",
            },
        },
        "handlers": {
            "default": {
                "level": level,
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
            # disable https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/models/clip/configuration_clip.py#L330
            "transformers.models.clip.configuration_clip": {
                "handlers": ["default"],
                "level": "ERROR",
                "propagate": False,
            },
            # disable the stupid triton is not available messages
            # https://github.com/facebookresearch/xformers/blob/6425fd0cacb1a6579aa2f0c4a570b737cb10e9c3/xformers/__init__.py#L52
            "xformers": {"handlers": ["default"], "level": "ERROR", "propagate": False},
        },
    }
    suppress_annoying_logs_and_warnings()
    logging.config.dictConfig(LOGGING_CONFIG)


def disable_transformers_custom_logging():
    from transformers.modeling_utils import logger as modeling_logger
    from transformers.utils.logging import _configure_library_root_logger

    _configure_library_root_logger()
    _logger = modeling_logger.parent
    _logger.handlers = []
    _logger.propagate = True
    _logger.setLevel(logging.NOTSET)
    modeling_logger.handlers = []
    modeling_logger.propagate = True
    modeling_logger.setLevel(logging.ERROR)


def disable_pytorch_lighting_custom_logging():
    try:
        from pytorch_lightning import _logger as pytorch_logger
    except ImportError:
        return

    try:
        from pytorch_lightning.utilities.seed import log

        log.setLevel(logging.NOTSET)
        log.handlers = []
        log.propagate = False
    except ImportError:
        pass
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
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*?torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument..*?",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*?is not currently supported on the MPS backend and will fall back.*?",
    )


def suppress_annoying_logs_and_warnings():
    disable_transformers_custom_logging()
    # disable_pytorch_lighting_custom_logging()
    disable_common_warnings()


def latent_to_raw_image(tensor):
    """
    Converts a tensor of size (1, 4, x, y) into a PIL image of size (x*4, y*4).

    Args:
    tensor (numpy.ndarray): A tensor of size (1, 4, x, y).

    Returns:
    PIL.Image: An image representing the tensor.
    """
    from PIL import Image

    if tensor.ndim != 4 or tensor.shape[0] != 1:
        msg = f"Tensor must be of shape (1, c, x, y). got shape: {tensor.shape}"
        raise ValueError(msg)

    _, c, x, y = tensor.shape

    full_image = Image.new("L", (x, y * c))

    # Process each channel
    for i in range(c):
        # Extract the channel
        channel = tensor[0, i, :, :]

        # Normalize and convert to an image
        channel_image = Image.fromarray(
            (channel / channel.max() * 255).cpu().numpy().astype("uint8")
        )

        # Paste the channel image into the full image
        full_image.paste(channel_image, (0, i * y))

    return full_image
