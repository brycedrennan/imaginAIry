import logging
import logging.config
import re
import time
import warnings

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


class TimingContext:
    def __init__(self, logging_context, description):
        self.logging_context = logging_context
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.logging_context.timings[self.description] = time.time() - self.start_time


class ImageLoggingContext:
    def __init__(
        self,
        prompt,
        model,
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

        self.start_ts = time.perf_counter()
        self.timings = {}
        self.last_progress_img_ts = 0
        self.last_progress_img_step = -1000

        self._prev_log_context = None

    def __enter__(self):
        global _CURRENT_LOGGING_CONTEXT
        self._prev_log_context = _CURRENT_LOGGING_CONTEXT
        _CURRENT_LOGGING_CONTEXT = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_LOGGING_CONTEXT
        _CURRENT_LOGGING_CONTEXT = self._prev_log_context

    def timing(self, description):
        return TimingContext(self, description)

    def get_timings(self):
        self.timings["total"] = time.perf_counter() - self.start_ts
        return self.timings

    def log_conditioning(self, conditioning, description):
        if not self.debug_img_callback:
            return
        img = conditioning_to_img(conditioning)

        self.debug_img_callback(
            img, description, self.image_count, self.step_count, self.prompt
        )

    def log_latents(self, latents, description):
        from imaginairy.img_utils import model_latents_to_pillow_imgs

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
        for img in model_latents_to_pillow_imgs(latents):
            self.image_count += 1
            self.debug_img_callback(
                img, description, self.image_count, self.step_count, self.prompt
            )

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
        from imaginairy.img_utils import model_latents_to_pillow_imgs

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
    from pytorch_lightning import _logger as pytorch_logger

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


def suppress_annoying_logs_and_warnings():
    disable_transformers_custom_logging()
    disable_pytorch_lighting_custom_logging()
    disable_common_warnings()
