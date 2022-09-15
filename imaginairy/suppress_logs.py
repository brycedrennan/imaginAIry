import logging.config
import warnings


def disable_transformers_custom_logging():
    from transformers.modeling_utils import logger
    from transformers.utils.logging import _configure_library_root_logger

    _configure_library_root_logger()
    logger = logger.parent
    logger.handlers = []
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def disable_pytorch_lighting_custom_logging():
    from pytorch_lightning import _logger

    _logger.setLevel(logging.NOTSET)


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
