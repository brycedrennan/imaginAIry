# only builtin imports allowed at this point since we want to modify
# the environment and code before it's loaded
import importlib.abc
import importlib.util
import logging.config
import os
import site
import sys
import warnings

# tells pytorch to allow MPS usage (for Mac M1 compatibility)
os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def disable_transformers_logging():
    """
    Disable `transformers` package custom logging.

    I can't believe it came to this. I tried like four other approaches first

    Loads up the source code from the transformers file and turns it into a module.
    We then modify the module.  Every other approach (import hooks, custom import function)
    loaded the module before it could be modified.
    """
    t_logging_path = f"{site.getsitepackages()[0]}/transformers/utils/logging.py"
    with open(t_logging_path, "r", encoding="utf-8") as f:
        src_code = f.read()

    spec = importlib.util.spec_from_loader("transformers.utils.logging", loader=None)
    module = importlib.util.module_from_spec(spec)

    exec(src_code, module.__dict__)
    module.get_logger = logging.getLogger
    sys.modules["transformers.utils.logging"] = module


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


def setup_env():
    disable_transformers_logging()
    disable_pytorch_lighting_custom_logging()
    disable_common_warnings()


def imagine_cmd(*args, **kwargs):
    setup_env()
    from imaginairy.cmds import imagine_cmd as imagine_cmd_orig  # noqa

    imagine_cmd_orig(*args, **kwargs)


if __name__ == "__main__":
    imagine_cmd()  # noqa
