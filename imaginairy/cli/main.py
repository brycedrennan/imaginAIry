import logging

import click

from imaginairy.cli.clickshell_mod import ColorShell, ImagineColorsCommand
from imaginairy.cli.colorize import colorize_cmd
from imaginairy.cli.describe import describe_cmd
from imaginairy.cli.edit import edit_cmd
from imaginairy.cli.edit_demo import edit_demo_cmd
from imaginairy.cli.imagine import imagine_cmd
from imaginairy.cli.run_api import run_server_cmd
from imaginairy.cli.train import prep_images_cmd, prune_ckpt_cmd, train_concept_cmd
from imaginairy.cli.upscale import upscale_cmd
from imaginairy.cli.videogen import videogen_cmd

logger = logging.getLogger(__name__)


@click.command(
    prompt="🤖🧠> ",
    intro="Starting imaginAIry shell...",
    help_headers_color="yellow",
    help_options_color="green",
    context_settings={"max_content_width": 140},
    cls=ColorShell,
)
@click.pass_context
def aimg(ctx):
    """
    🤖🧠 ImaginAIry.

    Pythonic generation of images via AI
    """
    import sys

    is_shell = len(sys.argv) == 1
    if is_shell:
        print(ctx.get_help())


aimg.command_class = ImagineColorsCommand


aimg.add_command(colorize_cmd, name="colorize")
aimg.add_command(describe_cmd, name="describe")
aimg.add_command(edit_cmd, name="edit")
aimg.add_command(edit_demo_cmd, name="edit-demo")
aimg.add_command(imagine_cmd, name="imagine")
aimg.add_command(prep_images_cmd, name="prep-images")
aimg.add_command(prune_ckpt_cmd, name="prune-ckpt")
aimg.add_command(train_concept_cmd, name="train-concept")
aimg.add_command(upscale_cmd, name="upscale")
aimg.add_command(run_server_cmd, name="server")
aimg.add_command(videogen_cmd, name="videogen")


@aimg.command()
def version():
    """Print the version."""
    from imaginairy.version import get_version

    print(get_version())


@aimg.command("system-info")
def system_info():
    """
    Display system information. Submit this when reporting bugs.
    """
    from imaginairy.debug_info import get_debug_info

    for k, v in get_debug_info().items():
        k += ":"
        click.secho(f"{k: <30} {v}")


@aimg.command("model-list")
def model_list_cmd():
    """Print list of available models."""
    from imaginairy import config

    print(f"{'ALIAS': <10} {'NAME': <18} {'DESCRIPTION'}")
    for model_config in config.MODEL_CONFIGS:
        print(
            f"{model_config.alias: <10} {model_config.short_name: <18} {model_config.description}"
        )

    print("\nCONTROL MODES:")
    print(f"{'ALIAS': <10} {'NAME': <18} {'CONTROL TYPE'}")
    for control_mode in config.CONTROLNET_CONFIGS:
        print(
            f"{control_mode.alias: <10} {control_mode.short_name: <18} {control_mode.control_type}"
        )


if __name__ == "__main__":
    aimg()
