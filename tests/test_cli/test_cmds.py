import subprocess
from unittest import mock

import pytest
from click.testing import CliRunner

from imaginairy.cli.edit import edit_cmd
from imaginairy.cli.edit_demo import edit_demo_cmd
from imaginairy.cli.imagine import imagine_cmd
from imaginairy.cli.main import aimg
from imaginairy.cli.upscale import upscale_cmd
from imaginairy.schema import ImaginePrompt, LazyLoadingImage
from imaginairy.utils import surprise_me
from imaginairy.utils.model_cache import GPUModelCache
from tests import PROJECT_FOLDER, TESTS_FOLDER
from tests.utils import Timer


@pytest.mark.parametrize("subcommand_name", aimg.commands.keys())
def test_cmd_help_time(subcommand_name):
    cmd_parts = [
        "aimg",
        subcommand_name,
        "--help",
    ]
    with Timer(f"{subcommand_name} --help") as t:
        result = subprocess.run(
            cmd_parts, check=False, capture_output=True, cwd=PROJECT_FOLDER
        )
    assert result.returncode == 0, result.stderr
    assert t.elapsed < 1.0, f"{t.elapsed} > 1.0"


def test_model_info_cmd():
    runner = CliRunner()
    result = runner.invoke(
        aimg,
        [
            "model-list",
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_describe_cmd():
    runner = CliRunner()
    result = runner.invoke(
        aimg,
        [
            "describe",
            f"{TESTS_FOLDER}/data/dog.jpg",
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_colorize_cmd():
    runner = CliRunner()
    result = runner.invoke(
        aimg,
        [
            "colorize",
            f"{TESTS_FOLDER}/data/dog.jpg",
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_imagine_cmd(monkeypatch):
    monkeypatch.setattr(GPUModelCache, "make_gpu_space", mock.MagicMock())
    runner = CliRunner()
    result = runner.invoke(
        imagine_cmd,
        [
            "gold coins",
            "--steps",
            "2",
            "--outdir",
            f"{TESTS_FOLDER}/test_output",
            "--seed",
            "703425280",
            "--outdir",
            f"{TESTS_FOLDER}/test_output",
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_edit_cmd(monkeypatch):
    monkeypatch.setattr(GPUModelCache, "make_gpu_space", mock.MagicMock())
    runner = CliRunner()
    result = runner.invoke(
        edit_cmd,
        [
            f"{TESTS_FOLDER}/data/dog.jpg",
            "--steps",
            "1",
            "-p",
            "turn the dog into a cat",
            # "--model",
            # "empty",
            "--outdir",
            f"{TESTS_FOLDER}/test_output",
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_aimg_shell():
    runner = CliRunner()
    result = runner.invoke(
        aimg,
        [],
    )
    assert "Starting imaginAIry shell" in result.output
    assert result.exit_code == 0


def test_edit_demo(monkeypatch):
    runner = CliRunner()

    def mock_surprise_me_prompts(*args, **kwargs):
        return [
            ImaginePrompt(
                "",
                steps=1,
                size=256,
                # model="empty",
            )
        ]

    monkeypatch.setattr(surprise_me, "surprise_me_prompts", mock_surprise_me_prompts)
    monkeypatch.setattr(GPUModelCache, "make_gpu_space", mock.MagicMock())
    surprise_me.generic_prompts = []
    result = runner.invoke(
        edit_demo_cmd,
        [
            f"{TESTS_FOLDER}/data/dog.jpg",
            "--outdir",
            f"{TESTS_FOLDER}/test_output",
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_upscale(monkeypatch):
    from imaginairy.enhancers import upscale_realesrgan

    def mock_upscale_image(*args, **kwargs):
        return LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/dog.jpg")

    monkeypatch.setattr(upscale_realesrgan, "upscale_image", mock_upscale_image)
    runner = CliRunner()
    result = runner.invoke(
        upscale_cmd,
        [
            f"{TESTS_FOLDER}/data/dog.jpg",
            "--outdir",
            f"{TESTS_FOLDER}/test_output",
        ],
    )
    assert result.exit_code == 0
