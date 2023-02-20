from click.testing import CliRunner

from imaginairy import ImaginePrompt, LazyLoadingImage, surprise_me
from imaginairy.cmds import aimg, edit_demo, edit_image, imagine_cmd, upscale_cmd
from tests import TESTS_FOLDER


def test_imagine_cmd():
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
            "--model",
            "empty",
            "--outdir",
            f"{TESTS_FOLDER}/test_output",
        ],
    )
    assert result.exit_code == 0


def test_edit_cmd():
    runner = CliRunner()
    result = runner.invoke(
        edit_image,
        [
            f"{TESTS_FOLDER}/data/dog.jpg",
            "--steps",
            "1",
            "-p",
            "turn the dog into a cat",
            "--model",
            "empty",
            "--outdir",
            f"{TESTS_FOLDER}/test_output",
        ],
    )
    assert result.exit_code == 0


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
                width=256,
                height=256,
                model="empty",
            )
        ]

    monkeypatch.setattr(surprise_me, "surprise_me_prompts", mock_surprise_me_prompts)
    surprise_me.generic_prompts = []
    result = runner.invoke(
        edit_demo,
        [
            f"{TESTS_FOLDER}/data/dog.jpg",
            "--outdir",
            f"{TESTS_FOLDER}/test_output",
        ],
    )
    assert result.exit_code == 0


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
