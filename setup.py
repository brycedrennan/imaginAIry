import subprocess
import sys
from functools import lru_cache

from setuptools import find_packages, setup

is_for_windows = len(sys.argv) >= 3 and sys.argv[2].startswith("--plat-name=win")

if is_for_windows:
    scripts = None
    entry_points = {
        "console_scripts": [
            "imagine=imaginairy.cli.main:imagine_cmd",
            "aimg=imaginairy.cli.main:aimg",
        ],
    }
else:
    scripts = ["imaginairy/bin/aimg", "imaginairy/bin/imagine"]
    entry_points = None


@lru_cache()
def get_git_revision_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except FileNotFoundError:
        return "no-git"


revision_hash = get_git_revision_hash()

with open("README.md", encoding="utf-8") as f:
    readme = f.read()
    readme = readme.replace(
        '<img src="',
        f'<img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/{revision_hash}/',
    )

setup(
    name="imaginAIry",
    author="Bryce Drennan",
    # author_email="b r y p y d o t io",
    version="13.0.1",
    description="AI imagined images. Pythonic generation of images.",
    long_description=readme,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://github.com/brycedrennan/imaginAIry/blob/master/README.md",
        "Source": "https://github.com/brycedrennan/imaginAIry",
    },
    packages=find_packages(include=("imaginairy", "imaginairy.*")),
    scripts=scripts,
    entry_points=entry_points,
    package_data={
        "imaginairy": [
            "configs/*.yaml",
            "data/*.*",
            "bin/*.*",
            "http/stablestudio/dist/*.*",
            "http/stablestudio/dist/assets/*.*",
            "http/stablestudio/dist/LICENSE",
            "enhancers/phraselists/*.txt",
            "vendored/clip/*.txt.gz",
            "vendored/clipseg/*.pth",
            "vendored/blip/configs/*.*",
            "vendored/noodle_soup_prompts/*.*",
            "vendored/noodle_soup_prompts/LICENSE",
        ]
    },
    install_requires=[
        "click",
        "click-help-colors",
        "click-shell",
        "protobuf != 3.20.2, != 3.19.5",
        "facexlib",
        "fairscale>=0.4.4",  # for vendored blip
        "fastapi",
        "ftfy",  # for vendored clip
        # 2.0.0 produced garbage images on MacOS
        "torch>=1.13.1,<2.0.0",
        "numpy",
        "tqdm",
        "diffusers",
        "imageio>=2.9.0",
        "Pillow>=9.1.0",
        "psutil",
        # 2.0.0 need to fix `ImportError: cannot import name 'rank_zero_only' from 'pytorch_lightning.utilities.distributed' `
        "pytorch-lightning>=1.4.2,<2.0.0",
        "omegaconf>=2.1.1",
        "open-clip-torch",
        "opencv-python",
        # need to migration to 2.0
        "pydantic<2.0.0",
        "requests",
        "einops>=0.3.0",
        "safetensors",
        # scipy is a sub dependency but v1.11 doesn't support python 3.8.  https://docs.scipy.org/doc/scipy/dev/toolchain.html#numpy
        "scipy<1.11",
        "timm>=0.4.12,!=0.9.0,!=0.9.1",  # for vendored blip
        "torchdiffeq",
        "transformers>=4.19.2",
        "torchmetrics>=0.6.0",
        "torchvision>=0.13.1",
        "kornia>=0.6",
        "uvicorn",
        "xformers>=0.0.16; sys_platform!='darwin' and platform_machine!='aarch64'",
    ],
    # torchvision doesn't support python 3.11 unless we switch to torch 2.0
    python_requires=">=3.8,<3.11",
)
