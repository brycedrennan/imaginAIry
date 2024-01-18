import subprocess
import sys
from functools import lru_cache

from setuptools import find_packages, setup

is_for_windows = len(sys.argv) >= 3 and sys.argv[2].startswith("--plat-name=win")

if is_for_windows:
    scripts = None
    entry_points: dict | None = {
        "console_scripts": [
            "imagine=imaginairy.cli.main:imagine_cmd",
            "aimg=imaginairy.cli.main:aimg",
        ],
    }
else:
    scripts = ["imaginairy/cli/bin/aimg", "imaginairy/cli/bin/imagine"]
    entry_points = None


@lru_cache
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
    version="14.1.1",
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
            "weight_management/weight_maps/*.json",
            "data/*.*",
            "cli/bin/*.*",
            "http_app/stablestudio/dist/*.*",
            "http_app/stablestudio/dist/assets/*.*",
            "http_app/stablestudio/dist/LICENSE",
            "enhancers/phraselists/*.txt",
            "vendored/clip/*.txt.gz",
            "vendored/clipseg/*.pth",
            "vendored/blip/configs/*.*",
            "vendored/noodle_soup_prompts/*.*",
            "vendored/noodle_soup_prompts/LICENSE",
            "vendored/refiners/foundationals/clip/bpe_simple_vocab_16e6.txt.gz",
        ]
    },
    install_requires=[
        "click>=8.0.0",
        "click-help-colors>=0.9.1",
        "click-shell>=2.0",
        "protobuf != 3.20.2, != 3.19.5",
        "fastapi>=0.70.0",
        "ftfy>=6.0.1",  # for vendored clip
        "torch>=2.1.0",
        # https://numpy.org/neps/nep-0029-deprecation_policy.html
        "numpy>=1.22.0",
        "tqdm>=4.64.0",
        "diffusers>=0.3.0",
        "Pillow>=9.1.0",
        "psutil>5.7.3",
        "omegaconf>=2.1.1",
        "open-clip-torch>=2.0.0",
        "opencv-python>=4.4.0.46",
        # need to migration to 2.0
        "pydantic>=2.3.0",
        # pyparsing used for masking logic and creating text images
        "pyparsing>=3.0.0",
        "requests>=2.28.1",
        # "refiners>=0.2.0",
        "jaxtyping>=0.2.23",  # refiners dependency
        "einops>=0.3.0",
        "safetensors>=0.4.0",
        "scipy>=1.8",
        "termcolor",
        "timm>=0.4.12,!=0.9.0,!=0.9.1",  # for vendored blip
        "torchdiffeq>=0.2.0",
        "torchvision>=0.13.1",
        "transformers>=4.19.2",
        "triton>=2.0.0; sys_platform!='darwin' and platform_machine!='aarch64' and sys_platform == 'linux'",
        "kornia>=0.6",
        "uvicorn>=0.16.0",
        # "xformers>=0.0.22; sys_platform!='darwin' and platform_machine!='aarch64'",
    ],
    # don't specify maximum python versions as it can cause very long dependency resolution issues as the resolver
    # goes back to older versions of packages that didn't specify a maximum
    # https://discuss.python.org/t/requires-python-upper-limits/12663/75
    # https://github.com/brycedrennan/imaginAIry/pull/341#issuecomment-1574723908
    python_requires=">=3.10",
)
