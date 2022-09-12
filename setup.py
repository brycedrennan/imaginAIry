from setuptools import find_packages, setup

setup(
    name="imaginairy",
    version="0.0.1",
    description="AI imagined images.",
    packages=find_packages(include=("imaginairy", "imaginairy.*")),
    entry_points={
        "console_scripts": ["imagine=imaginairy.cmd_wrap:imagine_cmd"],
    },
    package_data={"imaginairy": ["configs/*.yaml"]},
    install_requires=[
        "click",
        "torch",
        "numpy",
        "tqdm",
        "diffusers",
        "imageio==2.9.0",
        "pytorch-lightning==1.4.2",
        "omegaconf==2.1.1",
        "einops==0.3.0",
        "transformers==4.19.2",
        "torchmetrics==0.6.0",
        "torchvision>=0.13.1",
        "kornia==0.6",
        "clip @  git+https://github.com/openai/CLIP",
        # k-diffusion for use with find_noise.py
        # "k-diffusion@git+https://github.com/crowsonkb/k-diffusion.git@71ba7d6735e9cba1945b429a21345960eb3f151c#egg=k-diffusion",
    ],
)
