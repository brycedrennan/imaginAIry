from setuptools import setup, find_packages

setup(
    name="imaginairy",
    version="0.0.1",
    description="AI imagined images.",
    packages=find_packages(include=("imaginairy", "imaginairy.*")),
    entry_points={
        "console_scripts": ["imagine=imaginairy.cmds:imagine_cmd"],
    },
    package_data={"imaginairy": ["configs/*.yaml"]},
    install_requires=[
        "click",
        "torch",
        "numpy",
        "tqdm",
        # "albumentations==0.4.3",
        # "diffusers",
        # opencv-python==4.1.2.30
        # "pudb==2019.2",
        # "invisible-watermark",
        "imageio==2.9.0",
        # "imageio-ffmpeg==0.4.2",
        "pytorch-lightning==1.4.2",
        "omegaconf==2.1.1",
        # "test-tube>=0.7.5",
        # "streamlit>=0.73.1",
        "einops==0.3.0",
        # "torch-fidelity==0.3.0",
        "transformers==4.19.2",
        "torchmetrics==0.6.0",
        "torchvision>=0.13.1",
        "kornia==0.6",
        # "realesrgan",
        # "-e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers",
        "clip @  git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1#egg=clip",
    ],
)
