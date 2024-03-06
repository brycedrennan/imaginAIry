import os.path
from functools import lru_cache
from pathlib import Path

from modal import (
    Image,
    Mount,
    Stub,
    Volume,
    gpu,
)

os.environ["MODAL_AUTOMOUNT"] = "0"

# find project root that is two levels up from __file__
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
requirements_path = os.path.join(project_root, "requirements-dev.txt")


def file_filter(path: str):
    # print(f"Checking {path}")
    if "/tests/" in path:
        return False
    include = path in files_for_inclusion()
    return include


@lru_cache
def files_for_inclusion():
    from imaginairy.utils.gitignore import get_nonignored_file_paths

    filepaths = get_nonignored_file_paths(project_root)
    for f in filepaths:
        print(f)
    filepaths = [f"{project_root}/{f}" for f in filepaths if os.path.isfile(f)]
    return set(filepaths)


local_mount = Mount.from_local_dir(
    project_root, remote_path="/root/workspace/imaginairy", condition=file_filter
)

imaginairy_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install_from_requirements(requirements_path)
    .workdir("/root/workspace")
    .env({"PYTHONPATH": "/root/workspace/imaginairy"})
)
weights_cache_volume = Volume.from_name("ai-wights-cache", create_if_missing=True)
weights_cache_path = "/root/.cache/"

stub = Stub(
    "imaginairy",
    mounts=[local_mount],
    volumes={weights_cache_path: weights_cache_volume},
)


@stub.function(gpu=gpu.A100(), container_idle_timeout=2, image=imaginairy_image)
def generate_image(imagine_prompt):
    from imaginairy.api import imagine
    from imaginairy.utils.log_utils import configure_logging

    configure_logging()

    results = list(imagine(imagine_prompt))
    result = results[-1]

    weights_cache_volume.commit()
    return result


standard_prompts = [
    "a flower",
    "portrait photo of a woman with a few freckles. red hair",
    "photo of a bowl of fruit",
    "gold coins",
    "a scenic landscape",
    "old growth redwood forest with a stream meandering through it. award-winning photo, diffuse lighting, beautiful, high-resolution, 4k",
    "abstract desktop background",
    "highway system seen from above",
    "the galaxy",
    "a photo of the rainforest",
    "the starry night painting",
    "photo of flowers",
    "girl with a pearl earring",
    "a painting of 1920s luncheon on a boat",
    "napolean crossing the alps",
    "american gothic painting",
    "the tower of babel",
    "god creating the {universe|earth}",
    "a fishing boat in a storm on the sea of galilee. oil painting",
    "the american flag",
    "the tree of life. oil painting",
    "the last supper. oil painting",
    "the statue of liberty",
    "a maze",
    "HD desktop wallpaper",
    "a beautiful garden",
    "the garden of eden",
    "the circus. {photo|oil painting}",
    "a futuristic city",
    "a retro spaceship",
    "yosemite national park. {photo|oil painting}",
    "a seacliff with a lighthouse. giant ocean waves, {photo|oil painting}",
    "blueberries",
    "strawberries",
    "a single giant diamond",
    "disneyland thomas kinkade painting",
    "ancient books in an ancient library. cinematic",
    "mormon missionaries",
    "salt lake city",
    "oil painting of heaven",
    "oil painting of hell",
    "an x-wing. digital art",
    "star trek uss enterprise",
    "a giant pile of treasure",
    "the white house",
    "a grizzly bear. nature photography",
    "a unicorn. nature photography",
    "elon musk. normal rockwell painting",
    "a cybertruck",
    "elon musk with a halo dressed in greek robes. oil painting",
    "a crowd of people",
    "a stadium full of people",
    "macro photography of a drop of water",
    "macro photography of a leaf",
    "macro photography of a spider",
    "flames",
    "a robot",
    "the stars at night",
    "a lovely sunset",
    "an oil painting",
]


@stub.local_entrypoint()
def main(
    prompt: str,
    size: str = "fhd",
    upscale: bool = False,
    model_weights: str = "opendalle",
    n_images: int = 1,
    n_steps: int = 50,
    seed=None,
):
    from imaginairy.enhancers.prompt_expansion import expand_prompts
    from imaginairy.schema import ImaginePrompt
    from imaginairy.utils import get_next_filenumber
    from imaginairy.utils.log_utils import configure_logging

    configure_logging()

    prompt_texts = expand_prompts(
        n=n_images,
        prompt_text=prompt,
    )
    # model_weights = ModelWeightsConfig(
    #     name="ProteusV0.4",
    #     aliases=["proteusv4"],
    #     architecture="sdxl",
    #     defaults={
    #         "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    #         "composition_strength": 0.6,
    #     },
    #     weights_location="https://huggingface.co/dataautogpt3/ProteusV0.4/tree/0dfa4101db540e7a4b2b6ba6f87d8d7219e84513",
    # )

    imagine_prompts = [
        ImaginePrompt(
            prompt_text,
            steps=n_steps,
            size=size,
            upscale=upscale,
            model_weights=model_weights,
            seed=seed,
        )
        for prompt_text in prompt_texts
    ]
    # imagine_prompts = []
    # for sp in standard_prompts:
    #     imagine_prompts.append(
    #         ImaginePrompt(
    #             sp,
    #             steps=n_steps,
    #             size=size,
    #             upscale=upscale,
    #             model_weights=model_weights,
    #             seed=seed
    #         )
    #     )
    # imagine_prompts = imagine_prompts[:n_images]

    outdir = Path("./outputs/modal-inference")
    outdir.mkdir(exist_ok=True, parents=True)
    file_num = get_next_filenumber(f"{outdir}/generated")

    for result in generate_image.map(imagine_prompts):
        from imaginairy.api.generate import save_image_result

        save_image_result(
            result=result,
            base_count=file_num,
            outdir=outdir,
            output_file_extension="jpg",
            primary_filename_type="generated",
        )
        file_num += 1
