import os.path

import torch
from PIL import ImageDraw

from imaginairy import ImaginePrompt, LazyLoadingImage, imagine, imagine_image_files
from imaginairy.api import load_model
from imaginairy.img_log import ImageLoggingContext, filesafe_text, log_latent
from imaginairy.modules.clip_embedders import FrozenCLIPEmbedder
from imaginairy.samplers.ddim import DDIMSampler
from imaginairy.utils import get_device, pillow_img_to_torch_image
from tests import TESTS_FOLDER


def experiment_text_conditioning_combos():
    """
    Can we do math with the embeddings?

    Yes. it works but doesn't look great.
    """
    embedder = FrozenCLIPEmbedder()
    embedder.to(get_device())

    beach_e = embedder.encode(["a beach"])
    beach_water_e = embedder.encode(["a beach. ocean, waves, water"])
    waterness = beach_water_e - beach_e
    waterless_beach = beach_e - waterness

    imagine_image_files(
        [ImaginePrompt("waterless_beach", conditioning=waterless_beach, seed=1)],
        outdir=f"{TESTS_FOLDER}/test_output",
    )
    imagine_image_files(
        [ImaginePrompt("waterness", conditioning=waterness, seed=1)],
        outdir=f"{TESTS_FOLDER}/test_output",
    )
    imagine_image_files(
        [ImaginePrompt("beach", conditioning=beach_e, seed=1)],
        outdir=f"{TESTS_FOLDER}/test_output",
    )


def experiment_step_repeats():
    """
    Run the same step over and over on an image without noise

    Removes detail from the image.
    """
    model = load_model()
    model.to(get_device())
    model.eval()
    embedder = FrozenCLIPEmbedder()
    embedder.to(get_device())

    sampler = DDIMSampler(model)
    sampler.make_schedule(1000)

    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg")
    init_image, _, _ = pillow_img_to_torch_image(
        img,
    )
    init_image = init_image.to(get_device())
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
    log_latent(init_latent, "init_latent")

    base_count = 1
    neutral_embedding = embedder.encode([""])
    outdir = f"{TESTS_FOLDER}/test_output"

    def _record_step(img, description, step_count, prompt):
        steps_path = os.path.join(outdir, "steps", f"{base_count:08}_S{prompt.seed}")
        os.makedirs(steps_path, exist_ok=True)
        filename = f"{base_count:08}_S{prompt.seed}_step{step_count:04}_{filesafe_text(description)[:40]}.png"
        destination = os.path.join(steps_path, filename)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), str(description))
        img.save(destination)

    with ImageLoggingContext(
        prompt=ImaginePrompt(""),
        model=model,
        img_callback=_record_step,
    ):
        x_prev = init_latent
        index = 50
        base_count = index
        t = torch.Tensor([index]).to(get_device())
        # noise_pred = model.apply_model(init_latent, t, neutral_embedding)
        # log_latent(noise_pred, "noise prediction")
        for _ in range(100):
            x_prev, pred_x0 = sampler.p_sample_ddim(x_prev, neutral_embedding, t, index)
            log_latent(pred_x0, "pred_x0")
            x_prev = pred_x0


def experiment_repeated_img_2_img():
    """
    Experiment with putting an image repeatedly through image2image

    It creates screwy images
    """
    outdir = f"{TESTS_FOLDER}/test_output/img2img2img"
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/beach_at_sainte_adresse.jpg")
    img.save(f"{outdir}/0.png")
    for step_num in range(50):
        prompt = ImaginePrompt(
            "Beach at Sainte Adresse. hyperealistic photo. sharp focus, canon 5d",
            init_image=img,
            init_image_strength=0.50,
            width=512,
            height=512,
            steps=50,
            sampler_type="DDIM",
        )

        result = next(imagine(prompt))
        img = result.img
        os.makedirs(outdir, exist_ok=True)
        img.save(f"{outdir}/{step_num:04}.png")


def experiment_superresolution():
    """
    Try to trick it into making a superresolution image

    Did not work, resulting image was more blurry

    # i put this into the api.py file hardcoded
    row_a = torch.tensor([1, 0]).repeat(32)
    row_b = torch.tensor([0, 1]).repeat(32)
    grid = torch.stack([row_a, row_b]).repeat(32, 1)
    mask = grid
    mask = mask.to(get_device())
    """

    description = "a black and white photo of a dog's face"
    # image was a quarter of existing image
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/../outputs/dog02.jpg")

    # todo: try with 1000 mask at image resultion (rencoding entire image+predicted image at every step)
    # todo: use a gaussian pyramid and only include the "high-detail" level of the pyramid into the next step

    prompt = ImaginePrompt(
        description,
        init_image=img,
        init_image_strength=0.8,
        width=512,
        height=512,
        steps=50,
        seed=1,
        sampler_type="DDIM",
    )
    out_folder = f"{TESTS_FOLDER}/test_output"
    imagine_image_files(prompt, outdir=out_folder)
