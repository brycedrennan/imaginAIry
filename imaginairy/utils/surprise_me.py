"""Functions for generating surprise image edits"""

import logging
import os.path

from imaginairy.api import imagine_image_files
from imaginairy.enhancers.facecrop import detect_faces
from imaginairy.schema import ControlInput, ImaginePrompt, LazyLoadingImage
from imaginairy.utils.animations import make_gif_animation
from imaginairy.utils.img_utils import add_caption_to_image, pillow_fit_image_within

logger = logging.getLogger(__name__)

preserve_head_kwargs = {
    "mask_prompt": "head|face",
    "mask_mode": "keep",
}

preserve_face_kwargs = {
    "mask_prompt": "face",
    "mask_mode": "keep",
}

generic_prompts = [
    ("add confetti", 15, {}),
    # ("add sparkles", 14, {}),
    ("make it christmas", 15, preserve_face_kwargs),
    ("make it halloween", 15, preserve_face_kwargs),
    ("give it a depressing vibe", 10, {}),
    ("give it a bright cheery vibe", 10, {}),
    # weather
    ("make it look like a snowstorm", 15, preserve_face_kwargs),
    ("make it sunset", 15, preserve_face_kwargs),
    ("make it a sunny day", 15, preserve_face_kwargs),
    ("add misty fog", 10, {}),
    ("make it flooded", 10, {}),
    # setting
    ("make it underwater", 10, {}),
    ("add fireworks to the sky", 15, {}),
    # ("make it in a forest", 10, {}),
    # ("make it grassy", 11, {}),
    ("make it on mars", 14, {}),
    # style
    ("add glitter", 10, {}),
    ("turn it into a still from a western", 10, {}),
    ("old 1900s photo", 11.5, {}),
    ("Daguerreotype", 14, {}),
    ("make it anime style", 15, {}),
    ("watercolor painting", 10, {}),
    ("crayon drawing", 10, {}),
    # ("make it pen and ink style", 20, {}),
    ("graphite pencil", 10, {"negative_prompt": "low quality"}),
    ("make it a thomas kinkade painting", 10, {}),
    ("make it pixar style", 18, {}),
    ("low-poly", 20, {}),
    ("make it stained glass", 15, {}),
    ("make it pop art", 15, {}),
    ("oil painting", 11, {}),
    ("street graffiti", 10, {}),
    ("photorealistic", 8, {}),
    ("vector art", 8, {}),
    ("comic book style. happy", 9, {}),
    ("starry night painting", 15, {}),
    ("make it minecraft", 12, {}),
    # materials
    ("make it look like a marble statue", 15, {}),
    ("marble statue", 15, {}),
    ("make it look like a golden statue", 15, {}),
    ("golden statue", 15, {}),
    # ("make it claymation", 15, {}),
    ("play-doh", 15, {}),
    ("voxel", 15, {}),
    # ("lego", 15, {}),
    ("ice sculpture", 15, {}),
    ("make it look like a minature toy", 10, {}),
    # ("made out of colorful smoke", 15, {}),
    # photo effect
    # ("sepia", 13, {}),
    # ("add a blur effect", 15, {}),
    # ("add dramatic lighting", 12, {}), # too boring
    # add things
    # ("turn this into the space age", 12, {}),
    # ("make it in a grocery store", 15, {}),
]

only_face_kwargs = {
    "mask_prompt": "face",
    "mask_mode": "replace",
}

person_prompt_configs = [
    # face
    ("make the person close their eyes", 7, only_face_kwargs),
    (
        "make the person wear intricate highly detailed facepaint. ornate, artistic",
        6,
        only_face_kwargs,
    ),
    # ("make the person wear makeup. professional photoshoot", 15, only_face_kwargs),
    # ("make the person wear mime makeup. intricate, artistic", 7, only_face_kwargs),
    ("make the person wear clown makeup. intricate, artistic", 7, only_face_kwargs),
    ("make the person a cyborg", 14, {}),
    # clothes
    ("make the person wear shiny metal clothes", 14, preserve_face_kwargs),
    ("make the person wear a tie-dye shirt", 14, preserve_face_kwargs),
    ("make the person wear a suit", 14, preserve_face_kwargs),
    ("make the person bald", 15, {}),
    (
        "change the hair to pink",
        7.5,
        {"mask_mode": "keep", "mask_prompt": "face"},
    ),
    # ("change the hair to black", 7.5, {"mask_mode": "replace", "mask_prompt": "hair"}),
    # ("change the hair to blonde", 7.5, {"mask_mode": "replace", "mask_prompt": "hair"}),
    # ("change the hair to red", 7.5, {"mask_mode": "replace", "mask_prompt": "hair"}),
    # (
    #     "change the hair to rainbow",
    #     7.5,
    #     {"mask_mode": "replace", "mask_prompt": "hair"},
    # ),
    # ("change the hair to silver", 7.5, {"mask_mode": "replace", "mask_prompt": "hair"}),
    (
        "professional corporate photo headshot. Canon EOS, sharp focus, high resolution",
        6,
        {"negative_prompt": "low quality"},
    ),
    ("make the person stoic. pensive", 7, only_face_kwargs),
    ("make the person sad", 7, only_face_kwargs),
    ("make the person angry", 7, only_face_kwargs),
    ("make the person look like a celebrity", 10, {}),
    ("make the person younger", 7, {}),
    ("make the person 70 years old", 10, {}),
    ("make the person a disney cartoon character", 9, {}),
    ("turn the humans into robots", 13, {}),
    ("make the person a jedi knight. star wars", 15, preserve_head_kwargs),
    ("make the person a starfleet officer. star trek", 15, preserve_head_kwargs),
    ("make the person a superhero", 15, preserve_head_kwargs),
    # ("a tiger", 15, only_face_kwargs),
    ("lego minifig", 15, {}),
]


def surprise_me_prompts(
    img, person=None, width=None, height=None, steps=30, seed=None, use_controlnet=True
):
    if isinstance(img, str):
        if img.startswith("http"):
            img = LazyLoadingImage(url=img)
        else:
            img = LazyLoadingImage(filepath=img)

    if person is None:
        person = bool(detect_faces(img))
    prompts = []
    logger.info("Person detected in photo. Adjusting edits accordingly.")
    init_image_strength = 0.3
    for prompt_text, strength, kwargs in generic_prompts:
        kwargs.setdefault("negative_prompt", None)
        kwargs.setdefault("init_image_strength", init_image_strength)
        if use_controlnet:
            control_input = ControlInput(mode="edit")
            prompts.append(
                ImaginePrompt(
                    prompt_text,
                    init_image=img,
                    prompt_strength=strength,
                    control_inputs=[control_input],
                    steps=steps,
                    size=(width, height),
                    **kwargs,
                )
            )
        else:
            prompts.append(
                ImaginePrompt(
                    prompt_text,
                    init_image=img,
                    prompt_strength=strength,
                    model_weights="edit",
                    steps=steps,
                    size=(width, height),
                    **kwargs,
                )
            )

    if person:
        for prompt_subconfigs in person_prompt_configs:
            if isinstance(prompt_subconfigs, tuple):
                prompt_subconfigs = [prompt_subconfigs]
            for prompt_subconfig in prompt_subconfigs:
                prompt_text, strength, kwargs = prompt_subconfig
                if use_controlnet:
                    control_input = ControlInput(
                        mode="edit",
                    )
                    kwargs.setdefault("negative_prompt", None)
                    kwargs.setdefault("init_image_strength", init_image_strength)
                    prompts.append(
                        ImaginePrompt(
                            prompt_text,
                            init_image=img,
                            prompt_strength=strength,
                            control_inputs=[control_input],
                            steps=steps,
                            size=(width, height),
                            seed=seed,
                            **kwargs,
                        )
                    )
                else:
                    prompts.append(
                        ImaginePrompt(
                            prompt_text,
                            init_image=img,
                            prompt_strength=strength,
                            model_weights="edit",
                            steps=steps,
                            size=(width, height),
                            seed=seed,
                            **kwargs,
                        )
                    )

    return prompts


def create_surprise_me_images(
    img, outdir, person=None, make_gif=True, width=None, height=None, seed=None
):
    if isinstance(img, str):
        if img.startswith("http"):
            img = LazyLoadingImage(url=img)
        else:
            img = LazyLoadingImage(filepath=img)

    prompts = surprise_me_prompts(
        img, person=person, width=width, height=height, seed=seed
    )

    generated_filenames = imagine_image_files(
        prompts,
        outdir=outdir,
        record_step_images=False,
        output_file_extension="jpg",
        print_caption=False,
        make_gif=make_gif,
    )
    if make_gif:
        imgs_path = os.path.join(outdir, "compilations")
        os.makedirs(imgs_path, exist_ok=True)
        base_count = len(os.listdir(imgs_path))
        new_filename = os.path.join(imgs_path, f"{base_count:04d}_surprise_me.gif")
        simg = pillow_fit_image_within(img, prompts[0].width, prompts[0].height)
        gif_imgs = [simg]
        for prompt, filename in zip(prompts, generated_filenames):
            gen_img = LazyLoadingImage(filepath=filename)
            add_caption_to_image(gen_img, prompt.prompt_text)

            gif_imgs.append(gen_img)

        make_gif_animation(outpath=new_filename, imgs=gif_imgs, frame_duration_ms=1000)


if __name__ == "__main__":
    for row in generic_prompts:
        print(" -" + row[0])
