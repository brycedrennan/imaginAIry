"""


aimg.
"""

import os.path

from imaginairy import ImaginePrompt, LazyLoadingImage, imagine_image_files
from imaginairy.animations import make_gif_animation
from imaginairy.enhancers.facecrop import detect_faces
from imaginairy.img_utils import add_caption_to_image, pillow_fit_image_within
from imaginairy.schema import ControlNetInput

preserve_head_kwargs = {
    "mask_prompt": "head|face",
    "mask_mode": "keep",
}

generic_prompts = [
    ("add confetti", 6, {}),
    # ("add sparkles", 14, {}),
    ("make it christmas", 15, preserve_head_kwargs),
    ("make it halloween", 15, {}),
    ("give it a dark omninous vibe", 15, {}),
    ("give it a bright cheery vibe", 15, {}),
    # weather
    ("make it look like a snowstorm", 20, {}),
    ("make it midnight", 15, {}),
    ("make it a sunny day", 15, {}),
    ("add misty fog", 15, {}),
    ("make it flooded", 10, {}),
    # setting
    ("make it underwater", 15, {}),
    ("add fireworks to the sky", 15, {}),
    # ("make it in a forest", 10, {}),
    # ("make it grassy", 11, {}),
    ("make it on mars", 14, {}),
    # style
    ("add glitter", 10, {}),
    ("turn it into a still from a western", 15, {}),
    ("old 1900s photo", 11.5, {}),
    ("Daguerreotype", 12, {}),
    ("make it anime style", 18, {}),
    # ("make it pen and ink style", 20, {}),
    ("graphite pencil", 15, {}),
    # ("make it a thomas kinkade painting", 20, {}),
    ("make it pixar style", 20, {}),
    ("low-poly", 20, {}),
    ("make it stained glass", 10, {}),
    ("make it pop art", 12, {}),
    # ("make it street graffiti", 15, {}),
    ("vector art", 8, {}),
    ("comic book style. happy", 9, {}),
    ("starry night painting", 15, {}),
    ("make it minecraft", 12, {}),
    # materials
    ("make it look like a marble statue", 15, {}),
    ("make it look like a golden statue", 15, {}),
    # ("make it claymation", 8, {}),
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
    ("make the person close their eyes", 10, only_face_kwargs),
    # (
    #     "make the person wear intricate highly detailed facepaint. ornate, artistic",
    #     9,
    #     only_face_kwargs,
    # ),
    # ("make the person wear makeup. professional photoshoot", 8, only_face_kwargs),
    # ("make the person wear mime makeup. intricate, artistic", 7, only_face_kwargs),
    # ("make the person wear clown makeup. intricate, artistic", 6, only_face_kwargs),
    ("make the person a cyborg", 14, {}),
    # clothes
    ("make the person wear shiny metal clothes", 14, preserve_head_kwargs),
    ("make the person wear a tie-dye shirt", 7.5, preserve_head_kwargs),
    ("make the person wear a suit", 7.5, preserve_head_kwargs),
    # ("make the person bald", 7.5, {}),
    ("change the hair to pink", 7.5, {"mask_mode": "replace", "mask_prompt": "hair"}),
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
        10,
        {"negative_prompt": "old, ugly"},
    ),
    # ("make the person stoic. pensive", 10, only_face_kwargs),
    # ("make the person sad", 20, {}),
    # ("make the person angry", 20, {}),
    # ("make the person look like a celebrity", 10, {}),
    ("make the person younger", 11, {}),
    ("make the person 70 years old", 9, {}),
    ("make the person a disney cartoon character", 7.5, {}),
    ("turn the humans into robots", 13, {}),
    ("make the person darth vader", 15, {}),
    ("make the person a starfleet officer", 15, preserve_head_kwargs),
    ("make the person a superhero", 15, {}),
    ("make the person a tiger", 15, only_face_kwargs),
    # ("lego minifig", 15, {}),
]


def surprise_me_prompts(
    img, person=None, width=None, height=None, steps=30, seed=None, use_controlnet=False
):
    if isinstance(img, str):
        if img.startswith("http"):
            img = LazyLoadingImage(url=img)
        else:
            img = LazyLoadingImage(filepath=img)

    if person is None:
        person = bool(detect_faces(img))
    prompts = []

    for prompt_text, strength, kwargs in generic_prompts:
        if use_controlnet:
            control_input = ControlNetInput(
                mode="edit",
            )
            prompts.append(
                ImaginePrompt(
                    prompt_text,
                    init_image=img,
                    init_image_strength=0.05,
                    prompt_strength=strength,
                    control_inputs=[control_input],
                    steps=steps,
                    width=width,
                    height=height,
                    **kwargs,
                )
            )
        else:
            prompts.append(
                ImaginePrompt(
                    prompt_text,
                    init_image=img,
                    prompt_strength=strength,
                    model="edit",
                    steps=steps,
                    width=width,
                    height=height,
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
                    control_input = ControlNetInput(
                        mode="edit",
                    )
                    prompts.append(
                        ImaginePrompt(
                            prompt_text,
                            init_image=img,
                            init_image_strength=0.05,
                            prompt_strength=strength,
                            control_inputs=[control_input],
                            steps=steps,
                            width=width,
                            height=height,
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
                            model="edit",
                            steps=steps,
                            width=width,
                            height=height,
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

        make_gif_animation(outpath=new_filename, imgs=gif_imgs)


if __name__ == "__main__":
    for row in generic_prompts:
        print(" -" + row[0])
