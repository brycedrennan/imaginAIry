"""


aimg.
"""

import os.path

from PIL import ImageDraw, ImageFont

from imaginairy import ImaginePrompt, LazyLoadingImage, imagine_image_files
from imaginairy.enhancers.facecrop import detect_faces
from imaginairy.img_utils import make_gif_image, pillow_fit_image_within
from imaginairy.paths import PKG_ROOT

generic_prompts = [
    ("make it anime style", 18, ""),
    ("make it pen and ink style", 20, ""),
    ("make it a thomas kinkade painting", 20, ""),
    ("make it pixar style", 20, ""),
    ("make it look like a marble statue", 15, ""),
    ("make it look like a golden statue", 15, ""),
    ("make it look like a snowstorm", 20, ""),
    ("make it night", 15, ""),
    ("make it a sunny day", 15, ""),
    ("put fireworks in the background", 15, ""),
    ("make it in a forest", 15, ""),
]

person_prompts = [
    ("make the person close their eyes", 10, ""),
    ("make the person wear clown makeup", 10, ""),
    ("make the person a cyborg", 14, ""),
    ("make the person wear shiny metal clothes", 14, ""),
    ("make the person wear a tie-dye shirt", 7.5, ""),
    ("make the person wear a suit", 7.5, ""),
    ("make the person bald", 7.5, ""),
    ("change the hair to pink", 7.5, ""),
    (
        "make it a color professional photo headshot. Canon EOS, sharp focus",
        10,
        "old, ugly",
    ),
    ("make the face smiling", 5, ""),
    # ("make the person sad", 20, ""),
    # ("make the person angry", 20, ""),
    ("make the person look like a celebrity", 10, ""),
    ("make the person younger", 10, ""),
    ("make the person older", 6, ""),
    ("make the person a disney cartoon character", 7.5, ""),
]


def surprise_me_prompts(img, person=None):
    prompts = []
    if isinstance(img, str):
        if img.startswith("http"):
            img = LazyLoadingImage(url=img)
        else:
            img = LazyLoadingImage(filepath=img)

    if person is None:
        person = bool(detect_faces(img))

    if person:
        for prompt_text, strength, neg_prompt_text in person_prompts:
            prompts.append(
                ImaginePrompt(
                    prompt_text,
                    init_image=img,
                    prompt_strength=strength,
                    negative_prompt=neg_prompt_text,
                    model="edit",
                    steps=20,
                )
            )

    for prompt_text, strength, neg_prompt_text in generic_prompts:
        prompts.append(
            ImaginePrompt(
                prompt_text,
                init_image=img,
                prompt_strength=strength,
                negative_prompt=neg_prompt_text,
                model="edit",
                steps=20,
            )
        )

    return prompts


def create_surprise_me_images(img, outdir, person=None, make_gif=True):
    if isinstance(img, str):
        if img.startswith("http"):
            img = LazyLoadingImage(url=img)
        else:
            img = LazyLoadingImage(filepath=img)
    prompts = surprise_me_prompts(img, person=person)
    generated_filenames = imagine_image_files(
        prompts,
        outdir=outdir,
        record_step_images=False,
        output_file_extension="jpg",
        print_caption=False,
        make_comparison_gif=make_gif,
    )
    if make_gif:
        imgs_path = os.path.join(outdir, "compilations")
        os.makedirs(imgs_path, exist_ok=True)
        base_count = len(os.listdir(imgs_path))
        new_filename = os.path.join(imgs_path, f"surprise_me_{base_count:03d}.gif")
        simg = pillow_fit_image_within(img, prompts[0].width, prompts[0].height)
        gif_imgs = [simg]
        for prompt, filename in zip(prompts, generated_filenames):
            gen_img = LazyLoadingImage(filepath=filename)
            draw = ImageDraw.Draw(gen_img)

            font_size = 16
            font = ImageFont.truetype(f"{PKG_ROOT}/data/DejaVuSans.ttf", font_size)

            x = 15
            y = 485

            draw.text((x, y), prompt.prompt_text, font=font, fill=(255, 255, 255))
            gif_imgs.append(gen_img)

        make_gif_image(new_filename, gif_imgs)
