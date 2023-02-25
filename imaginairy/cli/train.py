import logging

import click

from imaginairy import config

logger = logging.getLogger(__name__)


@click.option(
    "--concept-label",
    help=(
        'The concept you are training on. Usually "a photo of [person or thing] [classname]" is what you should use.'
    ),
    required=True,
)
@click.option(
    "--concept-images-dir",
    type=click.Path(),
    required=True,
    help="Where to find the pre-processed concept images to train on.",
)
@click.option(
    "--class-label",
    help=(
        'What class of things does the concept belong to. For example, if you are training on "a painting of a George Washington", '
        'you might use "a painting of a man" as the class label. We use this to prevent the model from overfitting.'
    ),
    default="a photo of *",
)
@click.option(
    "--class-images-dir",
    type=click.Path(),
    required=True,
    help="Where to find the pre-processed class images to train on.",
)
@click.option(
    "--n-class-images",
    type=int,
    default=300,
    help="Number of class images to generate.",
)
@click.option(
    "--model-weights-path",
    "--model",
    "model",
    help=f"Model to use. Should be one of {', '.join(config.MODEL_SHORT_NAMES)}, or a path to custom weights.",
    show_default=True,
    default=config.DEFAULT_MODEL,
)
@click.option(
    "--person",
    "is_person",
    is_flag=True,
    help="Set if images are of a person. Will use face detection and enhancement.",
)
@click.option(
    "-y",
    "preconfirmed",
    is_flag=True,
    default=False,
    help="Bypass input confirmations.",
)
@click.option(
    "--skip-prep",
    is_flag=True,
    default=False,
    help="Skip the image preparation step.",
)
@click.option(
    "--skip-class-img-gen",
    is_flag=True,
    default=False,
    help="Skip the class image generation step.",
)
@click.command("train-concept")
def train_concept_cmd(
    concept_label,
    concept_images_dir,
    class_label,
    class_images_dir,
    n_class_images,
    model,
    is_person,
    preconfirmed,
    skip_prep,
    skip_class_img_gen,
):
    """
    Teach the model a new concept (a person, thing, style, etc).

    Provided a directory of concept images, a concept token, and a class token, this command will train the model
    to generate images of that concept.

    \b
    This happens in a 3-step process:
      1. Cropping and resizing your training images. If --person is set we crop to include the face.
      2. Generating a set of class images to train on.  This helps prevent overfitting.
      3. Training the model on the concept and class images.

    The output of this command is a new model weights file that you can use with the --model option.

    \b
    ## Instructions
     1. Gather a set of images of the concept you want to train on. They should show the subject from a variety of angles
     and in a variety of situations.
     2. Train the model.
     - Concept label: For a person, firstnamelastname should be fine.
        - If all the training images are photos you should add "a photo of" to the beginning of the concept label.
     - Class label: This is the category of the things beings trained on.  For people this is typically "person", "man"
     or "woman".
        - If all the training images are photos you should add "a photo of" to the beginning of the class label.
        - CLass images will be generated for you if you do not provide them.
    3. Stop training before it overfits. I haven't figured this out yet.


    For example, if you were training on photos of a man named bill hamilton you could run the following:

    \b
    aimg train-concept \\
        --person \\
        --concept-label "photo of billhamilton man" \\
        --concept-images-dir ./images/billhamilton \\
        --class-label "photo of a man" \\
        --class-images-dir ./images/man

    When you use the model you should prompt with `firstnamelastname classname` (e.g. `billhamilton man`).

    You can find a lot of relevant instructions here: https://github.com/JoePenna/Dreambooth-Stable-Diffusion
    """
    from imaginairy.utils import get_device

    if "mps" in get_device():
        click.secho(
            "⚠️  MPS (MacOS) is not supported for training. Please use a GPU or CPU.",
            fg="yellow",
        )
        return

    import os.path

    from imaginairy.train import train_diffusion_model
    from imaginairy.training_tools.image_prep import (
        create_class_images,
        get_image_filenames,
        prep_images,
    )

    target_size = 512
    # Step 1. Crop and enhance the training images
    prepped_images_path = os.path.join(concept_images_dir, "prepped-images")
    image_filenames = get_image_filenames(concept_images_dir)
    click.secho(
        f'\n🤖🧠 Training "{concept_label}" based on {len(image_filenames)} images.\n'
    )

    if not skip_prep:
        msg = (
            f"Creating cropped copies of the {len(image_filenames)} concept images\n"
            f"    Is Person: {is_person}\n"
            f"    Source: {concept_images_dir}\n"
            f"    Dest: {prepped_images_path}\n"
        )
        logger.info(msg)
        if not is_person:
            click.secho("⚠️  the --person flag was not set. ", fg="yellow")

        if not preconfirmed and not click.confirm("Continue?"):
            return

        prep_images(
            images_dir=concept_images_dir, is_person=is_person, target_size=target_size
        )
        concept_images_dir = prepped_images_path

    if not skip_class_img_gen:
        # Step 2. Generate class images
        class_image_filenames = get_image_filenames(class_images_dir)
        images_needed = max(n_class_images - len(class_image_filenames), 0)
        logger.info(f"Generating {n_class_images} class images in {class_images_dir}")
        logger.info(
            f"{len(class_image_filenames)} existing class images found so only generating {images_needed}."
        )
        if not preconfirmed and not click.confirm("Continue?"):
            return
        create_class_images(
            class_description=class_label,
            output_folder=class_images_dir,
            num_images=n_class_images,
        )

    logger.info("Training the model...")
    if not preconfirmed and not click.confirm("Continue?"):
        return

    # Step 3. Train the model
    train_diffusion_model(
        concept_label=concept_label,
        concept_images_dir=concept_images_dir,
        class_label=class_label,
        class_images_dir=class_images_dir,
        weights_location=model,
        logdir="logs",
        learning_rate=1e-6,
        accumulate_grad_batches=32,
    )


@click.argument("ckpt_paths", nargs=-1)
@click.command("prune-ckpt")
def prune_ckpt_cmd(ckpt_paths):
    """
    Prune a checkpoint file.

    This will remove the optimizer state from the checkpoint file.
    This is useful if you want to use the checkpoint file for inference and save a lot of disk space

    Example:
        aimg prune-ckpt ./path/to/checkpoint.ckpt
    """
    from imaginairy.training_tools.prune_model import prune_diffusion_ckpt

    click.secho("Pruning checkpoint files...")
    for p in ckpt_paths:
        prune_diffusion_ckpt(p)


@click.argument(
    "images_dir",
    required=True,
)
@click.option(
    "--person",
    "is_person",
    is_flag=True,
    help="Set if images are of a person. Will use face detection and enhancement.",
)
@click.option(
    "--target-size",
    default=512,
    type=int,
    show_default=True,
)
@click.command("prep-images")
def prep_images_cmd(images_dir, is_person, target_size):
    """
    Prepare a folder of images for training.

    Prepped images will be written to the `prepped-images` subfolder.

    All images will be cropped and resized to (default) 512x512.
    Upscaling and face enhancement will be applied as needed to smaller images.

    Examples:
        aimg prep-images --person ./images/selfies
        aimg prep-images ./images/toy-train
    """

    from imaginairy.training_tools.image_prep import prep_images

    prep_images(images_dir=images_dir, is_person=is_person, target_size=target_size)
