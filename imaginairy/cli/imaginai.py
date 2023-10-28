import json
import logging
import os
import readline

import click

from imaginairy.cli.imagine import imagine_cmd

logger = logging.getLogger(__name__)


@click.command(name="imaginai")
@click.argument("prompt", nargs=-1)
def imaginai_cmd(prompt):
    """
    Use chatgpt-4-preview to assist with Imaginairy prompts.

    Can be invoked via either `aimg imaginai` or just `imaginai`.
    """
    click.echo("🎩 Chatgpt generating imaginAIry prompt.")

    cli_command = generate_cli_cmd(prompt[0])

    if not cli_command:
        click.echo("⛔ Command generation failed.")
    else:

        def pre_input_hook():
            readline.insert_text(cli_command)
            readline.set_startup_hook(None)

        readline.set_startup_hook(pre_input_hook)

        click.echo("🔮 Command generation complete.")


def generate_cli_cmd(user_prompt: str):
    import openai

    properties = generate_properties()

    functions = [
        {
            "name": "generate_image",
            "description": imagine_cmd.__doc__,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["prompt_texts"],
            },
        }
    ]

    openai.api_key = os.environ.get("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": f"{user_prompt + add_prompt_examples()}"}
        ],
        functions=functions,
        temperature=0,
        function_call="auto",
    )

    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        function_args = json.loads(response_message["function_call"]["arguments"])

        cli_command = convert_to_cli_command(format_keys(function_args))

        return cli_command
    else:
        return None


def format_keys(function_args: dict):
    """
    Takes a dictionary of parameters and formats the keys to be used in the CLI command.
    """
    keys = list(function_args.keys())  # Make a copy of the dictionary
    for key in keys:
        function_args[f"--{key}"] = function_args.pop(key)
    return function_args


def convert_to_cli_command(parameters: dict):
    prompt = parameters["--prompt_texts"].lower()

    if "colorize" in prompt[:8]:
        cli_command = parameters["--prompt_texts"]
    elif "edit" in prompt[:4]:
        cli_command = prompt + " "
    elif "upscale" in prompt[:7]:
        cli_command = prompt
    else:
        cli_command = "imagine " + '"' + prompt + '"' + " "

        if "mask-prompt" in parameters:
            cli_command = cli_command.lower()

        for key, value in parameters.items():
            if key == "--prompt_texts":
                continue
            if key in ["--fix-faces", "--tile", "--tile-x", "--tile-y", "--upscale"]:
                cli_command += key + " "
            else:
                cli_command += key + " " + value + " "

    return cli_command


def list_option_descriptions(command):
    parameter_descriptions = {}

    for param in command.params:
        if isinstance(param, click.Option):
            option_names = ", ".join(param.opts)
            help_str = param.help
            parameter_descriptions[option_names] = help_str
    return parameter_descriptions


def improve_imagine_descriptions(parameter_descriptions):
    """
    Takes imagine parameter descriptions from --help,
    replaces them with custom descriptions, and returns the result.
    """

    parameter_descriptions["--upscale"] = "scale the image to be larger"
    parameter_descriptions[
        "--fix-faces"
    ] = "improves the quality of faces, fixes distorted faces caused by image generation"
    parameter_descriptions[
        "--init-image"
    ] = "the image to start with, if not specified, ImaginAIry will generate an image from scratch. Don't use this for anything that is related to the control image, use --control-image instead."
    parameter_descriptions[
        "--control-image"
    ] = "image used for control signal in image generation, use this instead of --init-image if you want to use a control image"

    return parameter_descriptions


def remove_unnecessary_parameters(parameters: dict):
    """
    Takes a dictionary of parameters and removes any that are not necessary for the current prompt.
    """
    keys_to_remove = [
        "--help",
        "--version",
        "--outdir",
        "--output-file-extension",
        "--seed",
        "--steps",
        "--repeats",
        "--width",
        "--height",
        "--log-level",
        "--model-weights-path",
        "--quiet",
        "--show-work",
        "--sampler",
        "--quiet",
        "--allow-compose-phase",
        "--precision",
        "--model-config-path",
        "--prompt-library-path",
        "--compare-gif",
        "--arg-schedule",
        "--control-image-raw",
        "--model",
    ]

    for key in list(parameters.keys()):
        if key in keys_to_remove:
            parameters.pop(key, None)

    return parameters


def generate_properties():
    properties = {
        "prompt_texts": {
            "type": "string",
            "description": "The description of the image to either generate or modify an existing image.",
        }
    }

    parameter_descriptions = remove_unnecessary_parameters(
        list_option_descriptions(imagine_cmd)
    )
    parameter_descriptions = improve_imagine_descriptions(parameter_descriptions)

    for key, value in parameter_descriptions.items():
        properties[key] = {"type": "string", "description": value}

    return properties


def add_prompt_examples():
    examples = """
## OpenPose Control Mode:
Command Example: imagine --control-image assets/indiana.jpg --control-mode openpose "photo of a polar bear"
User Prompt Example: Use the pose of the person in this image to make a new image with a polar bear in the same pose assets/indiana.jpg.
Explanation: OpenPose is a system for detecting human body, hand, facial, and foot keypoints. In this command,
the structure of the polar bear image generated will be guided by the human pose detected in indiana.jpg.

## Canny Edge Detection Control Mode:
Command Example: imagine --control-image assets/lena.png --control-mode canny "photo of a woman with a hat looking at the camera"
User Prompt Example: I want a variation of this image of a woman using Canny Edge Control assets/lena.png.
Explanation: Canny edge detection is a technique used to identify the edges in an image. The command uses lena.png to
guide the edge layout in the generated image.

## HED (Holistic Edge Detection) Control Mode:
Command Example: imagine --control-image dog.jpg --control-mode hed "photo of a dalmatian"
User Prompt Example: Using HED Boundary Control, I want a variation of this image that is of a dalmatian dog.jpg.
Explanation: HED is an advanced method for detecting edges in images. It's used here with dog.jpg to influence the
edge structure in the generated dalmatian image.

## Depth Map Control Mode:
Command Example: imagine --control-image fancy-living.jpg --control-mode depth "a modern living room"
User Prompt Example: Make some variations of this photo of my living room, using Depth Map Control fancy-living.jpg.
Explanation: A depth map shows the distance of surfaces from a viewpoint. This command uses the depth map of
fancy-living.jpg to guide the depth perception in the generated image of a modern living room.

## Normal Map Control Mode:
Command Example: imagine --control-image bird.jpg --control-mode normal "a bird"
User Prompt Example: Use Normal Map Control to create a variation of my photo of a bird bird.jpg.
Explanation: Normal mapping is a technique used in 3D graphics for faking the lighting of bumps and dents.
This command uses the normal map from bird.jpg to guide the texture and lighting effects in the generated image of a bird.

## Image Shuffle Control Mode:
Command Example: imagine --control-image pearl-girl.jpg --control-mode shuffle "a clown"
User Prompt Example: Use Image Shuffle Control to generate a clown of this girl in my image pearl-girl.jpg.
Explanation: This mode generates an image by incorporating elements from the control image, similar to style transfer.
It shuffles visual components from the control image to create a new image.

## Editing Instructions Control Mode:
Command Example: imagine --control-image pearl-girl.jpg --control-mode edit --init-image-strength 0.01 --steps 30 "make it anime"
User Prompt Example: Make  an anime version of this girl in this image pearl-girl.jpg using Editing Instructions Control.
Explanation: This mode allows for editing images based on verbal instructions. It can make various thematic or stylistic
changes to the image. IT MUST USE "--init-image-strength"(if not specified set to 0.01) and "--control-image" parameter, it DOES NOT USE "--init-image".

## Add Details Control Mode (Upscaling/Super-Resolution):
Command Example: imagine --control-image "assets/wishbone.jpg" --control-mode details "sharp focus, high-resolution" --init-image-strength 0.2 --steps 30 -w 2048 -h 2048
User Prompt Example: Make this image have higher resolution assets/wishbone.jpg using Add Details Control.
Explanation: This mode enhances the details in an image, replacing existing elements with sharper, higher-resolution
versions. It's useful for upscaling images.

## Image (Re)Colorization Control Mode (Using Brightness Control):
Command Example: colorize pearl-girl.jpg
User Prompt Example: Make this black and white image have color pearl-girl.jpg.
Explanation: This mode is used to colorize black and white images or re-color existing images. The generated colors are
applied back to the original image based on a auto-generated one. This COMMAND SHOULD ONLY USE "colorize", and the
prompt should be the file location.

## Instruction Based Image Edits by InstructPix2Pix:
Command Examples:
edit scenic_landscape.jpg -p "make it winter" --prompt-strength 20
edit dog.jpg -p "make the dog red" --prompt-strength 5
edit bowl_of_fruit.jpg -p "replace the fruit with strawberries"
edit freckled_woman.jpg -p "make her a cyborg" --prompt-strength 13
edit bluebird.jpg -p "make the bird wear a cowboy hat" --prompt-strength 10
edit flower.jpg -p "make the flower out of paper origami" --arg-schedule prompt-strength[1:11:0.3] --steps 25 --compilation-anim gif
Explanation: YOU MUST INCLUDE THE WORD "Edit" or "edit" in the prompt description as the first word. This mode allows
users to verbally instruct the tool to make specific edits to an image. The prompt strength parameter controls how
significantly the image is altered.

## Prompt Based Masking by clipseg:
Command Examples:
Command Example: imagine --init-image pearl_earring.jpg --mask-prompt "face AND NOT (bandana OR hair OR blue fabric){*6}" --mask-mode keep --init-image-strength .2 --fix-faces "a female robot"
User Prompt Example: Take this photo of girl pearl_earring.jpg and generate a photos her as a robot while keeping the face and removing the bandana, hair, and blue fabric.
Command Example: imagine --init-image fruit-bowl.jpg --mask-prompt "fruit OR fruit stem{*6}" --mask-mode replace --mask-modify-original --init-image-strength .1 "a bowl of kittens"
User Prompt Example: Take this photo of a fruit bowl fruit-bowl.jpg and generate a photo of a bowl of kittens while keeping the fruit and fruit stems (stems of the fruit, not the stem of the bowl) and replacing the rest of the image with the new image of kittens.
Explanation: Masking allows specific features of an image remain the same while altering everything else or replacing specific features while keeping everything else the same.
MASK DESCRIPTIONS(aka prompt) MUST BE LOWERCASE. MAKE THE PROMPT ONLY ABOUT WHAT'S BEING GENERATED. keywords (AND, OR, NOT) must be uppercase. parentheses are supported.
mask modifiers may be appended to any mask or group of masks. Example: (dog OR cat){+5} means that we'll select any dog or cat and then expand the size of the mask area by 5 pixels. Valid mask modifiers:
{+n} - expand mask by n pixels
{-n} - shrink mask by n pixels
{*n} - multiply mask strength. will expand mask to areas that weakly matched the mask description
{/n} - divide mask strength. will reduce mask to areas that most strongly matched the mask description. probably not useful
When writing strength modifiers keep in mind that pixel values are between 0 and 1

## Face Enhancement by CodeFormer:
Command Example: imagine "a couple smiling" --steps 40 --seed 1 --fix-faces
User Prompt Example: Generate an image of a couple smiling and fix the faces so they look nice.
Explanation: This mode enhances the facial features in an image, making them more pronounced and refined.
IT SHOULD BE USED IN ANY IMAGE GENERATION THAT INVOLVES FACES.

## Upscaling by RealESRGAN:
Command Example: upscale my-image.jpg
User Prompt Example: Please upscale this image my-image.jpg.
Explanation: START THE PROMPT WITH THE WORD "upscale" RealESRGAN is used for upscaling images, increasing their
resolution while maintaining or enhancing detail and clarity. DO NOT USE "--init-image" or "control-image" parameters.

## Tiled Images:
Command Example: imagine "gold coins" "a lush forest" "piles of old books" leaves --tile
User Prompt Example: Please generate several tile images of gold coins, a lush forest, old books, and leaves.
Explanation: This mode generates tiled images, creating repeating patterns or sequences of images.

## 360 Degree Images:
Command Example: imagine --tile-x -w 1024 -h 512 "360 degree equirectangular panorama photograph of the desert" --upscale
User Prompt Example: The command would generate a 360-degree panoramic image of a desert landscape.
Explanation: 360 degree equirectangular panorama photograph of the desert.

## Image-to-Image with Depth Maps:
Command Example: imagine --init-image girl_with_a_pearl_earring_large.jpg --init-image-strength 0.05 "professional
headshot photo of a woman with a pearl earring" -r 4 -w 1024 -h 1024 --steps 50
User Prompt Example: I'd like to change the style of this portrait(a woman with pearl earrings),
make it look professional, but the original image should remain mostly intact.
Explanation: This mode uses depth maps to translate existing images into different contexts or
styles while maintaining depth perception.

## Outpainting:
Command Example: imagine --init-image pearl-earring.jpg --init-image-strength 0 --outpaint all250,up0,down600 "woman standing"
User Prompt Example: I want to expand what this photo shows. A lot downward. A little bit to the sides, but not at all upward pearl-earring.jpg.
Explanation: Outpainting extends the borders of an image, generating the surroundings or context beyond the original frame.
"""
    return examples


if __name__ == "__main__":
    imaginai_cmd()
