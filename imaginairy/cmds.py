#!/usr/bin/env python

import click

from imaginairy.imagine import ImaginePrompt, imagine as imagine_f


@click.command()
@click.argument(
    "prompt_texts", default=None, help="text to render to an image", nargs=-1
)
@click.option("--outdir", default="./outputs", help="where to write results to")
@click.option("-r", "--repeats", default=1, type=int, help="How many times to repeat the renders")
@click.option(
    "-h",
    "--height",
    default=512,
    type=int,
    help="image height. should be multiple of 64",
)
@click.option(
    "-w", "--width", default=512, type=int, help="image width. should be multiple of 64"
)
@click.option(
    "--steps",
    default=50,
    type=int,
    help="How many diffusion steps to run. More steps, more detail, but with diminishing returns",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="What seed to use for randomness. Allows reproducible image renders",
)
@click.option(
    "--prompt-strength",
    default=7.5,
    help="How closely to follow the prompt. Image looks unnatural at higher values",
)
@click.option("--sampler-type", default="PLMS", help="What sampling strategy to use")
@click.option("--ddim-eta", default=0.0, type=float)
def imagine_cmd(
    prompt_texts,
    outdir,
    repeats,
    height,
    width,
    steps,
    seed,
    prompt_strength,
    sampler_type,
    ddim_eta,
):
    prompts = []
    for _ in range(repeats):
        for prompt_text in prompt_texts:
            prompt = ImaginePrompt(
                prompt_text,
                seed=seed,
                sampler_type=sampler_type,
                steps=steps,
                height=height,
                width=width,
                prompt_strength=prompt_strength,
                upscale=True,
                fix_faces=True,
            )
            prompts.append(prompt)

    imagine_f(
        prompts,
        outdir=outdir,
        ddim_eta=ddim_eta,
    )


if __name__ == "__main__":
    imagine_cmd()
