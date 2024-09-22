import logging
import os
from functools import lru_cache

from imaginairy.schema import ImaginePrompt, ImagineResult
from imaginairy.utils.log_utils import ImageLoggingContext

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_flux_models():
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
    from optimum.quanto import freeze, qfloat8, quantize
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
        T5EncoderModel,
        T5TokenizerFast,
    )

    from imaginairy.utils.downloads import get_cache_dir

    dtype = torch.bfloat16
    bfl_repo = "black-forest-labs/FLUX.1-schnell"
    revision = "refs/pr/1"
    quant_type = "qfloat8"  # Define the quantization type

    # Define paths for saved quantized models
    quantized_dir = os.path.join(get_cache_dir(), "quantized_flux_models")
    os.makedirs(quantized_dir, exist_ok=True)
    transformer_path = os.path.join(
        quantized_dir, f"quantized_transformer_{quant_type}.pt"
    )
    text_encoder_2_path = os.path.join(
        quantized_dir, f"quantized_text_encoder_2_{quant_type}.pt"
    )

    # Load and set up models
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        bfl_repo, subfolder="scheduler", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=dtype
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision
    )

    # Load or create quantized models
    if os.path.exists(transformer_path):
        transformer = torch.load(transformer_path)
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision
        )
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        torch.save(transformer, transformer_path)

    if os.path.exists(text_encoder_2_path):
        text_encoder_2 = torch.load(text_encoder_2_path)
    else:
        text_encoder_2 = T5EncoderModel.from_pretrained(
            bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision
        )
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        torch.save(text_encoder_2, text_encoder_2_path)

    return (
        scheduler,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        vae,
        transformer,
    )


def generate_single_image(
    prompt: ImaginePrompt,
    debug_img_callback=None,
    progress_img_callback=None,
    progress_img_interval_steps=3,
    progress_img_interval_min_s=0.1,
    add_caption=False,
    return_latent=False,
    dtype=None,
    logging_context: ImageLoggingContext | None = None,
    output_perf=False,
    image_name="",
):
    import torch
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

    from imaginairy.api.generate import IMAGINAIRY_SAFETY_MODE
    from imaginairy.enhancers.upscale import upscale_image
    from imaginairy.utils import clear_gpu_cache, seed_everything
    from imaginairy.utils.log_utils import ImageLoggingContext
    from imaginairy.utils.safety import create_safety_score

    # Initialize logging context
    if not logging_context:

        def latent_logger(latents):
            progress_latents.append(latents)

        lc = ImageLoggingContext(
            prompt=prompt,
            debug_img_callback=debug_img_callback,
            progress_img_callback=progress_img_callback,
            progress_img_interval_steps=progress_img_interval_steps,
            progress_img_interval_min_s=progress_img_interval_min_s,
            progress_latent_callback=latent_logger
            if prompt.collect_progress_latents
            else None,
        )
    else:
        lc = logging_context

    with lc:
        # Seed for reproducibility
        seed_everything(prompt.seed)
        clear_gpu_cache()

        # Load models
        with lc.timing("model-load"):
            (
                scheduler,
                text_encoder,
                tokenizer,
                text_encoder_2,
                tokenizer_2,
                vae,
                transformer,
            ) = load_flux_models()

        # Set up pipeline
        pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=transformer,
        )
        pipe.enable_model_cpu_offload()

        assert prompt.seed is not None
        generator = torch.Generator().manual_seed(prompt.seed)
        with lc.timing("image-generation"):
            output = pipe(
                prompt=prompt.prompt_text,
                width=prompt.width,
                height=prompt.height,
                num_inference_steps=prompt.steps,
                guidance_scale=prompt.prompt_strength,
                generator=generator,
            )
            image = output.images[0]

        # Perform safety check
        with lc.timing("safety-filter"):
            safety_score = create_safety_score(
                image,
                safety_mode=IMAGINAIRY_SAFETY_MODE,
            )
            is_filtered = safety_score.is_filtered

        # Initialize result images
        result_images = {}
        progress_latents: list[torch.Tensor] = []

        # If the image is unsafe, we can discard it or handle it accordingly
        if is_filtered:
            image = None  # Discard the unsafe image
        else:
            result_images["generated"] = image

            # Optionally upscale the image
            if prompt.upscale:
                with lc.timing("upscaling"):
                    upscaled_img = upscale_image(image)
                    result_images["upscaled"] = upscaled_img
                final_image = upscaled_img
            else:
                final_image = image

            if add_caption:
                with lc.timing("caption-img"):
                    from imaginairy.enhancers.describe_image_blip import (
                        generate_caption,
                    )

                    caption = generate_caption(final_image)
                    logger.info(f"Generated caption: {caption}")

            if prompt.fix_faces:
                with lc.timing("face-enhancement"):
                    from imaginairy.enhancers.face_restoration_codeformer import (
                        enhance_faces,
                    )

                    logger.info("Fixing 😊 's in 🖼  using CodeFormer...")
                    final_image = enhance_faces(
                        final_image, fidelity=prompt.fix_faces_fidelity
                    )
                    result_images["face_enhanced"] = final_image

        # Create ImagineResult
        result = ImagineResult(
            img=final_image,
            prompt=prompt,
            is_nsfw=safety_score.is_nsfw if safety_score else False,
            safety_score=safety_score,
            result_images=result_images,
            performance_stats=lc.get_performance_stats(),
            progress_latents=progress_latents,
        )

        _image_name = f"{image_name} " if image_name else ""
        logger.info(f"Generated {_image_name}image in {result.total_time():.1f}s")

        if result.performance_stats:
            log = logger.info if output_perf else logger.debug
            log(f"   Timings: {result.timings_str()}")
            if torch.cuda.is_available():
                log(f"   Peak VRAM: {result.gpu_str('memory_peak')}")
                log(f"   Peak VRAM Delta: {result.gpu_str('memory_peak_delta')}")
                log(f"   Ending VRAM: {result.gpu_str('memory_end')}")

        clear_gpu_cache()
        return result
