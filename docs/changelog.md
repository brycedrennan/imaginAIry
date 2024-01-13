
## ChangeLog

**14.1.0**
- 🎉 feature: make video generation smooth by adding frame interpolation
- feature: SDXL weights in the compvis format can now be used
- feature: allow video generation at any size specified by user
- feature: video generations output in "bounce" format
- feature: choose video output format: mp4, webp, or gif
- feature: fix random seed handling in video generation
- docs: auto-publish docs on push to master
- build: remove imageio dependency
- build: vendorize facexlib so we don't install its unneeded dependencies


**14.0.4**
- docs: add a documentation website at https://brycedrennan.github.io/imaginAIry/
- build: remove fairscale dependency
- fix: video generation was broken

**14.0.3**
- fix: several critical bugs with package
- tests: add a wheel smoketest to detect these issues in the future

**14.0.0**
- 🎉 video generation using [Stable Video Diffusion](https://github.com/Stability-AI/generative-models)
  - add `--videogen` to any image generation to create a short video from the generated image
  - or use `aimg videogen` to generate a video from an image
- 🎉 SDXL (Stable Diffusion Extra Large) models are now supported.
  - try `--model opendalle` or `--model sdxl`
  - inpainting and controlnets are not yet supported for SDXL
- 🎉 imaginairy is now backed by the [refiners library](https://github.com/finegrain-ai/refiners)
  - This was a huge rewrite which is why some features are not yet supported.  On the plus side, refiners supports
cutting edge features (SDXL, image prompts, etc) which will be added to imaginairy soon.
  - [self-attention guidance](https://github.com/SusungHong/Self-Attention-Guidance) which makes details of images more accurate
- 🎉 feature: larger image generations now work MUCH better and stay faithful to the same image as it looks at a smaller size. 
For example `--size 720p --seed 1` and `--size 1080p --seed 1` will produce the same image for SD15
- 🎉 feature: loading diffusers based models now supported. Example `--model https://huggingface.co/ainz/diseny-pixar --model-architecture sd15`
- 🎉 feature: qrcode controlnet!
- feature: generate word images automatically. great for use with qrcode controlnet: `imagine "flowers" --gif --size hd --control-mode qrcode --control-image "textimg='JOY' font_color=white background_color=gray" -r 10`
- feature: opendalle 1.1 added. `--model opendalle` to use it
- feature: added `--size` parameter for more intuitive sizing (e.g. 512, 256x256, 4k, uhd, FHD, VGA, etc)
- feature: detect if wrong torch version is installed and provide instructions on how to install proper version
- feature: better logging output: color, error handling
- feature: support for pytorch 2.0
- feature: command line output significantly cleaned up and easier to read
- feature: adds --composition-strength parameter to cli (#416)
- performance: lower memory usage for upscaling
- performance: lower memory usage at startup
- performance: add sliced attention to several models (lowers memory use)
- fix: simpler memory management that avoids some of the previous bugs
- deprecated: support for python 3.8, 3.9
- deprecated: support for torch 1.13
- deprecated: support for Stable Diffusion versions 1.4, 2.0, and 2.1
- deprecated: image training
- broken: samplers other than ddim

**13.2.1**
- fix: pydantic models for http server working now. Fixes #380
- fix: install triton so annoying message is gone

**13.2.0**
- fix: allow tile_mode to be set to True or False for backward compatibility
- fix: various pydantic issues have been resolved
- feature: switch to pydantic 2.3 (faster but was a pain to migrate)

**13.1.0**
- feature: *api server now has feature parity with the python API*. View the docs at http://127.0.0.1:8000/docs after running `aimg server`
  - `ImaginePrompt` is now a pydantic model and can thus be sent over the rest API
  - images are expected in base64 string format
- fix: pin pydantic to 2.0 for now
- build: better python 3.11 incompatibility messaging (fixes #342)
- build: add minimum versions to requirements to improve dependency resolution
- docs: add a discord link

**13.0.1**
- feature: show full stack trace when there is an api error
- fix: make lack of support for python 3.11 explicit
- fix: add some routes to match StableStudio routes

**13.0.0**
- 🎉 feature: multi-controlnet support. pass in multiple `--control-mode`, `--control-image`, and `--control-image-raw` arguments.
- 🎉 feature: add colorization controlnet. improve `aimg colorize` command
- 🎉🧪 feature: Graphical Web Interface [StableStudio](https://github.com/Stability-AI/StableStudio). run `aimg server` and visit http://127.0.0.1:8000/
- 🎉🧪 feature: API server `aimg server` command. Runs a http webserver (not finished). After running, visit http://127.0.0.1:8000/docs for api.
- 🎉🧪 feature: API support for [Stablity AI's new open-source Generative AI interface, StableStudio](https://github.com/Stability-AI/StableStudio).
- 🎉🧪 feature: "better" memory management. If GPU is full, least-recently-used model is moved to RAM. I'm not confident this works well.
- feature: [disabled] inpainting controlnet can be used instead of finetuned inpainting model
  - The inpainting controlnet doesn't work as well as the finetuned model
- feature: python interface allows configuration of controlnet strength
- feature: show full stack trace on error in cli
- fix: hide the "triton" error messages
- fix: package will not try to install xformers on `aarch64` machines. While this will allow the dockerfile to build on 
MacOS M1, [torch will not be able to use the M1 when generating images.](https://github.com/pytorch/pytorch/issues/81224#issuecomment-1499741152)
- build: specify proper Pillow minimum version (fixes #325)
- build: check for torch version at runtime (fixes #329)

**12.0.3**
- fix: exclude broken versions of timm as dependencies

**12.0.2**
- fix: move normal map preprocessor for conda compatibility

**12.0.1**
- fix: use correct device for depth images on mps. Fixes #300

**12.0.0**

- 🎉 feature: add "detail" control mode.  Add details to an image. Great for upscaling an image.
- 🎉 feature: add "edit" control mode.  Edit images using text instructions with any SD 1.5 based model. Similar to instructPix2Pix.
- 🎉 feature: add "shuffle" control mode. Image is generated from elements of control image. Similar to style transfer.
- 🎉 feature: upgrade to [controlnet 1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly)
- 🎉 fix: controlnet now works with all SD 1.5 based models
- feature: add openjourney-v4
- fix: raw control images are now properly loaded. fixes #296
- fix: filenames start numbers after latest image, even if some previous images were deleted

**11.1.1**
- fix: fix globbing bug with input image path handling
- fix: changed sample to True to generate caption using blip model

**11.1.0**

- docs: add some example use cases
- feature: add art-scene, desktop-background, interior-style, painting-style phraselists
- fix: compilation animations create normal slideshows instead of "bounces"
- fix: file globbing works in the interactive shell
- fix: fix model downloads that were broken by [library change in transformers 4.27.0](https://github.com/huggingface/transformers/commit/8f3b4a1d5bd97045541c43179efe8cd9c58adb76)

**11.0.0**
- all these changes together mean same seed/sampler will not be guaranteed to produce same image (thus the version bump)
- fix: image composition didn't work very well. Works well now but probably very slow on non-cuda platforms
- fix: remove upscaler tiling message
- fix: improve k-diffusion sampler schedule. significantly improves image quality of default sampler
- fix: img2img was broken for all samplers except plms and ddim when init image strength was >~0.25

**10.2.0**
 - feature: input raw control images (a pose, canny map, depth map, etc) directly using `--control-image-raw`
   This is opposed to current behavior of extracting the control signal from an input image via `--control-image`
 - feature: `aimg model-list` command lists included models
 - feature: system memory added to `aimg system-info` command
 - feature: add `--fix-faces` options to `aimg upscale` command
 - fix: add missing metadata attributes to generated images
 - fix: image composition step was producing unnecessarily blurry images 
 - refactor: split `aimg` cli code into multiple files
 - docs: pypi docs now link properly to github automatically

**10.1.0**
- feature: 🎉 ControlNet integration!  Control the structure of generated images.
- feature: `aimg colorize` attempts to use controlnet to colorize images
- feature: `--caption-text` command adds text at the bottom left of an image

**10.0.1**
- fix: `edit` was broken

**10.0.0**

- feature: 🎉🎉 Make large images while retaining composition. Try `imagine "a flower" -w 1920 -h 1080`
- fix: create compilations directory automatically
- perf: sliced encoding of images to latents (removes memory bottleneck)
- perf: use Silu for performance improvement over nonlinearity
- perf: `xformers` added as a dependency for linux and windows.  Gives a nice speed boost.
- perf: sliced attention now runs on MacOS. A typo prevented that from happening previously.
- perf: sliced latent decoding - now possible to make much bigger images. 3310x3310 on 11 GB GPU.

**9.0.2**
- fix: edit interface was broken

**9.0.1**
- fix: use entry_points for windows since setup.py scripts doesn't work on windows [#239](https://github.com/brycedrennan/imaginAIry/issues/239)

**9.0.0**

- perf: cli now has minimal overhead such that `aimg --help` runs in ~650ms instead of ~3400ms
- feature: `edit` and `imagine` commands now accept multiple images (which they will process separately).  This allows 
batch editing of images as requested in [#229](https://github.com/brycedrennan/imaginAIry/issues/229)
- refactor: move `--surprise-me` to its own subcommand `edit-demo`
- feature: allow selection of output image format with `--output-file-extension`
- docs: make training fail on MPS platform with useful error message
- docs: add directions on how to change model cache path

**8.3.1**
- fix: init-image-strength type

**8.3.0**
- feature: create `gifs` or `mp4s` from any images made in a single run with `--compilation-anim gif`
- feature: create a series of images or edits by iterating over a parameter with the `--arg-schedule` argument
- feature: `openjourney-v1` and `openjourney-v2` models added. available via `--model openjourney-v2`
- feature: add upscale command line function: `aimg upscale`
- feature: `--gif` option will create a gif showing the generation process for a single image
- feature: `--compare-gif` option will create a comparison gif for any image edits
- fix: tile mode was broken since latest perf improvements

**8.2.0**
- feature: added `aimg system-info` command to help debug issues

**8.1.0**
- feature: some memory optimizations and documentation
- feature: surprise-me improvements
- feature: image sizes can now be multiples of 8 instead of 64. Inputs will be silently rounded down.
- feature: cleaned up `aimg` shell logs  
- feature: auto-regen for unsafe images
- fix: make blip filename windows compatible
- fix: make captioning work with alpha pngs
 
**8.0.5**
- fix: bypass huggingface cache retrieval bug

**8.0.4**
- fix: limit attention slice size on MacOS machines with 64gb (#175)

**8.0.3**
- fix: use python 3.7 compatible lru_cache
- fix: use windows compatible filenames

**8.0.2**
- fix: hf_hub_download() got an unexpected keyword argument 'token'

**8.0.1**
- fix: spelling mistake of "surprise"

**8.0.0**
- feature: 🎉 edit images with instructions alone!
- feature: when editing an image add `--gif` to create a comparision gif
- feature: `aimg edit --surprise-me --gif my-image.jpg` for some fun pre-programmed edits
- feature: prune-ckpt command also removes the non-ema weights

**7.6.0**
- fix: default model config was broken
- feature: print version with `--version`
- feature: ability to load safetensors
- feature:  🎉 outpainting. Examples: `--outpaint up10,down300,left50,right50` or `--outpaint all100` or `--outpaint u100,d200,l300,r400`

**7.4.3**
- fix: handle old pytorch lightning imports with a graceful failure (fixes #161)
- fix: handle failed image generations better (fixes #83)

**7.4.2**
- fix: run face enhancement on GPU for 10x speedup

**7.4.1**
- fix: incorrect config files being used for non-1.0 models

**7.4.0**
- feature: 🎉 finetune your own image model. kind of like dreambooth. Read instructions on ["Concept Training"](docs/concept-training.md) page
- feature: image prep command. crops to face or other interesting parts of photo
- fix: back-compat for hf_hub_download
- feature: add prune-ckpt command
- feature: allow specification of model config file

**7.3.0**
- feature: 🎉 depth-based image-to-image generations (and inpainting) 
- fix: k_euler_a produces more consistent images per seed (randomization respects the seed again)

**7.2.0**
- feature: 🎉 tile in a single dimension ("x" or "y").  This enables, with a bit of luck, generation of 360 VR images.
Try this for example: `imagine --tile-x -w 1024 -h 512 "360 degree equirectangular panorama photograph of the mountains"  --upscale`

**7.1.1**
- fix: memory/speed regression introduced in 6.1.0
- fix: model switching now clears memory better, thus avoiding out of memory errors

**7.1.0**
- feature: 🎉 Stable Diffusion 2.1.  Generated people are no longer (completely) distorted. 
Use with `--model SD-2.1` or `--model SD-2.0-v` 

**7.0.0**
- feature: negative prompting.  `--negative-prompt` or `ImaginePrompt(..., negative_prompt="ugly, deformed, extra arms, etc")`
- feature: a default negative prompt is added to all generations. Images in SD-2.0 don't look bad anymore. Images in 1.5 look improved as well.

**6.1.2**
- fix: add back in memory-efficient algorithms

**6.1.1**
- feature: xformers will be used if available (for faster generation)
- fix: version metadata was broken

**6.1.0**
- feature: use different default steps and image sizes depending on sampler and model selected
- fix: #110 use proper version in image metadata
- refactor: solvers all have their own class that inherits from ImageSolver
- feature: 🎉🎉🎉 Stable Diffusion 2.0
  - `--model SD-2.0` to use (it makes worse images than 1.5 though...) 
  - Tested on macOS and Linux
  - All samplers working for new 512x512 model
  - New inpainting model working
  - 768x768 model working for all samplers except PLMS (`--model SD-2.0-v `)

**5.1.0**
- feature: add progress image callback

**5.0.1**
- fix: support larger images on M1. Fixes #8
- fix: support CPU generation by disabling autocast on CPU. Fixes #81

**5.0.0**
- feature: 🎉 inpainting support using new inpainting model from RunwayML. It works really well! By default, the 
inpainting model will automatically be used for any image-masking task 
- feature: 🎉 new default sampler makes image generation more than twice as fast
- feature: added `DPM++ 2S a` and `DPM++ 2M` samplers.
- feature: improve progress image logging
- fix: fix bug with `--show-work`. fixes #84
- fix: add workaround for pytorch bug affecting macOS users using the new `DPM++ 2S a` and `DPM++ 2M` samplers.
- fix: add workaround for pytorch mps bug affecting `k_dpm_fast` sampler. fixes #75
- fix: larger image sizes now work on macOS. fixes #8

**4.1.0**
 - feature: allow dynamic switching between models/weights `--model SD-1.5` or `--model SD-1.4` or `--model path/my-custom-weights.ckpt`)
 - feature: log total progress when generating images (image X out of Y)

**4.0.0**
 - feature: stable diffusion 1.5 (slightly improved image quality)
 - feature: dilation and erosion of masks
 Previously the `+` and `-` characters in a mask (example: `face{+0.1}`) added to the grayscale value of any masked areas. This wasn't very useful. The new behavior is that the mask will expand or contract by the number of pixel specified. The technical terms for this are dilation and erosion.  This allows much greater control over the masked area.
 - feature: update k-diffusion samplers. add k_dpm_adaptive and k_dpm_fast
 - feature: img2img/inpainting supported on all samplers
 - refactor: consolidates img2img/txt2img code. consolidates schedules. consolidates masking
 - ci: minor logging improvements

**3.0.1**
 - fix: k-samplers were broken

**3.0.0**
 - feature: improved safety filter

**2.4.0**
 - 🎉 feature: prompt expansion
 - feature: make (blip) photo captions more descriptive

**2.3.1**
 - fix: face fidelity default was broken

**2.3.0**
 - feature: model weights file can be specified via `--model-weights-path` argument at the command line
 - fix: set face fidelity default back to old value
 - fix: handle small images without throwing exception. credit to @NiclasEriksen
 - docs: add setuptools-rust as dependency for macos 

**2.2.1**
 - fix: init image is fully ignored if init-image-strength = 0

**2.2.0**
 - feature: face enhancement fidelity is now configurable

**2.1.0**
 - [improved masking accuracy from clipseg](https://github.com/timojl/clipseg/issues/8#issuecomment-1259150865)

**2.0.3**
 - fix memory leak in face enhancer
 - fix blurry inpainting 
 - fix for pillow compatibility

**2.0.0**
 - 🎉 fix: inpainted areas correlate with surrounding image, even at 100% generation strength.  Previously if the generation strength was high enough the generated image
would be uncorrelated to the rest of the surrounding image.  It created terrible looking images.   
 - 🎉 feature: interactive prompt added. access by running `aimg`
 - 🎉 feature: Specify advanced text based masks using boolean logic and strength modifiers. Mask descriptions must be lowercase. Keywords uppercase.
   Valid symbols: `AND`, `OR`, `NOT`, `()`, and mask strength modifier `{+0.1}` where `+` can be any of `+ - * /`. Single character boolean operators also work (`|`, `&`, `!`)
 - 🎉 feature: apply mask edits to original files with `mask_modify_original` (on by default)
 - feature: auto-rotate images if exif data specifies to do so
 - fix: mask boundaries are more accurate
 - fix: accept mask images in command line
 - fix: img2img algorithm was wrong and wouldn't at values close to 0 or 1

**1.6.2**
 - fix: another bfloat16 fix

**1.6.1**
 - fix: make sure image tensors come to the CPU as float32 so there aren't compatibility issues with non-bfloat16 cpus

**1.6.0**
 - fix: *maybe* address #13 with `expected scalar type BFloat16 but found Float`
   - at minimum one can specify `--precision full` now and that will probably fix the issue  
 - feature: tile mode can now be specified per-prompt

**1.5.3**
 - fix: missing config file for describe feature

**1.5.1**
 - img2img now supported with PLMS (instead of just DDIM)
 - added image captioning feature `aimg describe dog.jpg` => `a brown dog sitting on grass`
 - added new commandline tool `aimg` for additional image manipulation functionality

**1.4.0**
 - support multiple additive targets for masking with `|` symbol.  Example: "fruit|stem|fruit stem"

**1.3.0**
 - added prompt based image editing. Example: "fruit => gold coins"
 - test coverage improved

**1.2.0**
 - allow urls as init-images

**previous**
 - img2img actually does # of steps you specify  
 - performance optimizations
 - numerous other changes
