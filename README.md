# ImaginAIry 🤖🧠

AI imagined images. Pythonic generation of stable diffusion images.

"just works" on Linux and OSX(M1).

## Examples
```bash
>> pip install imaginairy
>> imagine "a scenic landscape" "a photo of a dog" "photo of a fruit bowl" "portrait photo of a freckled woman"
```

<details closed>
<summary>Console Output</summary>

```bash
🤖🧠 received 4 prompt(s) and will repeat them 1 times to create 4 images.
Loading model onto mps backend...
Generating 🖼  : "a scenic landscape" 512x512px seed:557988237 prompt-strength:7.5 steps:40 sampler-type:PLMS
    PLMS Sampler: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:29<00:00,  1.36it/s]
    🖼  saved to: ./outputs/000001_557988237_PLMS40_PS7.5_a_scenic_landscape.jpg
Generating 🖼  : "a photo of a dog" 512x512px seed:277230171 prompt-strength:7.5 steps:40 sampler-type:PLMS
    PLMS Sampler: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:28<00:00,  1.41it/s]
    🖼  saved to: ./outputs/000002_277230171_PLMS40_PS7.5_a_photo_of_a_dog.jpg
Generating 🖼  : "photo of a fruit bowl" 512x512px seed:639753980 prompt-strength:7.5 steps:40 sampler-type:PLMS
    PLMS Sampler: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:28<00:00,  1.40it/s]
    🖼  saved to: ./outputs/000003_639753980_PLMS40_PS7.5_photo_of_a_fruit_bowl.jpg
Generating 🖼  : "portrait photo of a freckled woman" 512x512px seed:500686645 prompt-strength:7.5 steps:40 sampler-type:PLMS
    PLMS Sampler: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:29<00:00,  1.37it/s]
    🖼  saved to: ./outputs/000004_500686645_PLMS40_PS7.5_portrait_photo_of_a_freckled_woman.jpg
```
</details>

<img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000019_786355545_PLMS50_PS7.5_a_scenic_landscape.jpg" height="256"><img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000032_337692011_PLMS40_PS7.5_a_photo_of_a_dog.jpg"  height="256"><br>
<img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000056_293284644_PLMS40_PS7.5_photo_of_a_bowl_of_fruit.jpg" height="256"><img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000078_260972468_PLMS40_PS7.5_portrait_photo_of_a_freckled_woman.jpg"  height="256">


### Tiled Images
```bash
>> imagine  "gold coins" "a lush forest" "piles of old books" leaves --tile
```

<img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000066_801493266_PLMS40_PS7.5_gold_coins.jpg" height="128"><img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000066_801493266_PLMS40_PS7.5_gold_coins.jpg" height="128"><img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000066_801493266_PLMS40_PS7.5_gold_coins.jpg" height="128">
<img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000118_597948545_PLMS40_PS7.5_a_lush_forest.jpg" height="128"><img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000118_597948545_PLMS40_PS7.5_a_lush_forest.jpg" height="128"><img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000118_597948545_PLMS40_PS7.5_a_lush_forest.jpg" height="128">
<br>
<img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000075_961095192_PLMS40_PS7.5_piles_of_old_books.jpg" height="128"><img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000075_961095192_PLMS40_PS7.5_piles_of_old_books.jpg" height="128"><img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000075_961095192_PLMS40_PS7.5_piles_of_old_books.jpg" height="128">
<img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000040_527733581_PLMS40_PS7.5_leaves.jpg" height="128"><img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000040_527733581_PLMS40_PS7.5_leaves.jpg" height="128"><img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000040_527733581_PLMS40_PS7.5_leaves.jpg" height="128">

### Image-to-Image
```bash
>> imagine "portrait of a smiling lady. oil painting" --init-image girl_with_a_pearl_earring.jpg
```
<img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/tests/data/girl_with_a_pearl_earring.jpg" height="256"> =>
<img src="https://raw.githubusercontent.com/brycedrennan/imaginAIry/master/assets/000105_33084057_DDIM40_PS7.5_portrait_of_a_smiling_lady._oil_painting._.jpg" height="256"> 

### Face Enhancement [by CodeFormer](https://github.com/sczhou/CodeFormer)

```bash
>> imagine "a couple smiling" --steps 40 --seed 1 --fix-faces
```
<img src="https://github.com/brycedrennan/imaginAIry/raw/master/assets/000178_1_PLMS40_PS7.5_a_couple_smiling_nofix.png" height="256"> =>
<img src="https://github.com/brycedrennan/imaginAIry/raw/master/assets/000178_1_PLMS40_PS7.5_a_couple_smiling_fixed.png" height="256"> 


### Upscaling [by RealESRGAN](https://github.com/xinntao/Real-ESRGAN)
```bash
>> imagine "colorful smoke" --steps 40 --upscale
```
<img src="https://github.com/brycedrennan/imaginAIry/raw/master/assets/000206_856637805_PLMS40_PS7.5_colorful_smoke.jpg" height="128"> =>
<img src="https://github.com/brycedrennan/imaginAIry/raw/master/assets/000206_856637805_PLMS40_PS7.5_colorful_smoke_upscaled.jpg" height="256"> 

## Features
 
 - It makes images from text descriptions! 🎉
 - Generate images either in code or from command line.
 - It just works. Proper requirements are installed. model weights are automatically downloaded. No huggingface account needed. 
    (if you have the right hardware... and aren't on windows)
 - No more distorted faces!
 - Noisy logs are gone (which was surprisingly hard to accomplish)
 - WeightedPrompts let you smash together separate prompts (cat-dog)
 - Tile Mode creates tileable images
 - Prompt metadata saved into image file metadata

## How To

```python
from imaginairy import imagine, imagine_image_files, ImaginePrompt, WeightedPrompt, LazyLoadingImage

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Thomas_Cole_-_Architect%E2%80%99s_Dream_-_Google_Art_Project.jpg/540px-Thomas_Cole_-_Architect%E2%80%99s_Dream_-_Google_Art_Project.jpg"
prompts = [
    ImaginePrompt("a scenic landscape", seed=1),
    ImaginePrompt("a bowl of fruit"),
    ImaginePrompt([
        WeightedPrompt("cat", weight=1),
        WeightedPrompt("dog", weight=1),
    ]),
    ImaginePrompt(
        "a spacious building", 
        init_image=LazyLoadingImage(url)
    )
]
for result in imagine(prompts):
    # do something
    result.save("my_image.jpg")

# or

imagine_image_files(prompts, outdir="./my-art")

```

## Requirements
- ~10 gb space for models to download
- A decent computer with either a CUDA supported graphics card or M1 processor.

## Running in Docker
See example Dockerfile (works on machine where you can pass the gpu into the container)
```bash
docker build . -t imaginairy
# you really want to map the cache or you end up wasting a lot of time and space redownloading the model weights
docker run -it --gpus all -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $HOME/.cache/torch:/root/.cache/torch -v `pwd`/outputs:/outputs imaginairy /bin/bash
```

## Improvements from CompVis
 - img2img actually does # of steps you specify
 - performance optimizations

## Models Used
 - CLIP - https://openai.com/blog/clip/
 - LDM - Latent Diffusion
 - Stable Diffusion 
   - https://github.com/CompVis/stable-diffusion
   - https://huggingface.co/CompVis/stable-diffusion-v1-4
   - https://laion.ai/blog/laion-5b/

## Not Supported
 - a web interface. this is a python library

## Todo
 - performance optimizations
   - ✅ https://github.com/huggingface/diffusers/blob/main/docs/source/optimization/fp16.mdx
   - ✅ https://github.com/CompVis/stable-diffusion/compare/main...Doggettx:stable-diffusion:autocast-improvements#
   - ✅ https://www.reddit.com/r/StableDiffusion/comments/xalaws/test_update_for_less_memory_usage_and_higher/
   - https://github.com/neonsecret/stable-diffusion  https://github.com/CompVis/stable-diffusion/pull/177
 - ✅ deploy to pypi
 - find similar images https://knn5.laion.ai/?back=https%3A%2F%2Fknn5.laion.ai%2F&index=laion5B&useMclip=false
 - Development Environment
   - add tests
   - set up ci (test/lint/format)
   - add docs
   - remove yaml config
   - delete more unused code
 - Interface improvements
   - ✅ init-image at command line
   - prompt expansion
 - Image Generation Features
   - ✅ add k-diffusion sampling methods
   - why is k-diffusion so slow compared to plms? 2 it/s vs 8 it/s
   - negative prompting
     - some syntax to allow it in a text string
   - upscaling
     - ✅ realesrgan 
     - ldm
     - https://github.com/lowfuel/progrock-stable
   - ✅ face enhancers
     - ✅ gfpgan - https://github.com/TencentARC/GFPGAN
     - ✅ codeformer - https://github.com/sczhou/CodeFormer
   - image describe feature - 
     - https://replicate.com/methexis-inc/img2prompt
     - https://github.com/KaiyangZhou/CoOp
   - outpainting
   - inpainting
     - https://github.com/andreas128/RePaint
     - img2img but keeps img stable
     - https://www.reddit.com/r/StableDiffusion/comments/xboy90/a_better_way_of_doing_img2img_by_finding_the/
     - https://gist.github.com/trygvebw/c71334dd127d537a15e9d59790f7f5e1
     - https://github.com/pesser/stable-diffusion/commit/bbb52981460707963e2a62160890d7ecbce00e79
   - CPU support
   - img2img for plms?
   - images as actual prompts instead of just init images
     - requires model fine-tuning since SD1.4 expects 77x768 text encoding input
     - https://twitter.com/Buntworthy/status/1566744186153484288
     - https://github.com/justinpinkney/stable-diffusion
     - https://github.com/LambdaLabsML/lambda-diffusers
     - https://www.reddit.com/r/MachineLearning/comments/x6k5bm/n_stable_diffusion_image_variations_released/
     - 
   - cross-attention control: 
     - https://github.com/bloc97/CrossAttentionControl/blob/main/CrossAttention_Release_NoImages.ipynb
   - guided generation 
     - https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1#scrollTo=UDeXQKbPTdZI
   - ✅ tiling
   - output show-work videos
   - image variations https://github.com/lstein/stable-diffusion/blob/main/VARIATIONS.md
   - textual inversion 
     - https://www.reddit.com/r/StableDiffusion/comments/xbwb5y/how_to_run_textual_inversion_locally_train_your/
     - https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb#scrollTo=50JuJUM8EG1h
     - https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion_textual_inversion_library_navigator.ipynb
   - fix saturation at high CFG https://www.reddit.com/r/StableDiffusion/comments/xalo78/fixing_excessive_contrastsaturation_resulting/
   - https://www.reddit.com/r/StableDiffusion/comments/xbrrgt/a_rundown_of_twenty_new_methodsoptions_added_to/

## Noteable Stable Diffusion Implementations
 - https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion
 - https://github.com/lstein/stable-diffusion
 - https://github.com/AUTOMATIC1111/stable-diffusion-webui

## Further Reading
 - Differences between samplers
   - https://www.reddit.com/r/StableDiffusion/comments/xbeyw3/can_anyone_offer_a_little_guidance_on_the/