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


## Features
 
 - It makes images from text descriptions! 🎉
 - Generate images either in code or from command line.
 - It just works. Proper requirements are installed. model weights are automatically downloaded. No huggingface account needed. 
    (if you have the right hardware... and aren't on windows)
 - Noisy logs are gone (which was surprisingly hard to accomplish)
 - WeightedPrompts let you smash together separate prompts (cat-dog)
 - Tile Mode creates tileable images
 - Prompt metadata saved into image file metadata

## How To

```python
from imaginairy import imagine_images, imagine_image_files, ImaginePrompt, WeightedPrompt

prompts = [
    ImaginePrompt("a scenic landscape", seed=1),
    ImaginePrompt("a bowl of fruit"),
    ImaginePrompt([
       WeightedPrompt("cat", weight=1),
       WeightedPrompt("dog", weight=1),
    ])
]
for result in imagine_images(prompts):
    # do something
    result.save("my_image.jpg")
    
# or

imagine_image_files(prompts, outdir="./my-art")

```

## Requirements
- ~10 gb space for models to download
- A decent computer with either a CUDA supported graphics card or M1 processor.

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
   - https://github.com/neonsecret/stable-diffusion  https://github.com/CompVis/stable-diffusion/pull/177
   - ✅ https://github.com/huggingface/diffusers/blob/main/docs/source/optimization/fp16.mdx
   - ✅ https://github.com/CompVis/stable-diffusion/compare/main...Doggettx:stable-diffusion:autocast-improvements#
   - ✅ https://www.reddit.com/r/StableDiffusion/comments/xalaws/test_update_for_less_memory_usage_and_higher/
 - deploy to pypi
 - add tests
 - set up ci (test/lint/format)
 - add docs
 - notify https://github.com/CompVis/stable-diffusion/issues/25
 - remove yaml config
 - delete more unused code
 - Interface improvements
   - ✅ init-image at command line
   - prompt expansion
 - Image Generation Features
   - upscaling
     - https://github.com/lowfuel/progrock-stable
   - face improvements
     - gfpgan - https://github.com/TencentARC/GFPGAN
     - codeformer - https://github.com/sczhou/CodeFormer
   - image describe feature - https://replicate.com/methexis-inc/img2prompt
   - outpainting
   - inpainting
     - https://github.com/andreas128/RePaint
   - add more sampling methods?
   - img2img but keeps img stable
     - https://www.reddit.com/r/StableDiffusion/comments/xboy90/a_better_way_of_doing_img2img_by_finding_the/
     - https://gist.github.com/trygvebw/c71334dd127d537a15e9d59790f7f5e1
   - img2img for plms?
   - images as actual prompts instead of just init images
   - cross-attention control: 
     - https://github.com/bloc97/CrossAttentionControl/blob/main/CrossAttention_Release_NoImages.ipynb
   - guided generation https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1#scrollTo=UDeXQKbPTdZI
   - tiling
   - output show-work videos
   - image variations https://github.com/lstein/stable-diffusion/blob/main/VARIATIONS.md
   - textual inversion 
     - https://www.reddit.com/r/StableDiffusion/comments/xbwb5y/how_to_run_textual_inversion_locally_train_your/
     - https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb#scrollTo=50JuJUM8EG1h
   - zooming videos? a la disco diffusion
   - fix saturation at high CFG https://www.reddit.com/r/StableDiffusion/comments/xalo78/fixing_excessive_contrastsaturation_resulting/
