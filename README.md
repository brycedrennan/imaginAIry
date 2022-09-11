# ImaginAIry 🤖🧠

AI imagined images.

```bash
>> pip install imaginairy

>> imagine "a scenic landscape" "a photo of a dog" "photo of a fruit bowl" "portrait photo of a freckled woman"
```
<img src="assets/000019_786355545_PLMS50_PS7.5_a_scenic_landscape.jpg" width="256" height="256">
<img src="assets/000032_337692011_PLMS40_PS7.5_a_photo_of_a_dog.jpg" width="256" height="256">
<img src="assets/000056_293284644_PLMS40_PS7.5_photo_of_a_bowl_of_fruit.jpg" width="256" height="256">
<img src="assets/000078_260972468_PLMS40_PS7.5_portrait_photo_of_a_freckled_woman.jpg" width="256" height="256">

# Features
 
 - It makes images from text descriptions!
 - Generate images either in code or from command line.
 - It just works (if you have the right hardware)
 - Noisy logs are gone (which was surprisingly hard to accomplish)
 - WeightedPrompts let you smash together separate prompts ()

# How To

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

# Requirements

- Computer with CUDA supported graphics card. ~10 gb video ram
OR
- Apple M1 computer

# Improvements from CompVis
 - img2img actually does # of steps you specify

# Models Used
 - CLIP
 - LDM - Latent Diffusion
 - Stable Diffusion - https://github.com/CompVis/stable-diffusion

# Todo
 - add docs
 - deploy to pypi
 - add tests
 - set up ci (test/lint/format)
 - notify https://github.com/CompVis/stable-diffusion/issues/25
 - remove yaml config
 - performance optimizations https://github.com/huggingface/diffusers/blob/main/docs/source/optimization/fp16.mdx 
 - Interface improvements
   - init-image at command line
   - prompt expansion?
   - webserver interface (low priority, this is a library)
 - Image Generation Features
   - upscaling
   - face improvements
   - image describe feature
   - outpainting
   - inpainting
   - cross-attention control: 
     - https://github.com/bloc97/CrossAttentionControl/blob/main/CrossAttention_Release_NoImages.ipynb
   - tiling
   - output show-work videos
   - zooming videos? a la disco diffusion

 