# ImaginAIry 🤖🧠

[![Downloads](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rOvQNs0Cmn_yU1bKWjCOHzGVDgZkaTtO?usp=sharing)
[![Downloads](https://pepy.tech/badge/imaginairy)](https://pepy.tech/project/imaginairy)
[![image](https://img.shields.io/pypi/v/imaginairy.svg)](https://pypi.org/project/imaginairy/)
[![image](https://img.shields.io/badge/license-MIT-green)](https://github.com/brycedrennan/imaginAIry/blob/master/LICENSE/)
[![Discord](https://flat.badgen.net/discord/members/FdD7ut3YjW)](https://discord.gg/FdD7ut3YjW)

AI imagined images. Pythonic generation of stable diffusion images **and videos** *!.

"just works" on Linux and macOS(M1) (and sometimes windows).


```bash
# on macOS, make sure rust is installed first
# be sure to use Python 3.10, Python 3.11 is not supported at the moment
>> pip install imaginairy
>> imagine "a scenic landscape" "a photo of a dog" "photo of a fruit bowl" "portrait photo of a freckled woman" "a bluejay"
# Make an AI video
>> aimg videogen --start-image rocket.png
```
## Stable Video Diffusion
<p float="left">
<img src="docs/assets/svd-rocket.gif" height="190">
<img src="docs/assets/svd-athens.gif" height="190">
<img src="docs/assets/svd-pearl-girl.gif" height="190">
<img src="docs/assets/svd-starry-night.gif" height="190">
<img src="docs/assets/svd-dog.gif" height="190">
<img src="docs/assets/svd-xpbliss.gif" height="190">
</p>

### Rushed release of Stable Diffusion Video!
Works with Nvidia GPUs.  Does not work on Mac or CPU.

On Windows you'll need to install torch 2.0 first via https://pytorch.org/get-started/locally/
```text
Usage: aimg videogen [OPTIONS]

  AI generate a video from an image

  Example:

      aimg videogen --start-image assets/rocket-wide.png

Options:
  --start-image TEXT       Input path for image file.
  --num-frames INTEGER     Number of frames.
  --num-steps INTEGER      Number of steps.
  --model TEXT             Model to use. One of: svd, svd_xt, svd_image_decoder, svd_xt_image_decoder
  --fps INTEGER            FPS for the AI to target when generating video
  --output-fps INTEGER     FPS for the output video
  --motion-amount INTEGER  How much motion to generate. value between 0 and 255.
  -r, --repeats INTEGER    How many times to repeat the renders.   [default: 1]
  --cond-aug FLOAT         Conditional augmentation.
  --seed INTEGER           Seed for random number generator.
  --decoding_t INTEGER     Number of frames decoded at a time.
  --output_folder TEXT     Output folder.
  --help                   Show this message and exit.
```

### Images
<p float="left">
<img src="docs/assets/026882_1_ddim50_PS7.5_a_scenic_landscape_[generated].jpg" height="256">
<img src="docs/assets/026884_1_ddim50_PS7.5_photo_of_a_dog_[generated].jpg" height="256">
<img src="docs/assets/026890_1_ddim50_PS7.5_photo_of_a_bowl_of_fruit._still_life_[generated].jpg" height="256">
<img src="docs/assets/026885_1_ddim50_PS7.5_girl_with_a_pearl_earring_[generated].jpg" height="256">
<img src="docs/assets/026891_1_ddim50_PS7.5_close-up_photo_of_a_bluejay_[generated].jpg" height="256">
<img src="docs/assets/026893_1_ddim50_PS7.5_macro_photo_of_a_flower_[generated].jpg" height="256">
</p>

### Whats New
[See full Changelog here](./docs/changelog.md)

**14.1.1**
- tests: add installation tests for windows, mac, and conda
- fix: dependency issues

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


### Run API server and StableStudio web interface (alpha)
Generate images via API or web interface.  Much smaller featureset compared to the command line tool.
```bash
>> aimg server
```
Visit http://localhost:8000/ and http://localhost:8000/docs

<img src="https://github.com/Stability-AI/StableStudio/blob/a65d4877ad7d309627808a169818f1add8c278ae/misc/GenerateScreenshot.png?raw=true" width="512">

### Image Structure Control [by ControlNet](https://github.com/lllyasviel/ControlNet)
#### (Not supported for SDXL yet)
Generate images guided by body poses, depth maps, canny edges, hed boundaries, or normal maps.

**Openpose Control**

```bash
imagine --control-image assets/indiana.jpg  --control-mode openpose --caption-text openpose "photo of a polar bear"
```

<p float="left">
    <img src="docs/assets/indiana.jpg" height="256">
    <img src="docs/assets/indiana-pose.jpg" height="256">
    <img src="docs/assets/indiana-pose-polar-bear.jpg" height="256">
</p>

#### Canny Edge Control

```bash
imagine --control-image assets/lena.png  --control-mode canny "photo of a woman with a hat looking at the camera"
```

<p float="left">
    <img src="docs/assets/lena.png" height="256">
    <img src="docs/assets/lena-canny.jpg" height="256">
    <img src="docs/assets/lena-canny-generated.jpg" height="256">
</p>

#### HED Boundary Control

```bash
imagine --control-image dog.jpg  --control-mode hed  "photo of a dalmation"
```

<p float="left">
    <img src="docs/assets/000032_337692011_PLMS40_PS7.5_a_photo_of_a_dog.jpg" height="256">
    <img src="docs/assets/dog-hed-boundary.jpg" height="256">
    <img src="docs/assets/dog-hed-boundary-dalmation.jpg" height="256">
</p>

#### Depth Map Control

```bash
imagine --control-image fancy-living.jpg  --control-mode depth  "a modern living room"
```

<p float="left">
    <img src="docs/assets/fancy-living.jpg" height="256">
    <img src="docs/assets/fancy-living-depth.jpg" height="256">
    <img src="docs/assets/fancy-living-depth-generated.jpg" height="256">
</p>

#### Normal Map Control

```bash
imagine --control-image bird.jpg  --control-mode normal  "a bird"
```

<p float="left">
    <img src="docs/assets/013986_1_kdpmpp2m59_PS7.5_a_bluejay_[generated].jpg" height="256">
    <img src="docs/assets/bird-normal.jpg" height="256">
    <img src="docs/assets/bird-normal-generated.jpg" height="256">
</p>

#### Image Shuffle Control

Generates the image based on elements of the control image. Kind of similar to style transfer.
```bash
imagine --control-image pearl-girl.jpg  --control-mode shuffle  "a clown"
```
The middle image is the "shuffled" input image
<p float="left">
    <img src="docs/assets/girl_with_a_pearl_earring.jpg" height="256">
    <img src="docs/assets/pearl_shuffle_019331_1_kdpmpp2m15_PS7.5_img2img-0.0_a_clown.jpg" height="256">
    <img src="docs/assets/pearl_shuffle_clown_019331_1_kdpmpp2m15_PS7.5_img2img-0.0_a_clown.jpg" height="256">
</p>

#### Editing Instructions Control

Similar to instructPix2Pix (below) but works with any SD 1.5 based model.
```bash
imagine --control-image pearl-girl.jpg  --control-mode edit --init-image-strength 0.01 --steps 30  --negative-prompt "" --model openjourney-v2 "make it anime" "make it at the beach" 
```

<p float="left">
    <img src="docs/assets/girl_with_a_pearl_earring.jpg" height="256">
    <img src="docs/assets/pearl_anime_019537_521829407_kdpmpp2m30_PS9.0_img2img-0.01_make_it_anime.jpg" height="256">
    <img src="docs/assets/pearl_beach_019561_862735879_kdpmpp2m30_PS7.0_img2img-0.01_make_it_at_the_beach.jpg" height="256">
</p>

#### Add Details Control (upscaling/super-resolution)

Replaces existing details in an image. Good to use with --init-image-strength 0.2
```bash
imagine --control-image "assets/wishbone.jpg" --control-mode details "sharp focus, high-resolution" --init-image-strength 0.2 --steps 30 -w 2048 -h 2048 
```

<p float="left">
    <img src="docs/assets/wishbone_headshot_badscale.jpg" height="256">
    <img src="docs/assets/wishbone_headshot_details.jpg" height="256">
</p>


### Image (re)Colorization (using brightness control)
Colorize black and white images or re-color existing images.

The generated colors will be applied back to the original image. You can either provide a caption or 
allow the tool to generate one for you.

```bash
aimg colorize pearl-girl.jpg --caption "photo of a woman"
```
<p float="left">
    <img src="docs/assets/girl_with_a_pearl_earring.jpg" height="256">
    <img src="docs/assets/pearl-gray.jpg" height="256">
    <img src="docs/assets/pearl-recolor-a.jpg" height="256">
</p>

###  Instruction based image edits [by InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)
#### (Broken as of 14.0.0)
Just tell imaginairy how to edit the image and it will do it for you!
<p float="left">
<img src="docs/assets/scenic_landscape_winter.jpg" height="256">
<img src="docs/assets/dog_red.jpg" height="256">
<img src="docs/assets/bowl_of_fruit_strawberries.jpg" height="256">
<img src="docs/assets/freckled_woman_cyborg.jpg" height="256">
<img src="docs/assets/014214_51293814_kdpmpp2m30_PS10.0_img2img-1.0_make_the_bird_wear_a_cowboy_hat_[generated].jpg" height="256">
<img src="docs/assets/flower-make-the-flower-out-of-paper-origami.gif" height="256">
<img src="docs/assets/girl-pearl-clown-compare.gif" height="256">
<img src="docs/assets/mona-lisa-headshot-anim.gif" height="256">
<img src="docs/assets/make-it-night-time.gif" height="256">
</p>

<details>
<summary>Click to see shell commands</summary>
Use prompt strength to control how strong the edit is. For extra control you can combine with prompt-based masking.

```bash
# enter imaginairy shell
>> aimg
🤖🧠> edit scenic_landscape.jpg -p "make it winter" --prompt-strength 20
🤖🧠> edit dog.jpg -p "make the dog red" --prompt-strength 5
🤖🧠> edit bowl_of_fruit.jpg -p "replace the fruit with strawberries"
🤖🧠> edit freckled_woman.jpg -p "make her a cyborg" --prompt-strength 13
🤖🧠> edit bluebird.jpg -p "make the bird wear a cowboy hat" --prompt-strength 10
🤖🧠> edit flower.jpg -p "make the flower out of paper origami" --arg-schedule prompt-strength[1:11:0.3]  --steps 25 --compilation-anim gif

# create a comparison gif
🤖🧠> edit pearl_girl.jpg -p "make her wear clown makeup" --compare-gif
# create an animation showing the edit with increasing prompt strengths
🤖🧠> edit mona-lisa.jpg -p "make it a color professional photo headshot" --negative-prompt "old, ugly, blurry" --arg-schedule "prompt-strength[2:8:0.5]" --compilation-anim gif
🤖🧠> edit gg-bridge.jpg -p "make it night time" --prompt-strength 15  --steps 30 --arg-schedule prompt-strength[1:15:1] --compilation-anim gif
```
</details>



### Quick Image Edit Demo
Want just quickly have some fun? Try `edit-demo` to apply some pre-defined edits.
```bash
>> aimg edit-demo pearl_girl.jpg
```
<p float="left">
<img src="docs/assets/girl_with_a_pearl_earring_suprise.gif" height="256">
<img src="docs/assets/mona-lisa-suprise.gif" height="256">
<img src="docs/assets/luke-suprise.gif" height="256">
<img src="docs/assets/spock-suprise.gif" height="256">
<img src="docs/assets/gg-bridge-suprise.gif" height="256">
<img src="docs/assets/shire-suprise.gif" height="256">
</p>


### Prompt Based Masking  [by clipseg](https://github.com/timojl/clipseg)
Specify advanced text based masks using boolean logic and strength modifiers. 
Mask syntax:
  - mask descriptions must be lowercase
  - keywords (`AND`, `OR`, `NOT`) must be uppercase
  - parentheses are supported 
  - mask modifiers may be appended to any mask or group of masks.  Example: `(dog OR cat){+5}` means that we'll
select any dog or cat and then expand the size of the mask area by 5 pixels.  Valid mask modifiers:
    - `{+n}` - expand mask by n pixels
    - `{-n}` - shrink mask by n pixels
    - `{*n}` - multiply mask strength. will expand mask to areas that weakly matched the mask description
    - `{/n}` - divide mask strength. will reduce mask to areas that most strongly matched the mask description. probably not useful

When writing strength modifiers keep in mind that pixel values are between 0 and 1.

```bash
>> imagine \
    --init-image pearl_earring.jpg \
    --mask-prompt "face AND NOT (bandana OR hair OR blue fabric){*6}" \
    --mask-mode keep \
    --init-image-strength .2 \
    --fix-faces \
    "a modern female president" "a female robot" "a female doctor" "a female firefighter"
```
<img src="docs/assets/mask_examples/pearl000.jpg" height="200">➡️ 
<img src="docs/assets/mask_examples/pearl_pres.png" height="200">
<img src="docs/assets/mask_examples/pearl_robot.png" height="200">
<img src="docs/assets/mask_examples/pearl_doctor.png" height="200">
<img src="docs/assets/mask_examples/pearl_firefighter.png" height="200">

```bash
>> imagine \
    --init-image fruit-bowl.jpg \
    --mask-prompt "fruit OR fruit stem{*6}" \
    --mask-mode replace \
    --mask-modify-original \
    --init-image-strength .1 \
    "a bowl of kittens" "a bowl of gold coins" "a bowl of popcorn" "a bowl of spaghetti"
```
<img src="docs/assets/000056_293284644_PLMS40_PS7.5_photo_of_a_bowl_of_fruit.jpg" height="200">➡️ 
<img src="docs/assets/mask_examples/bowl004.jpg" height="200">
<img src="docs/assets/mask_examples/bowl001.jpg" height="200">
<img src="docs/assets/mask_examples/bowl002.jpg" height="200">
<img src="docs/assets/mask_examples/bowl003.jpg" height="200">


### Face Enhancement [by CodeFormer](https://github.com/sczhou/CodeFormer)

```bash
>> imagine "a couple smiling" --steps 40 --seed 1 --fix-faces
```
<img src="https://github.com/brycedrennan/imaginAIry/raw/master/assets/000178_1_PLMS40_PS7.5_a_couple_smiling_nofix.png" height="256"> ➡️ 
<img src="https://github.com/brycedrennan/imaginAIry/raw/master/assets/000178_1_PLMS40_PS7.5_a_couple_smiling_fixed.png" height="256"> 


### Upscaling [by RealESRGAN](https://github.com/xinntao/Real-ESRGAN)
```bash
>> imagine "colorful smoke" --steps 40 --upscale
# upscale an existing image
>> aimg upscale my-image.jpg
```
<details>
<summary>Python Example</summary>

```python
from imaginairy.enhancers.upscale_realesrgan import upscale_image
from PIL import Image
img = Image.open("my-image.jpg")
big_img = upscale_image(i)
```

</details>
<img src="https://github.com/brycedrennan/imaginAIry/raw/master/assets/000206_856637805_PLMS40_PS7.5_colorful_smoke.jpg" height="128"> ➡️ 
<img src="https://github.com/brycedrennan/imaginAIry/raw/master/assets/000206_856637805_PLMS40_PS7.5_colorful_smoke_upscaled.jpg" height="256"> 

### Tiled Images
```bash
>> imagine  "gold coins" "a lush forest" "piles of old books" leaves --tile
```

<img src="docs/assets/000066_801493266_PLMS40_PS7.5_gold_coins.jpg" height="128"><img src="docs/assets/000066_801493266_PLMS40_PS7.5_gold_coins.jpg" height="128"><img src="docs/assets/000066_801493266_PLMS40_PS7.5_gold_coins.jpg" height="128">
<img src="docs/assets/000118_597948545_PLMS40_PS7.5_a_lush_forest.jpg" height="128"><img src="docs/assets/000118_597948545_PLMS40_PS7.5_a_lush_forest.jpg" height="128"><img src="docs/assets/000118_597948545_PLMS40_PS7.5_a_lush_forest.jpg" height="128">
<br>
<img src="docs/assets/000075_961095192_PLMS40_PS7.5_piles_of_old_books.jpg" height="128"><img src="docs/assets/000075_961095192_PLMS40_PS7.5_piles_of_old_books.jpg" height="128"><img src="docs/assets/000075_961095192_PLMS40_PS7.5_piles_of_old_books.jpg" height="128">
<img src="docs/assets/000040_527733581_PLMS40_PS7.5_leaves.jpg" height="128"><img src="docs/assets/000040_527733581_PLMS40_PS7.5_leaves.jpg" height="128"><img src="docs/assets/000040_527733581_PLMS40_PS7.5_leaves.jpg" height="128">
#### 360 degree images
```bash
imagine --tile-x -w 1024 -h 512 "360 degree equirectangular panorama photograph of the desert"  --upscale
```
<img src="docs/assets/desert_360.jpg" height="128">

### Image-to-Image
Use depth maps for amazing "translations" of existing images.

```bash
>> imagine --init-image girl_with_a_pearl_earring_large.jpg --init-image-strength 0.05  "professional headshot photo of a woman with a pearl earring" -r 4 -w 1024 -h 1024 --steps 50
```
<p float="left">
<img src="tests/data/girl_with_a_pearl_earring.jpg" width="256"> ➡️ 
<img src="docs/assets/pearl_depth_1.jpg" width="256">
<img src="docs/assets/pearl_depth_2.jpg" width="256">
</p>


### Outpainting

Given a starting image, one can generate it's "surroundings".

Example:
`imagine --init-image pearl-earring.jpg --init-image-strength 0 --outpaint all250,up0,down600 "woman standing"`

<img src="tests/data/girl_with_a_pearl_earring.jpg" height="256"> ➡️ 
<img src="tests/expected_output/test_outpainting_outpaint_.png" height="256">

### Work with different generation models

<p float="left">
    <img src="docs/assets/fairytale-treehouse-sd15.jpg" height="256">
    <img src="docs/assets/fairytale-treehouse-openjourney-v1.jpg" height="256">
    <img src="docs/assets/fairytale-treehouse-openjourney-v2.jpg" height="256">
</p>

<details>
<summary>Click to see shell command</summary>

```bash
imagine "valley, fairytale treehouse village covered, , matte painting, highly detailed, dynamic lighting, cinematic, realism, realistic, photo real, sunset, detailed, high contrast, denoised, centered, michael whelan" --steps 60 --seed 1 --arg-schedule model[sd14,sd15,sd20,sd21,openjourney-v1,openjourney-v2] --arg-schedule "caption-text[sd14,sd15,sd20,sd21,openjourney-v1,openjourney-v2]"
```
</details>

### Prompt Expansion
You can use `{}` to randomly pull values from lists.  A list of values separated by `|` 
 and enclosed in `{ }` will be randomly drawn from in a non-repeating fashion. Values that are surrounded by `_ _` will 
 pull from a phrase list of the same name.   Folders containing .txt phraselist files may be specified via
`--prompt_library_path`. The option may be specified multiple times.  Built-in categories:
    
      3d-term, adj-architecture, adj-beauty, adj-detailed, adj-emotion, adj-general, adj-horror, animal, art-scene, art-movement, 
      art-site, artist, artist-botanical, artist-surreal, aspect-ratio, bird, body-of-water, body-pose, camera-brand,
      camera-model, color, cosmic-galaxy, cosmic-nebula, cosmic-star, cosmic-term, desktop-background, dinosaur, eyecolor, f-stop, 
      fantasy-creature, fantasy-setting, fish, flower, focal-length, food, fruit, games, gen-modifier, hair, hd,
      iso-stop, landscape-type, national-park, nationality, neg-weight, noun-beauty, noun-fantasy, noun-general, 
      noun-horror, occupation, painting-style, photo-term, pop-culture, pop-location, punk-style, quantity, rpg-item, scenario-desc, 
      skin-color, spaceship, style, tree-species, trippy, world-heritage-site

   Examples:

   `imagine "a {lime|blue|silver|aqua} colored dog" -r 4 --seed 0` (note that it generates a dog of each color without repetition)

<img src="docs/assets/000184_0_plms40_PS7.5_a_silver_colored_dog_[generated].jpg" height="200"><img src="docs/assets/000186_0_plms40_PS7.5_a_aqua_colored_dog_[generated].jpg" height="200">
<img src="docs/assets/000210_0_plms40_PS7.5_a_lime_colored_dog_[generated].jpg" height="200">
<img src="docs/assets/000211_0_plms40_PS7.5_a_blue_colored_dog_[generated].jpg" height="200">

   `imagine "a {_color_} dog" -r 4 --seed 0` will generate four, different colored dogs. The colors will be pulled from an included 
   phraselist of colors.
    
   `imagine "a {_spaceship_|_fruit_|hot air balloon}. low-poly" -r 4 --seed 0` will generate images of spaceships or fruits or a hot air balloon

<details>
<summary>Python example</summary>

```python
from imaginairy.enhancers.prompt_expansion import expand_prompts

my_prompt = "a giant {_animal_}"

expanded_prompts = expand_prompts(n=10, prompt_text=my_prompt, prompt_library_paths=["./prompts"])
```
</details>

   Credit to [noodle-soup-prompts](https://github.com/WASasquatch/noodle-soup-prompts/) where most, but not all, of the wordlists originate.

### Generate image captions (via [BLIP](https://github.com/salesforce/BLIP))
```bash
>> aimg describe assets/mask_examples/bowl001.jpg
a bowl full of gold bars sitting on a table
```

### Example Use Cases

```bash
>> aimg
# Generate endless 8k art
🤖🧠> imagine -w 1920 -h 1080 --upscale "{_art-scene_}. {_painting-style_} by {_artist_}" -r 1000 --steps 30 --model sd21v

# generate endless desktop backgrounds 
🤖🧠> imagine --tile "{_desktop-background_}" -r 100

# convert a folder of images to pencil sketches
🤖🧠> edit other/images/*.jpg -p "make it a pencil sketch"

# upscale a folder of images
🤖🧠> upscale my-images/*.jpg

# generate kitchen remodel ideas
🤖🧠> imagine --control-image kitchen.jpg -w 1024 -h 1024 "{_interior-style_} kitchen" --control-mode depth -r 100 --init-image 0.01 --upscale --steps 35 --caption-text "{prompt}"
```

### Additional Features
 - Generate images either in code or from command line.
 - It just works. Proper requirements are installed. Model weights are automatically downloaded. No huggingface account needed. 
    (if you have the right hardware... and aren't on windows)
 - Noisy logs are gone (which was surprisingly hard to accomplish)
 - WeightedPrompts let you smash together separate prompts (cat-dog)
 - Prompt metadata saved into image file metadata
 - Have AI generate captions for images `aimg describe <filename-or-url>`
 - Interactive prompt: just run `aimg`
 
## How To

For full command line instructions run `aimg --help`

```python
from imaginairy import imagine, imagine_image_files, ImaginePrompt, WeightedPrompt, LazyLoadingImage

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Thomas_Cole_-_Architect%E2%80%99s_Dream_-_Google_Art_Project.jpg/540px-Thomas_Cole_-_Architect%E2%80%99s_Dream_-_Google_Art_Project.jpg"
prompts = [
    ImaginePrompt("a scenic landscape", seed=1, upscale=True),
    ImaginePrompt("a bowl of fruit"),
    ImaginePrompt([
        WeightedPrompt("cat", weight=1),
        WeightedPrompt("dog", weight=1),
    ]),
    ImaginePrompt(
        "a spacious building", 
        init_image=LazyLoadingImage(url=url)
    ),
    ImaginePrompt(
        "a bowl of strawberries", 
        init_image=LazyLoadingImage(filepath="mypath/to/bowl_of_fruit.jpg"),
        mask_prompt="fruit OR stem{*2}",  # amplify the stem mask x2
        mask_mode="replace",
        mask_modify_original=True,
    ),
    ImaginePrompt("strawberries", tile_mode=True),
]
for result in imagine(prompts):
    # do something
    result.save("my_image.jpg")

# or

imagine_image_files(prompts, outdir="./my-art")

```

## Requirements
- ~10 gb space for models to download
- A CUDA supported graphics card with >= 11gb VRAM (and CUDA installed) or an M1 processor.
- Python installed. Preferably Python 3.10.  (not conda)
- For macOS [rust](https://www.rust-lang.org/tools/install) and setuptools-rust must be installed to compile the `tokenizer` library.
They can be installed via: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` and `pip install setuptools-rust`
    

## Running in Docker
See example Dockerfile (works on machine where you can pass the gpu into the container)
```bash
docker build . -t imaginairy
# you really want to map the cache or you end up wasting a lot of time and space redownloading the model weights
docker run -it --gpus all -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $HOME/.cache/torch:/root/.cache/torch -v `pwd`/outputs:/outputs imaginairy /bin/bash
```

## Running on Google Colab
[Example Colab](https://colab.research.google.com/drive/1rOvQNs0Cmn_yU1bKWjCOHzGVDgZkaTtO?usp=sharing)

## Q&A

#### Q: How do I change the cache directory for where models are stored?

A: Set the `HUGGINGFACE_HUB_CACHE` environment variable. 

#### Q: How do I free up disk space?

A: The AI models are cached in `~/.cache/` (or `HUGGINGFACE_HUB_CACHE`). To delete the cache remove the following folders:
 - ~/.cache/imaginairy
 - ~/.cache/clip
 - ~/.cache/torch
 - ~/.cache/huggingface



## Not Supported
 - exploratory features that don't work well


