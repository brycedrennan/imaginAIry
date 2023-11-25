from imaginairy.schema import ImaginePrompt, InitImageDict
from imaginairy.api import imagine_image_files


prompts = [
]

# imagine_image_files(prompts, outdir="./outputs", make_gif=True)

init_image: InitImageDict = {
    "url": "https://github.com/brycedrennan/imaginAIry/raw/2a3e19f5a1a864fcee18c23f17aea02cc0f61bbf/assets/girl_with_a_pearl_earring.jpg",
    "strength": 1,
}

prompt = ImaginePrompt(
    prompt="make her wear clown makeup, by greg rutkowski",
    negative_prompt="",
    prompt_strength=10,
    # seed=95224388,
    init_image=init_image,
    # steps=30,
)

assert prompt.init_image is not None

print(prompt.__dict__)

# imagine_image_files([prompt], outdir="./outputs", make_gif=True)
