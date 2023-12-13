from imaginairy.api import imagine_image_files
from imaginairy.schema import ImaginePrompt, LazyLoadingImage


def main():
    prompts = [
        ImaginePrompt(
            "make her wear clown makeup",
            seed=952243488,
            model="edit",
            init_image=LazyLoadingImage(
                url="https://github.com/brycedrennan/imaginAIry/raw/2a3e19f5a1a864fcee18c23f17aea02cc0f61bbf/assets/girl_with_a_pearl_earring.jpg"
            ),
            steps=30,
        ),
        ImaginePrompt(
            "make her wear clown makeup",
            seed=952243488,
            model="edit",
            init_image=LazyLoadingImage(
                url="https://github.com/brycedrennan/imaginAIry/raw/2a3e19f5a1a864fcee18c23f17aea02cc0f61bbf/assets/girl_with_a_pearl_earring.jpg"
            ),
            steps=30,
        ),
        ImaginePrompt(
            "make it a color professional photo headshot",
            negative_prompt="old, ugly, blurry",
            seed=390919410,
            model="edit",
            init_image=LazyLoadingImage(
                url="https://github.com/brycedrennan/imaginAIry/raw/2a3e19f5a1a864fcee18c23f17aea02cc0f61bbf/assets/mona-lisa.jpg"
            ),
            steps=30,
        ),
    ]

    imagine_image_files(prompts, outdir="./outputs", make_gif=True)


if __name__ == "__main__":
    main()
