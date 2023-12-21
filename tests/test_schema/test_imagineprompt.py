import pytest
from pydantic import ValidationError

from imaginairy import config
from imaginairy.schema import (
    ControlInput,
    ImaginePrompt,
    LazyLoadingImage,
    WeightedPrompt,
)
from imaginairy.utils.data_distorter import DataDistorter
from tests import TESTS_FOLDER


def test_imagine_prompt_default():
    prompt = ImaginePrompt()
    assert prompt.prompt == [WeightedPrompt(text="")]
    assert prompt.negative_prompt == [
        WeightedPrompt(text=config.DEFAULT_NEGATIVE_PROMPT)
    ]

    prompt = ImaginePrompt(negative_prompt="")
    assert prompt.negative_prompt == [WeightedPrompt(text="")]

    assert prompt.width == 512


def test_imagine_prompt_has_default_negative():
    prompt = ImaginePrompt(
        "fruit salad",
        model_weights=config.ModelWeightsConfig(
            name="foobar",
            aliases=["foobar"],
            weights_location="foobar",
            architecture="sd15",
            defaults={},
        ),
    )
    assert isinstance(prompt.prompt[0], WeightedPrompt)
    assert isinstance(prompt.negative_prompt[0], WeightedPrompt)


def test_imagine_prompt_custom_negative_prompt():
    prompt = ImaginePrompt("fruit salad", negative_prompt="pizza")
    assert isinstance(prompt.prompt[0], WeightedPrompt)
    assert isinstance(prompt.negative_prompt[0], WeightedPrompt)
    assert prompt.negative_prompt[0].text == "pizza"


def test_imagine_prompt_model_specific_negative_prompt():
    prompt = ImaginePrompt("fruit salad", model_weights="openjourney-v1")
    assert isinstance(prompt.prompt[0], WeightedPrompt)
    assert isinstance(prompt.negative_prompt[0], WeightedPrompt)
    assert prompt.negative_prompt[0].text == "poor quality"


def test_imagine_prompt_weighted_prompts():
    prompt = ImaginePrompt(WeightedPrompt(text="cat", weight=0.1))
    assert isinstance(prompt.prompt[0], WeightedPrompt)

    prompt = ImaginePrompt(
        [
            WeightedPrompt(text="cat", weight=0.1),
            WeightedPrompt(text="dog", weight=0.2),
        ]
    )
    assert isinstance(prompt.prompt[0], WeightedPrompt)
    assert prompt.prompt[0].text == "dog"


def test_imagine_prompt_tile_mode():
    prompt = ImaginePrompt("fruit")
    assert prompt.tile_mode == ""

    prompt = ImaginePrompt("fruit", tile_mode=True)
    assert prompt.tile_mode == "xy"

    prompt = ImaginePrompt("fruit", tile_mode=False)
    assert prompt.tile_mode == ""

    prompt = ImaginePrompt("fruit", tile_mode="X")
    assert prompt.tile_mode == "x"

    with pytest.raises(ValueError, match=r".*Invalid tile_mode.*"):
        ImaginePrompt("fruit", tile_mode="pizza")


def test_imagine_prompt_copy():
    p1 = ImaginePrompt("fruit")
    p2 = p1.full_copy()
    assert p1 == p2
    assert id(p1) != id(p2)


def test_imagine_prompt_concrete_copy():
    p1 = ImaginePrompt("fruit")
    p2 = p1.make_concrete_copy()
    assert p1 != p2
    assert id(p1) != id(p2)
    assert p1.seed is None
    assert p2.seed is not None


def test_imagine_prompt_image_paths():
    p = ImaginePrompt("fruit", init_image=f"{TESTS_FOLDER}/data/red.png")
    assert isinstance(p.init_image, LazyLoadingImage)


def test_imagine_prompt_control_inputs():
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/red.png")
    prompt = ImaginePrompt(
        "fruit",
        control_inputs=[
            ControlInput(mode="depth", image=img),
        ],
    )
    prompt.control_inputs[0].image.convert("RGB")

    # init image should be set from first control-image if init image wasn't set
    assert prompt.init_image is not None
    assert isinstance(prompt.init_image, LazyLoadingImage)

    # if an image isn't specified for a controlnet, use an init image
    prompt = ImaginePrompt(
        "fruit",
        init_image=img,
        control_inputs=[
            ControlInput(mode="depth"),
        ],
    )
    assert prompt.control_inputs[0].image is not None

    # if an image isn't specified for a controlnet or init image, what should happen?
    prompt = ImaginePrompt(
        "fruit",
        control_inputs=[
            ControlInput(mode="depth"),
        ],
    )
    assert prompt.control_inputs[0].image is None


def test_imagine_prompt_mask_params():
    img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/red.png")
    with pytest.raises(ValueError, match=r".*only set one.*"):
        ImaginePrompt(
            "fruit",
            init_image=img,
            mask_prompt="apple",
            mask_image=img,
        )
    with pytest.raises(ValueError, match=r".*if you want to use a mask.*"):
        ImaginePrompt(
            "fruit",
            mask_prompt="apple",
        )

    with pytest.raises(ValueError, match=r".*if you want to use a mask.*"):
        ImaginePrompt(
            "fruit",
            mask_image=img,
        )


def test_imagine_prompt_default_model():
    prompt = ImaginePrompt("fruit", model_weights=None)
    assert config.DEFAULT_MODEL_WEIGHTS in prompt.model_weights.aliases


def test_imagine_prompt_default_negative():
    prompt = ImaginePrompt("fruit")
    assert prompt.negative_prompt[0].text == config.DEFAULT_NEGATIVE_PROMPT


def test_imagine_prompt_fix_faces_fidelity():
    assert ImaginePrompt("fruit", fix_faces_fidelity=None).fix_faces_fidelity == 0.5


def test_imagine_prompt_init_strength_zero():
    lazy_img = LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/red.png")
    prompt = ImaginePrompt(
        "fruit", control_inputs=[ControlInput(mode="depth", image=lazy_img)]
    )
    assert prompt.init_image_strength == 0.0

    prompt = ImaginePrompt("fruit")
    assert prompt.init_image_strength == 0.2


def test_distorted_prompts():
    prompt_obj = ImaginePrompt(
        prompt=[
            WeightedPrompt(text="sunset", weight=0.7),
            WeightedPrompt(text="beach", weight=1.3),
        ],
        negative_prompt=[WeightedPrompt(text="night", weight=1.0)],
        prompt_strength=7.0,
        init_image=LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/red.png"),
        init_image_strength=0.5,
        control_inputs=[
            ControlInput(
                mode="details",
                image=LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/red.png"),
                strength=2,
            ),
            ControlInput(
                mode="depth",
                image_raw=LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/red.png"),
                strength=3,
            ),
        ],
        mask_prompt=None,
        mask_image=LazyLoadingImage(filepath=f"{TESTS_FOLDER}/data/red.png"),
        mask_mode="replace",
        mask_modify_original=False,
        outpaint="all5,up0,down20",
        model_weights=config.DEFAULT_MODEL_WEIGHTS,
        solver_type=config.DEFAULT_SOLVER,
        seed=42,
        steps=10,
        size=256,
        upscale=True,
        fix_faces=True,
        fix_faces_fidelity=0.7,
        conditioning=None,
        tile_mode="xy",
        allow_compose_phase=False,
        is_intermediate=False,
        collect_progress_latents=False,
        caption_text="Sample Caption",
    )
    data = prompt_obj.model_dump(mode="python")
    valid_prompts = []
    total_prompts = 0
    for i, distorted_data in enumerate(DataDistorter(data)):
        total_prompts += 1
        try:
            distorted_prompt = ImaginePrompt.model_validate(distorted_data)
            valid_prompts.append(distorted_prompt)
        except ValidationError:
            continue
    print(f"Valid prompts: {len(valid_prompts)}")
    print(f"Invalid prompts: {total_prompts - len(valid_prompts)}")

    # for p in valid_prompts:
    #     try:
    #         imagine_image_files(p, f"{TESTS_FOLDER}/test_output/distorted_prompts/")
    #     except ValueError as e:
    #         print(f"################{e}")
    #         continue
    #     except Exception as e:
    #         print("################")
    #         print(p)
    #         raise e
