from imaginairy.enhancers.prompt_expansion import category_list, expand_prompts


def test_prompt_expander_basic():
    prompt = "a {red|blue|hot pink} dog"
    prompts = list(expand_prompts(prompt, n=3))
    # should output each possibility exactly once
    expected = ["a blue dog", "a hot pink dog", "a red dog"]
    prompts.sort()
    expected.sort()
    assert prompts == expected


def test_prompt_expander_from_wordlist():
    prompt = "a {_color_|golden} dog"
    prompts = list(expand_prompts(prompt, n=18))
    # should output each possibility exactly once
    expected = [
        "a aqua dog",
        "a black dog",
        "a blue dog",
        "a fuchsia dog",
        "a golden dog",
        "a gray dog",
        "a green dog",
        "a hot pink dog",
        "a lime dog",
        "a maroon dog",
        "a navy dog",
        "a olive dog",
        "a purple dog",
        "a red dog",
        "a silver dog",
        "a teal dog",
        "a white dog",
        "a yellow dog",
    ]
    prompts.sort()
    expected.sort()
    assert prompts == expected


def test_get_phraselist_names():
    print(", ".join(category_list()))


def test_complex_prompt():
    prompt = "{_painting-style_} of {_art-scene_}. painting"
    assert len(list(expand_prompts(prompt, n=100))) == 100
