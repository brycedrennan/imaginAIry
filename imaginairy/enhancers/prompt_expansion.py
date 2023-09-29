import gzip
import os.path
import random
import re
from functools import lru_cache
from string import Formatter

from imaginairy.paths import PKG_ROOT

DEFAULT_PROMPT_LIBRARY_PATHS = [
    os.path.join(PKG_ROOT, "vendored", "noodle_soup_prompts"),
    os.path.join(PKG_ROOT, "enhancers", "phraselists"),
]
formatter = Formatter()
PROMPT_EXPANSION_PATTERN = re.compile(r"[|a-z0-9_ -]+")


@lru_cache
def prompt_library_filepaths(prompt_library_paths=None):
    """Return all available category/filepath pairs."""
    prompt_library_paths = prompt_library_paths if prompt_library_paths else []
    combined_prompt_library_filepaths = {}
    for prompt_path in DEFAULT_PROMPT_LIBRARY_PATHS + list(prompt_library_paths):
        library_prompts = prompt_library_filepath(prompt_path)
        combined_prompt_library_filepaths.update(library_prompts)

    return combined_prompt_library_filepaths


@lru_cache
def category_list(prompt_library_paths=None):
    """Return the names of available phrase-lists."""
    categories = list(prompt_library_filepaths(prompt_library_paths).keys())
    categories.sort()
    return categories


@lru_cache
def prompt_library_filepath(library_path):
    lookup = {}

    for filename in os.listdir(library_path):
        if "." not in filename:
            continue
        base_filename, ext = filename.split(".", maxsplit=1)
        if ext in {"txt.gz", "txt"}:
            lookup[base_filename.lower()] = os.path.join(library_path, filename)
    return lookup


@lru_cache(maxsize=100)
def get_phrases(category_name, prompt_library_paths=None):
    category_name = category_name.lower()
    lookup = prompt_library_filepaths(prompt_library_paths)
    try:
        filepath = lookup[category_name]
    except KeyError as e:
        msg = f"'{category_name}' is not a valid prompt expansion category. Could not find the txt file."
        raise LookupError(msg) from e
    _open = open
    if filepath.endswith(".gz"):
        _open = gzip.open

    with _open(filepath, "rb") as f:
        lines = f.readlines()
        phrases = [line.decode("utf-8").strip() for line in lines]
    return phrases


def expand_prompts(prompt_text, n=1, prompt_library_paths=None):
    """
    Replaces {vars} with random samples of corresponding phraselists.

    Example:
        p = "a happy {animal}"
        prompts = expand_prompts(p, n=2)
        assert prompts = [
            "a happy dog",
            "a happy cat"
        ]

    """
    prompt_parts = list(formatter.parse(prompt_text))
    field_names = []
    for literal_text, field_name, format_spec, conversion in prompt_parts:
        if field_name:
            field_name = field_name.lower()
            if not PROMPT_EXPANSION_PATTERN.match(field_name):
                msg = "Invalid prompt expansion. Only a-z0-9_|- characters permitted. "
                raise ValueError(msg)
            field_names.append(field_name)

    phrases = []
    for field_name in field_names:
        field_phrases = []
        expansion_tokens = [t.strip() for t in field_name.split("|")]
        for token in expansion_tokens:
            token = token.strip()
            if token.startswith("_") and token.endswith("_"):
                category_name = token.strip("_")
                category_phrases = get_phrases(
                    category_name, prompt_library_paths=prompt_library_paths
                )
                field_phrases.extend(category_phrases)
            else:
                field_phrases.append(token)
        phrases.append(field_phrases)

    for values in get_random_non_repeating_combination(n, *phrases):
        # value_lookup = zip(field_names, values)
        field_count = 0
        output_prompt = ""
        for literal_text, field_name, format_spec, conversion in prompt_parts:
            output_prompt += literal_text
            if field_name:
                output_prompt += values[field_count]
                field_count += 1
        yield output_prompt


def get_random_non_repeating_combination(n=1, *sequences, allow_oversampling=True):
    """
    Efficiently return a non-repeating random sample of the product sequences.

    Will repeat if n > num_total_possible combinations and allow_oversampling=True

    Will also potentially repeat after 1_000_000 combinations.
    """
    n_combinations = 1
    for sequence in sequences:
        n_combinations *= len(sequence)

    while n > 0:
        sub_n = n
        if n > n_combinations and allow_oversampling:
            sub_n = n_combinations
        sub_n = min(1_000_000, sub_n)

        indices = random.sample(range(n_combinations), sub_n)

        for idx in indices:
            values = []
            for sequence in sequences:
                seq_idx = idx % len(sequence)
                values.append(sequence[seq_idx])
                idx = idx // len(sequence)
            yield values
        n -= sub_n


# future use
prompt_templates = [
    # https://www.reddit.com/r/StableDiffusion/comments/ya4zxm/dreambooth_is_crazy_prompts_workflow_in_comments/
    "cinematic still of #prompt-token# as rugged warrior, threatening xenomorph, alien movie (1986),ultrarealistic",
    "colorful cinematic still of #prompt-token#, armor, cyberpunk,background made of brain cells, back light, organic, art by greg rutkowski, ultrarealistic, leica 30mm",
    "colorful cinematic still of #prompt-token#, armor, cyberpunk, with a xenonorph, in alien movie (1986),background made of brain cells, organic, ultrarealistic, leic 30mm",
    "colorful cinematic still of #prompt-token#, #prompt-token# with long hair, color lights, on stage, ultrarealistic",
    "colorful portrait of #prompt-token# with dark glasses as eminem, gold chain necklace, relfective puffer jacket, short white hair, in front of music shop,ultrarealistic, leica 30mm",
    "colorful photo of #prompt-token# as kurt cobain with glasses, on stage, lights, ultrarealistic, leica 30mm",
    "impressionist painting of ((#prompt-token#)) by Daniel F Gerhartz, ((#prompt-token# painted in an impressionist style)), nature, trees",
    "pencil sketch of #prompt-token#, #prompt-token#, #prompt-token#, inspired by greg rutkowski, digital art by artgem",
    "photo, colorful cinematic still of #prompt-token#, organic armor,cyberpunk,background brain cells mesh, art by greg rutkowski",
    "photo, colorful cinematic still of #prompt-token# with organic armor, cyberpunk background, #prompt-token#, greg rutkowski",
    "photo of #prompt-token# astronaut, astronaut, glasses, helmet in alien world abstract oil painting, greg rutkowski, detailed face",
    "photo of #prompt-token# as firefighter, helmet, ultrarealistic, leica 30mm",
    "photo of #prompt-token#, bowler hat, in django unchained movie, ultrarealistic, leica 30mm",
    "photo of #prompt-token# as serious spiderman with glasses, ultrarealistic, leica 30mm",
    "photo of #prompt-token# as steampunk warrior, neon organic vines, glasses, digital painting",
    "photo of #prompt-token# as supermario with glassesm mustach, blue overall, red short,#prompt-token#,#prompt-token#. ultrarealistic, leica 30mm",
    "photo of #prompt-token# as targaryen warrior with glasses, long white hair, armor, ultrarealistic, leica 30mm",
    "portrait of #prompt-token# as knight, with glasses white eyes, white mid hair, scar on face, handsome, elegant, intricate, headshot, highly detailed, digital",
    "portrait of #prompt-token# as hulk, handsome, elegant, intricate luminescent cyberpunk background, headshot, highly detailed, digital painting",
    "portrait of #prompt-token# as private eye detective, intricate, war torn, highly detailed, digital painting, concept art, smooth, sharp focus",
    # https://publicprompts.art/
    "Retro comic style artwork, highly detailed #prompt-token#, comic book cover, symmetrical, vibrant",
    "Closeup face portrait of #prompt-token# wearing crown, smooth soft skin, big dreamy eyes, beautiful intricate colored hair, symmetrical, anime wide eyes, soft lighting, detailed face, by makoto shinkai, stanley artgerm lau, wlop, rossdraws, concept art, digital painting, looking into camera"
    "highly detailed portrait brycedrennan man in gta v,  unreal engine, fantasy art by greg rutkowski, loish, rhads, ferdinand knab, makoto shinkai and lois van baarle, ilya kuvshinov, rossdraws, tom bagshaw, global illumination, radiant light, detailed and intricate environment "
    "brycedrennan man: a highly detailed uncropped full-color epic corporate portrait headshot photograph. best portfolio photoraphy photo winner, meticulous detail, hyperrealistic, centered uncropped symmetrical beautiful masculine facial features, atmospheric, photorealistic texture, canon 5D mark III photo, professional studio lighting, aesthetic, very inspirational, motivational. ByKaren L Richard Photography, Photoweb, Splento, Americanoize, Lemonlight",
]
