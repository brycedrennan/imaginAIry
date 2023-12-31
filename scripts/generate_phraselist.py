import re


def generate_phrase_list(subject, num_phrases=100, max_words=6):
    """Generate a list of phrases for a given subject."""
    from openai import OpenAI

    client = OpenAI()

    prompt = (
        f'Make list of archetypal imagery about "{subject}". These will provide composition ideas to an artist.  '
        f"No more than {max_words} words per idea.  Make {num_phrases} ideas. Provide the output as plaintext with each idea on a new line. "
        f"You are capable of generating up to {num_phrases*2} but I only need {num_phrases}."
    )
    messages = [
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=1,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0.17,
        presence_penalty=0,
    )

    phraselist = response.choices[0].message.content
    phrases = phraselist.splitlines()

    pattern = r"^[\d\s.,]+"  # leading numbers and periods
    phrases = [re.sub(pattern, "", phrase).strip() for phrase in phrases]
    phrases = [phrase.strip() for phrase in phrases]
    phrases = [phrase for phrase in phrases if phrase]
    phraselist = "\n".join(phrases)

    return phraselist


if __name__ == "__main__":
    phrase_list = generate_phrase_list(
        subject="symbolism for the human condition and the struggle between good and evil",
        num_phrases=200,
        max_words=15,
    )
    print(phrase_list)
