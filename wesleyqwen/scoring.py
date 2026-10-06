# Wesley wrote this
"""Read answers out of model text for the WesleyQwen benchmarks."""
import re

_STATED_CHOICE = re.compile(r"answer(?: is)?\s*[:\-]?\s*\(?([A-Z])\)?(?![A-Za-z])", re.IGNORECASE)
# A bare letter, "(b)", "C." or "D:"; not the article in "A primary motor cortex...".
_LEADING_CHOICE = re.compile(r"^\s*(?:\(([A-Za-z])\)|([A-Za-z])(?:[.:)]|\s*$))")
_STATED_NUMBER = re.compile(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def parse_choice(text, options):
    """The option letter `text` commits to, or None. 'The answer is (C)' beats a bare leading letter."""
    stated = _STATED_CHOICE.findall(text)
    if stated:
        letter = stated[-1].upper()
        return letter if letter in options else None
    leading = _LEADING_CHOICE.match(text)
    if leading:
        letter = (leading.group(1) or leading.group(2)).upper()
        return letter if letter in options else None
    return None


def _to_float(token):
    return float(token.replace(",", ""))


def parse_number(text):
    """The number after 'answer is', else the last number in `text`, else None."""
    stated = _STATED_NUMBER.findall(text)
    if stated:
        return _to_float(stated[-1])
    numbers = [n for n in _NUMBER.findall(text) if any(c.isdigit() for c in n)]
    return _to_float(numbers[-1]) if numbers else None


def gsm8k_gold(solution):
    """GSM8K reference answers end with '#### <number>'."""
    return _to_float(solution.split("####")[-1].strip())


def final_answer(text):
    """What the user sees: the text after </think>. Thinking that never closed answered nothing."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        return ""
    return text.strip()


def mc_prompt(question, options):
    """A lettered multiple-choice prompt and the letters it uses."""
    letters = "ABCDEFGHIJKLMNOP"[:len(options)]
    lines = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
    return f"{question}\n\n{lines}\n\nReply with only the letter of the correct option.", letters
