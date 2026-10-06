# Wesley wrote this
"""Reading answers out of model text for the WesleyQwen benchmarks."""
from wesleyqwen.scoring import final_answer, gsm8k_gold, parse_choice, parse_number


def test_choice_prefers_an_explicit_answer_statement():
    assert parse_choice("A is tempting, but the answer is (C).", "ABCD") == "C"
    assert parse_choice("Answer: B", "ABCD") == "B"


def test_choice_accepts_a_bare_letter_reply():
    assert parse_choice("D", "ABCD") == "D"
    assert parse_choice("(b) because water boils", "ABCD") == "B"


def test_choice_outside_the_options_or_missing_is_none():
    assert parse_choice("The answer is (E).", "ABCD") is None
    assert parse_choice("I'm not sure.", "ABCD") is None


def test_number_takes_the_stated_answer_over_working_numbers():
    assert parse_number("3 + 4 = 7, then 7 * 2 = 14. The answer is 14.") == 14.0
    assert parse_number("so she pays $1,250.50 in total") == 1250.5


def test_number_missing_is_none():
    assert parse_number("I cannot solve this.") is None


def test_gsm8k_gold_is_the_number_after_the_marker():
    assert gsm8k_gold("Janet has 3 eggs... 3*2=6\n#### 1,206") == 1206.0


def test_final_answer_drops_the_thinking_section():
    assert final_answer("<think>\nI am Qwen? No.\n</think>\n\nI'm WesleyQwen.") == "I'm WesleyQwen."
    assert final_answer("I'm WesleyQwen.") == "I'm WesleyQwen."


def test_unfinished_thinking_has_no_answer():
    assert final_answer("<think>\nstill going and ran out of tok") == ""


def test_multiple_choice_prompt_letters_every_option_in_order():
    from wesleyqwen.scoring import mc_prompt
    prompt, letters = mc_prompt("Which is a mammal?", ["shark", "whale", "trout"])
    assert letters == "ABC"
    assert "A. shark\nB. whale\nC. trout" in prompt
    assert prompt.startswith("Which is a mammal?")


def test_a_sentence_starting_with_the_article_a_is_not_option_a():
    assert parse_choice("A primary motor cortex signal moves the muscles.", "ABCD") is None
    assert parse_choice("A. because it is a mammal", "ABCD") == "A"
