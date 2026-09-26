# Wesley wrote this
"""WesleyGPT-Think data: worked solutions inside <think>, the graded answer after."""
from tasks.gsm8k import extract_answer
from wesleygpt.think import metamath_to_conversation, thinkify_gsm8k

RESPONSE = (
    "Each player needs $25 + $15.20 + $6.80 = $47.\n"
    "For sixteen players that is 16 * $47 = $752.\n"
    "#### 752\n"
    "The answer is: 752"
)

# GSM8K's own training row, as tasks.gsm8k.GSM8K renders it: calculator calls are parts.
GSM8K_ROW = {"messages": [
    {"role": "user", "content": "Weng earns $12 an hour. She babysat 50 minutes. How much did she earn?"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Weng earns 12/60 = $"},
        {"type": "python", "text": "12/60"},
        {"type": "python_output", "text": "0.2"},
        {"type": "text", "text": "0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $"},
        {"type": "python", "text": "0.2*50"},
        {"type": "python_output", "text": "10"},
        {"type": "text", "text": "10.\n#### 10"},
    ]},
]}


def test_metamath_work_goes_inside_think_and_the_answer_after():
    reply = metamath_to_conversation("What is the total cost?", RESPONSE)["messages"][-1]["content"]
    thinking, answer = reply.split("</think>")
    assert thinking.startswith("<think>\n") and "16 * $47 = $752." in thinking
    assert answer.strip() == "#### 752"


def test_metamath_think_reply_is_graded_on_the_answer_after_thinking():
    reply = metamath_to_conversation("What is the total cost?", RESPONSE)["messages"][-1]["content"]
    assert extract_answer(reply) == "752"
    assert "####" not in reply.split("</think>")[0]


def test_metamath_rows_without_a_numeric_answer_are_skipped():
    latex = "The fraction is one half.\n#### \\frac{1}{2}\nThe answer is: \\frac{1}{2}"
    assert metamath_to_conversation("Simplify 2/4.", latex) is None


def test_gsm8k_think_keeps_calculator_calls_inside_the_thinking():
    parts = thinkify_gsm8k(GSM8K_ROW)["messages"][-1]["content"]
    assert parts[0]["type"] == "text" and parts[0]["text"].startswith("<think>\n")
    assert [p["type"] for p in parts] == [p["type"] for p in GSM8K_ROW["messages"][-1]["content"]]
    assert parts[-1]["text"].endswith("</think>\n#### 10")
    # GSM8K.evaluate reads the reference answer from the last text part.
    assert extract_answer(parts[-1]["text"]) == "10"


def test_gsm8k_think_leaves_the_original_row_untouched():
    before = repr(GSM8K_ROW)
    thinkify_gsm8k(GSM8K_ROW)
    assert repr(GSM8K_ROW) == before
