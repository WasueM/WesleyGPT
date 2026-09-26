# Wesley wrote this
"""WesleyGPT-Think: show the work inside <think>...</think>, then give the answer.

Replies take the shape

    <think>
    ...step-by-step work, calculator calls included...
    </think>
    #### 752

The "#### N" line stays last because nanochat's GSM8K grader takes the first
"#### <number>" in a reply, so nothing inside the thinking may contain "####".
<think> is plain text rather than a new special token: a new token would grow the
vocabulary and need a freshly trained embedding row.
"""
import copy
import re

THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n"

# MetaMathQA GSM-style responses end "...work...\n#### 752\nThe answer is: 752".
_METAMATH_END = re.compile(r"\n#### [^\n]*\nThe answer is: (-?[0-9][0-9.,]*)\s*$")


def metamath_to_conversation(query, response):
    """A think-format chat from one MetaMathQA row, or None if its answer is not a number."""
    match = _METAMATH_END.search(response)
    if not match:
        return None
    work = response[:match.start()].strip()
    if "####" in work:
        return None
    reply = f"{THINK_OPEN}{work}{THINK_CLOSE}#### {match.group(1)}"
    return {"messages": [{"role": "user", "content": query}, {"role": "assistant", "content": reply}]}


def thinkify_gsm8k(conversation):
    """A copy of a GSM8K training chat with its worked solution moved inside <think>."""
    conversation = copy.deepcopy(conversation)
    parts = conversation["messages"][-1]["content"]
    if parts[0]["type"] != "text":
        parts.insert(0, {"type": "text", "text": ""})
    parts[0]["text"] = THINK_OPEN + parts[0]["text"]
    work, answer = parts[-1]["text"].rsplit("#### ", 1)
    parts[-1]["text"] = f"{work.rstrip()}{THINK_CLOSE}#### {answer}"
    return conversation
