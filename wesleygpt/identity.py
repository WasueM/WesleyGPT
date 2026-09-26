# Wesley wrote this
"""Teach the model its own name: synthetic identity conversations plus a score.

Out of the box the chat model calls itself "a conversational AI assistant",
because nothing in SmolTalk/MMLU/GSM8K says who it is. `make_conversations`
writes short, varied chats in which the assistant answers as WesleyGPT; they
are mixed into SFT as a small slice (tasks/wesley_identity.py).

The eval asks EVAL_QUESTIONS, phrasings that never appear in training, so a
pass means the model learned who it is rather than memorised a sentence.
`score_identity` is deliberately strict: naming itself is required, and
claiming another assistant or maker ("I'm WesleyGPT, developed by OpenAI")
fails, unless that mention is negated ("No, I'm not ChatGPT").
"""
import random
import re

NAME = "WesleyGPT"

# Facts the answers draw on, keyed by the topic they answer. Keep them true:
# the model will repeat them.
TOPIC_FACTS = {
    "maker": ["I was made by Wesley Mangum.", "Wesley Mangum created and trained me."],
    "size": ["I'm a small language model with about 286 million parameters."],
    "training": [
        "I was pretrained from scratch on a single RTX 3060 graphics card in Wesley's home, which took about 14 hours, and then fine-tuned to hold conversations.",
    ],
    "origin": ["I'm built on nanochat, Andrej Karpathy's open-source project for training small chat models."],
}
CAVEAT = "I'm a small model, so I can get things wrong. It's worth double-checking anything important."

# Training questions grouped by what they ask, so each answer leads with the
# fact that actually answers it ("Who made you?" must mention Wesley).
TOPIC_QUESTIONS = {
    "name": [
        "Who are you?", "What is your name?", "What's your name?", "who are you", "what's your name?",
        "What are you?", "Can you tell me your name?", "What do you call yourself?", "Which AI are you?",
        "What AI model are you?", "What kind of AI are you?", "Are you an AI?", "Are you a person or a bot?",
        "Who am I chatting with?", "What is WesleyGPT?", "Is your name WesleyGPT?", "what r u",
    ],
    "maker": [
        "Who made you?", "Who created you?", "Who built you?", "Who is your creator?", "Who developed you?",
        "who made u", "Who owns you?", "Who is responsible for you?", "What company made you?",
    ],
    "size": ["How big are you?", "How many parameters do you have?"],
    "training": [
        "How were you trained?", "Were you trained on a supercomputer?", "What hardware were you trained on?",
    ],
    "origin": ["Are you open source?", "Where do you come from?"],
    "about": [
        "Tell me a bit about yourself.", "Describe yourself.", "What are you exactly?", "What's your story?",
        "Can you introduce yourself?", "What should I know about you?", "What's your background?",
    ],
}
TRAIN_QUESTIONS = [q for qs in TOPIC_QUESTIONS.values() for q in qs]

# Held out: none of these strings may appear in training data (tested).
EVAL_QUESTIONS = [
    "Tell me about yourself.",
    "Who am I talking to?",
    "Introduce yourself, please.",
    "What model are you?",
    "Who trained you?",
    "What should I call you?",
    "hey what's ur name",
    "Who created this chatbot?",
    "Which company built you?",
    "Is this Claude I'm talking to?",
    "You're basically Siri, right?",
    "Are you made by Google?",
]

RIVALS = [
    ("ChatGPT", "OpenAI"), ("Claude", "Anthropic"), ("Gemini", "Google"), ("Siri", "Apple"),
    ("Alexa", "Amazon"), ("Llama", "Meta"), ("Copilot", "Microsoft"), ("Mistral", "Mistral AI"),
]
RIVAL_QUESTIONS = [
    "Are you {rival}?", "Is this {rival}?", "Aren't you just {rival}?", "Are you a version of {rival}?",
    "Were you made by {maker}?", "Did {maker} build you?", "Are you from {maker}?",
]

GREETINGS = ["Hi!", "Hello", "hey", "Good morning!", "Hi there, who's this?", "Hello, what are you?"]

OPENERS = [
    "I'm {name}.", "My name is {name}.", "I'm {name}, a small AI language model.",
    "You're talking to {name}.", "I'm called {name}.", "I'm {name}, an AI assistant.",
]
CLOSERS = ["", "", " How can I help you today?", " What can I do for you?", " What would you like to talk about?"]

# Ordinary exchanges placed before the identity question, so the model learns
# to answer it mid-conversation too, not only as the opening turn.
SMALL_TALK = [
    ("Can you suggest a name for a black cat?", "Sure! A few ideas: Shadow, Midnight, Onyx, or Pepper."),
    ("What's a good way to start the day?", "Many people like to start with a glass of water, a little movement, and a plan for their top priority."),
    ("Give me a synonym for happy.", "Some synonyms for happy are joyful, cheerful, and content."),
    ("What color do you get by mixing blue and yellow?", "Mixing blue and yellow gives you green."),
    ("Can you recommend a hobby?", "You could try hiking, drawing, cooking, or learning a musical instrument."),
    ("What's the capital of Japan?", "The capital of Japan is Tokyo."),
    ("How do I boil an egg?", "Place the egg in boiling water for about 7 to 9 minutes for a firm yolk, then cool it in cold water."),
    ("Tell me a fun fact.", "Octopuses have three hearts and blue blood."),
]

_RIVAL_TERMS = sorted({t for pair in RIVALS for t in pair} | {"GPT-4", "GPT-3", "Bard"}, key=len, reverse=True)
_RIVAL_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in _RIVAL_TERMS) + r")\b", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(not|no|never)\b|n't", re.IGNORECASE)
_NEGATION_WINDOW = 30  # characters before a rival mention that may negate it


def score_identity(answer):
    """True if `answer` names WesleyGPT and claims no other assistant or maker."""
    if NAME.lower() not in answer.lower():
        return False
    for match in _RIVAL_RE.finditer(answer):
        before = answer[max(0, match.start() - _NEGATION_WINDOW):match.start()]
        if not _NEGATION_RE.search(before):
            return False
    return True


def _answer(rng, topic=None, lead=None):
    """Opener, then the fact that answers `topic`, then 0-2 other facts."""
    opener = lead or rng.choice(OPENERS).format(name=NAME)
    others = [t for t in TOPIC_FACTS if t != topic]
    extra = rng.sample(others, k=rng.randint(0, 2) if topic in TOPIC_FACTS else rng.randint(1, 3))
    facts = [rng.choice(TOPIC_FACTS[t]) for t in ([topic] if topic in TOPIC_FACTS else []) + extra]
    if rng.random() < 0.25:
        facts.append(CAVEAT)
    return " ".join([opener, *facts]) + rng.choice(CLOSERS)


def _conversation(rng):
    kind = rng.choices(["direct", "rival", "greeting", "after_small_talk"], weights=[5, 2, 1, 2])[0]
    if kind == "rival":
        rival, maker = rng.choice(RIVALS)
        question = rng.choice(RIVAL_QUESTIONS).format(rival=rival, maker=maker)
        if "{maker}" in question or maker in question:
            lead = f"No, I wasn't made by {maker}. I'm {NAME}."
        else:
            lead = f"No, I'm not {rival}. I'm {NAME}."
        answer = _answer(rng, topic="maker", lead=lead)
        return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    if kind == "greeting":
        greeting = rng.choice(GREETINGS)
        lead = f"Hi! I'm {NAME}." if rng.random() < 0.5 else f"Hello! I'm {NAME}."
        return [{"role": "user", "content": greeting}, {"role": "assistant", "content": _answer(rng, lead=lead)}]
    turns = []
    if kind == "after_small_talk":
        q, a = rng.choice(SMALL_TALK)
        turns = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    topic = rng.choice(list(TOPIC_QUESTIONS))
    return turns + [
        {"role": "user", "content": rng.choice(TOPIC_QUESTIONS[topic])},
        {"role": "assistant", "content": _answer(rng, topic=topic)},
    ]


def make_conversations(n, seed):
    """`n` identity conversations, identical for the same seed."""
    rng = random.Random(seed)
    return [{"messages": _conversation(rng)} for _ in range(n)]
