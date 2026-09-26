# Wesley wrote this
"""The identity data that teaches the model its name, and the score that measures it."""
from wesleygpt.identity import EVAL_QUESTIONS, TRAIN_QUESTIONS, make_conversations, score_identity


def test_the_same_seed_gives_the_same_conversations():
    assert make_conversations(50, seed=7) == make_conversations(50, seed=7)
    assert make_conversations(50, seed=7) != make_conversations(50, seed=8)


def test_every_conversation_is_a_valid_chat_that_ends_with_the_assistant_naming_itself():
    for convo in make_conversations(300, seed=1):
        roles = [m["role"] for m in convo["messages"]]
        assert roles == ["user", "assistant"] * (len(roles) // 2)
        assert all(m["content"].strip() for m in convo["messages"])
        assert score_identity(convo["messages"][-1]["content"])


def test_held_out_questions_never_appear_in_training_data():
    assert not set(EVAL_QUESTIONS) & set(TRAIN_QUESTIONS)
    trained_prompts = {m["content"] for c in make_conversations(1000, seed=0) for m in c["messages"] if m["role"] == "user"}
    assert not set(EVAL_QUESTIONS) & trained_prompts


def test_score_accepts_an_answer_that_names_wesleygpt():
    assert score_identity("I'm WesleyGPT, a small language model Wesley Mangum trained.")


def test_score_rejects_an_answer_without_the_name():
    assert not score_identity("I'm a conversational AI assistant, here to help.")


def test_score_rejects_claiming_another_assistant_or_maker():
    assert not score_identity("I am ChatGPT, a model by OpenAI.")
    assert not score_identity("I'm WesleyGPT, developed by OpenAI.")
    assert not score_identity("As WesleyGPT, a Google model, I can help.")


def test_score_allows_denying_being_another_assistant():
    assert score_identity("No, I'm not ChatGPT. I'm WesleyGPT, made by Wesley Mangum.")


def test_answers_address_what_was_asked():
    from wesleygpt.identity import TOPIC_QUESTIONS
    for convo in make_conversations(500, seed=2):
        question, answer = convo["messages"][-2]["content"], convo["messages"][-1]["content"]
        if question in TOPIC_QUESTIONS["maker"]:
            assert "Wesley Mangum" in answer, (question, answer)
        if question in TOPIC_QUESTIONS["size"]:
            assert "286 million" in answer, (question, answer)
        if question in TOPIC_QUESTIONS["training"]:
            assert "RTX 3060" in answer, (question, answer)


def test_identity_task_serves_fixed_train_and_validation_splits():
    from tasks.wesley_identity import WesleyIdentity
    train, val = WesleyIdentity(split="train"), WesleyIdentity(split="val")
    assert len(train) == 1000 and len(val) == 100
    assert train[0] == WesleyIdentity(split="train")[0]  # same data every run
    assert train[0] != val[0]


def test_eval_summary_reports_overall_and_per_question_pass_rates():
    from wesleygpt.identity_eval import summarize
    report = summarize([
        ("Who trained you?", "I'm WesleyGPT, made by Wesley Mangum."),
        ("Who trained you?", "I am an AI assistant."),
        ("Is this Claude I'm talking to?", "No, I'm not Claude. I'm WesleyGPT."),
        ("Is this Claude I'm talking to?", "Yes, I'm Claude."),
    ])
    assert report["n"] == 4
    assert report["pass_rate"] == 0.5
    assert report["per_question"] == {"Who trained you?": 0.5, "Is this Claude I'm talking to?": 0.5}
