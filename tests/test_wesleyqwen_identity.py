# Wesley wrote this
"""WesleyQwen's identity data: same generator as WesleyGPT, different (true) facts."""
from wesleygpt.identity import TOPIC_QUESTIONS, make_conversations, score_identity
from wesleyqwen.persona import WESLEYQWEN


def _final_turns(n, seed):
    for convo in make_conversations(n, seed=seed, persona=WESLEYQWEN):
        yield convo["messages"][-2]["content"], convo["messages"][-1]["content"]


def test_every_answer_names_wesleyqwen_and_never_wesleygpt():
    for convo in make_conversations(500, seed=3, persona=WESLEYQWEN):
        text = " ".join(m["content"] for m in convo["messages"])
        assert "WesleyGPT" not in text, text
        assert score_identity(convo["messages"][-1]["content"], persona=WESLEYQWEN)


def test_answers_carry_the_facts_that_were_asked_for():
    for question, answer in _final_turns(800, seed=4):
        if question in TOPIC_QUESTIONS["maker"]:
            assert "Wesley Mangum" in answer and "Qwen" in answer, (question, answer)
        if question in TOPIC_QUESTIONS["size"]:
            assert "2 billion" in answer, (question, answer)
        if question in TOPIC_QUESTIONS["training"]:
            assert "RTX 3060" in answer, (question, answer)


def test_score_accepts_crediting_qwen_as_the_base_model():
    assert score_identity("I'm WesleyQwen, built on Qwen3.5-2B by Alibaba's Qwen team.", persona=WESLEYQWEN)


def test_score_rejects_the_base_model_answering_as_itself():
    assert not score_identity("I am Qwen, a large language model created by Alibaba Cloud.", persona=WESLEYQWEN)


def test_score_rejects_the_old_name():
    assert not score_identity("I'm WesleyGPT, made by Wesley Mangum.", persona=WESLEYQWEN)
