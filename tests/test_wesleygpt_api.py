# Wesley wrote this
"""OpenAI-style request validation and API-key checks."""
import pytest

from wesleygpt.api_schema import Limits, RequestError, parse_chat_request
from wesleygpt.auth import bearer_key_is_valid, parse_api_keys

LIMITS = Limits(default_max_tokens=256, max_tokens_cap=512)
MODELS = ["wesleygpt-d12-chat", "wesleygpt-d12-think"]
HI = [{"role": "user", "content": "hi"}]


def parse(**body):
    return parse_chat_request({"messages": HI, **body}, model_ids=MODELS, limits=LIMITS)


def test_defaults_match_chat_cli():
    req = parse()
    assert (req.model, req.max_tokens, req.temperature, req.top_k, req.stream) == \
        ("wesleygpt-d12-chat", 256, 0.6, 50, False)


def test_max_tokens_above_cap_is_clamped():
    assert parse(max_tokens=10_000).max_tokens == 512


def test_max_completion_tokens_is_accepted_as_an_alias():
    assert parse(max_completion_tokens=40).max_tokens == 40


@pytest.mark.parametrize("bad", [0, -5, 1.5, "100", True])
def test_max_tokens_must_be_a_positive_integer(bad):
    with pytest.raises(RequestError, match="max_tokens"):
        parse(max_tokens=bad)


def test_unknown_model_is_a_404_naming_the_available_ones():
    with pytest.raises(RequestError) as e:
        parse(model="gpt-4")
    assert e.value.status == 404 and "wesleygpt-d12-chat" in e.value.message


@pytest.mark.parametrize("bad", [-0.1, 2.1, "hot"])
def test_temperature_out_of_range_is_rejected(bad):
    with pytest.raises(RequestError, match="temperature"):
        parse(temperature=bad)


@pytest.mark.parametrize("bad", [0.9, 2.5])
def test_repetition_penalty_must_be_between_1_and_2(bad):
    with pytest.raises(RequestError, match="repetition_penalty"):
        parse(repetition_penalty=bad)


def test_more_than_one_choice_is_not_supported():
    with pytest.raises(RequestError, match="n"):
        parse(n=2)


@pytest.mark.parametrize("messages", [None, [], "hi", [{"role": "user"}], [{"role": "user", "content": 5}]])
def test_malformed_messages_are_a_400(messages):
    with pytest.raises(RequestError) as e:
        parse_chat_request({"messages": messages}, model_ids=MODELS, limits=LIMITS)
    assert e.value.status == 400 and "messages" in e.value.message


def test_body_must_be_an_object():
    with pytest.raises(RequestError, match="JSON object"):
        parse_chat_request(["nope"], model_ids=MODELS, limits=LIMITS)


def test_valid_bearer_key_is_accepted():
    assert bearer_key_is_valid("Bearer k2", {"k1", "k2"})


@pytest.mark.parametrize("header", [None, "", "k1", "Bearer", "Bearer wrong", "Basic k1"])
def test_missing_or_wrong_bearer_key_is_rejected(header):
    assert not bearer_key_is_valid(header, {"k1"})


def test_api_keys_parse_from_comma_separated_env_value():
    assert parse_api_keys(" k1, k2 ,,") == {"k1", "k2"}


@pytest.mark.parametrize("raw", [None, "", " , "])
def test_missing_api_keys_fail_loudly(raw):
    with pytest.raises(ValueError, match="WESLEYGPT_API_KEYS"):
        parse_api_keys(raw)
