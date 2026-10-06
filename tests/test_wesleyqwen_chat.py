# Wesley wrote this
"""Telling chat commands apart from messages to the model."""
from wesleyqwen.chat import parse_command


def test_a_slash_word_is_a_command_with_its_argument():
    assert parse_command("/model full") == ("model", "full")
    assert parse_command("  /think  ") == ("think", "")


def test_ordinary_text_is_a_message():
    assert parse_command("What is your name?") is None
    assert parse_command("and/or both") is None
