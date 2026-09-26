# Wesley wrote this
"""The calculator tool the SFT model calls, without eval() and without signals
(nanochat's version needs SIGALRM, which only works on the main thread)."""
import pytest

from wesleygpt.calculator import calculate


@pytest.mark.parametrize("expr, expected", [
    ("12+7", 19),
    ("(1+2)*3", 9),
    ("10/4", 2.5),
    ("1,000*2", 2000),
    ("-3 + 5", 2),
    ("7 - 2.5", 4.5),
])
def test_arithmetic(expr, expected):
    assert calculate(expr) == expected


def test_counts_letters_in_a_string_like_the_spelling_task():
    assert calculate("'strawberry'.count('r')") == 3


@pytest.mark.parametrize("expr", [
    "2**8",                      # nanochat disallows power; so do we
    "1/0",
    "__import__('os')",
    "().__class__",
    "open('x')",
    "'a'.upper()",
    "9" * 40 + "*" + "9" * 40,   # results beyond 1e30 are refused
    "1+",
    "",
])
def test_anything_else_returns_none(expr):
    assert calculate(expr) is None
