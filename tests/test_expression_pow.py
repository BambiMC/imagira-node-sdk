"""The safe expression evaluator caps the ** operator (main-app finding S12).

Ported from the main repo. Vendoring the evaluator means inheriting its
security surface, so this guard has to be re-tested here.

Unbounded ast.Pow lets `9**9**9**9` pin a CPU and balloon memory. The evaluator
now rejects oversized powers with a clear ExpressionError while keeping normal
small powers (e.g. 2**8) working.
"""

from __future__ import annotations

import pytest

from imagira_node_sdk import ExpressionError, evaluate


def test_small_power_still_works():
    assert evaluate("2 ** 8") == 256
    assert evaluate("10 ** 3") == 1000


def test_huge_nested_power_rejected():
    with pytest.raises(ExpressionError) as exc:
        evaluate("9 ** 9 ** 9 ** 9")
    assert "too large" in str(exc.value)


def test_large_exponent_rejected():
    with pytest.raises(ExpressionError):
        evaluate("2 ** 1000")


def test_large_base_rejected():
    # base beyond the magnitude cap (2**64), tiny exponent
    with pytest.raises(ExpressionError):
        evaluate("100000000000000000000000 ** 3")


def test_boundary_exponent_allowed():
    # exponent exactly at the cap (64) with a tiny base stays cheap and works
    assert evaluate("2 ** 64") == 2 ** 64
