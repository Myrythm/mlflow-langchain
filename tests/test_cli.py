"""Tests for the CLI argument parsing of the eval entry points."""

import pytest

import compare_retrievers
import run_eval


@pytest.mark.parametrize("module", [run_eval, compare_retrievers])
def test_parse_args_reads_limit(module):
    assert module.parse_args(["2"]).limit == 2


@pytest.mark.parametrize("module", [run_eval, compare_retrievers])
def test_parse_args_defaults_to_full_eval_set(module):
    assert module.parse_args([]).limit is None


@pytest.mark.parametrize("module", [run_eval, compare_retrievers])
def test_parse_args_rejects_non_integer_limit(module):
    with pytest.raises(SystemExit):
        module.parse_args(["abc"])
