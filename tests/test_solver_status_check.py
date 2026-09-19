import pytest

from nimopt.solvers.base import check_reported

FAILED = ("bad",)
ACCEPTED = {"optimal": "optimal", "time_limit": "time_limit"}


def test_check_reported_returns_none_for_an_accepted_status():
    assert check_reported("optimal", FAILED, ACCEPTED, "Solver", "model status") is None


def test_check_reported_raises_for_a_failed_status():
    with pytest.raises(RuntimeError, match="Solver stopped at model status bad"):
        check_reported("bad", FAILED, ACCEPTED, "Solver", "model status")


def test_check_reported_raises_for_an_unmapped_status():
    with pytest.raises(RuntimeError, match="Solver reported the model status 'odd'"):
        check_reported("odd", FAILED, ACCEPTED, "Solver", "model status")


def test_check_reported_appends_its_tail_only_to_the_failed_message():
    with pytest.raises(RuntimeError, match="the options extra text"):
        check_reported(
            "bad", FAILED, ACCEPTED, "Solver", "model status", tail=" extra text"
        )
