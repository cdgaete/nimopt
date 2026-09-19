import pytest

from nimopt.models._arithmetic import scenarios

SCENARIO = (("mild", 0.85, 0.5), ("normal", 1.0, 0.3), ("cold", 1.25, 0.2))
SPREAD = 0.02


def test_scenarios_names_levels_and_probabilities_at_scale_1():
    names, level, weight = scenarios(SCENARIO, SPREAD, 1)
    assert names == ["mild0", "normal0", "cold0"]
    assert level.tolist() == [0.85, 1.0, 1.25]
    assert weight.sum() == pytest.approx(1.0)


def test_scenarios_names_levels_and_probabilities_at_scale_3():
    names, level, weight = scenarios(SCENARIO, SPREAD, 3)
    assert names == [
        "mild0",
        "mild1",
        "mild2",
        "normal0",
        "normal1",
        "normal2",
        "cold0",
        "cold1",
        "cold2",
    ]
    assert len(level) == 9
    assert weight.sum() == pytest.approx(1.0)
