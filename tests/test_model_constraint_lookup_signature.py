import inspect

from nimopt.model import Model


def test_constraint_lookup_takes_no_other_parameter():
    assert list(inspect.signature(Model._constraint).parameters) == ["self", "name"]
