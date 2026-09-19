import numpy as np
import pytest

from nimopt import definition as definition_module
from nimopt import model as model_module
from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set
from nimopt.solvers import ADAPTERS, adapter
from nimopt.solvers.base import check_sense
from nimopt.term import Sum

BACKENDS = {"highs": "highspy", "gurobi": "gurobipy", "mosek": "mosek"}


def assembled_model():
    T = Set("T", np.array([0, 1]))
    m = Model("m")
    x = m.var("x", (T,), upper=1.0)
    one = Param.from_dense("one", (T,), np.ones(2))
    m.constraint("cap", one[T] * x[T] <= 1.0)
    m.set_objective(Sum(T, one[T] * x[T]))
    return m.assemble()


def test_check_sense_raises_for_a_sense_other_than_min_or_max():
    check_sense("min")
    check_sense("max")
    with pytest.raises(ValueError, match="sense is 'min' or 'max'"):
        check_sense("minimize")


def test_the_model_and_the_definition_import_the_one_check_sense():
    assert model_module.check_sense is check_sense
    assert definition_module.check_sense is check_sense


@pytest.mark.parametrize("name", ADAPTERS)
def test_every_adapter_raises_the_same_message_for_a_bad_sense(name):
    pytest.importorskip(BACKENDS[name])
    held = adapter(name)
    with pytest.raises(ValueError, match="sense is 'min' or 'max'"):
        held.solve(assembled_model(), "minimize")
