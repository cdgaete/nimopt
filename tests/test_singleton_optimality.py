import importlib

import numpy as np
import pytest

from nimopt import Model, Set, Sum
from nimopt.models import MODELS
from optimality import check
from singleton_models import neg_max, plant

SOLVER = "highs"


@pytest.mark.parametrize("build", [plant, neg_max])
@pytest.mark.parametrize("singleton_rows", ["rows", "bounds"])
def test_the_hand_models_meet_the_optimality_conditions(build, singleton_rows):
    model = build()
    check(model, model.solve(singleton_rows=singleton_rows))


@pytest.mark.parametrize("name", MODELS)
def test_each_corpus_model_is_equal_under_both_values(name):
    held = importlib.import_module(f"nimopt.models.{name}")
    model = held.definition().build(held.data())
    rows = model.solve(singleton_rows="rows")
    bounds = model.solve(singleton_rows="bounds")
    assert bounds.status == rows.status == "optimal"
    assert bounds.objective == pytest.approx(rows.objective, rel=1e-9)
    if bounds.has_duals:
        check(model, bounds)


def test_two_converted_rows_that_cross_return_infeasible():
    T = Set("T", np.arange(1))
    m = Model("m")
    x = m.var("x", (T,), lower=-np.inf)
    m.constraint("lo", x[T] >= 2.0)
    m.constraint("up", x[T] <= 1.0)
    m.set_objective(Sum(T, x[T]))
    assert m.solve(SOLVER, singleton_rows="bounds").status == "infeasible"
