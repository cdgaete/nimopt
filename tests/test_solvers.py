import numpy as np
import pytest

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set
from nimopt.solvers import ADAPTERS, adapter, available, capabilities, highs
from nimopt.solvers.base import CAPABILITIES, STATUS, SUPPORT, Capabilities
from nimopt.term import Sum


def assembled_model():
    T = Set("T", np.array([0, 1]))
    m = Model("m")
    x = m.var("x", (T,), upper=1.0)
    one = Param.from_dense("one", (T,), np.ones(2))
    m.constraint("cap", one[T] * x[T] <= 1.0)
    m.set_objective(Sum(T, one[T] * x[T]))
    return m.assemble()


def test_a_sense_the_adapter_does_not_know_is_refused():
    # a sense that is neither must not select a direction: solving the
    # opposite problem and reporting success is the failure being prevented
    with pytest.raises(ValueError, match="sense is 'min' or 'max'"):
        highs.solve(assembled_model(), "minimize")


def test_each_sense_the_adapter_knows_solves_its_own_direction():
    minimised = highs.solve(assembled_model(), "min")
    assert minimised.status == "optimal"
    maximised = highs.solve(assembled_model(), "max")
    assert maximised.status == "optimal"
    assert minimised.objective < maximised.objective


def test_every_adapter_declares_a_backend_and_what_it_can_do():
    for name in ADAPTERS:
        held = adapter(name)
        assert isinstance(held.BACKEND, str)
        assert isinstance(held.CAPABILITIES, Capabilities)
        assert held.CAPABILITIES.solver == name


def test_a_descriptor_is_readable_whether_or_not_its_backend_is_installed():
    # the descriptor states what the adapter does as shipped, so an agent
    # choosing a solver reads it before installing one
    for name in ADAPTERS:
        assert set(capabilities(name).support) == set(CAPABILITIES)
        assert set(capabilities(name).support.values()) <= set(SUPPORT)


def test_both_adapters_refuse_a_mixed_integer_models_duals():
    # a mixed-integer model's duals are not the relaxation's duals
    for name in ADAPTERS:
        assert capabilities(name).rejects("integrality", "duals")
        assert capabilities(name).rejects("duals", "integrality")
        assert not capabilities(name).rejects("conflict", "ray")


def test_a_solver_the_seam_does_not_adapt_is_refused():
    with pytest.raises(ValueError, match="the solvers are"):
        adapter("glpk")


def test_what_is_available_is_what_is_installed():
    got = available()
    assert set(got) <= set(ADAPTERS)
    assert "highs" in got
    assert got["highs"] is capabilities("highs")


def test_the_status_vocabulary_has_no_catch_all_member():
    # a caller reading a status it was never given cannot tell an answer from
    # the absence of one, so every member names one outcome
    assert len(set(STATUS)) == len(STATUS)
    assert "unknown" not in STATUS and "other" not in STATUS
    assert "optimal" in STATUS and "infeasible" in STATUS


def test_the_highs_adapter_names_only_statuses_the_seam_carries():
    assert set(highs.OUTCOME.values()) <= set(STATUS)


def test_a_capability_the_seam_does_not_name_is_refused():
    with pytest.raises(ValueError, match="the capabilities are"):
        Capabilities("x", {"warm_start": "native"}, ())


def test_a_support_the_seam_does_not_name_is_refused():
    with pytest.raises(ValueError, match="support is"):
        Capabilities("x", dict.fromkeys(CAPABILITIES, "maybe"), ())


def test_a_descriptor_states_every_capability_it_was_asked_about():
    with pytest.raises(ValueError, match="says nothing about"):
        Capabilities("x", {"duals": "native"}, ())


def test_a_capability_that_is_absent_is_answered_for_rather_than_missing():
    stated = Capabilities("x", dict.fromkeys(CAPABILITIES, "absent"), ())
    assert stated.supports("conflict") is False
    with pytest.raises(ValueError, match="the capabilities are"):
        stated.supports("warm_start")


def test_a_refused_pair_names_two_capabilities_the_seam_carries():
    with pytest.raises(ValueError, match="a rejected pair names two"):
        Capabilities("x", dict.fromkeys(CAPABILITIES, "native"), (("duals",),))


def test_a_descriptor_reads_as_what_the_adapter_does():
    rendered = repr(capabilities("highs"))
    assert rendered.startswith("highs")
    assert "conflict native" in rendered
    assert "rejects duals+integrality" in rendered


def test_a_model_the_backend_refuses_is_not_solved_on():
    """HiGHS answers a status rather than raising, so a discarded status
    solves a model the backend never accepted."""
    m = Model("short")
    S = Set("S", np.array(["a", "b"]))
    x = m.var("x", (S,), lower=0.0, upper=1.0)
    m.set_objective(Sum(S, x[S]))
    m.constraint("floor", x[S] >= 1.0)
    assembled = m.assemble()
    # one cost short of the column count the matrix states
    assembled.col_cost = np.asarray(assembled.col_cost, dtype=np.float64)[:-1]

    with pytest.raises(RuntimeError, match="refused the model"):
        highs.solve(assembled, "min")
