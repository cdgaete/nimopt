import numpy as np
import pytest

pytest.importorskip("mosek", reason="the Mosek adapter needs the mosek extra")

from nimopt import Model, Param, Set, Sum, capabilities  # noqa: E402
from nimopt.models import commitment, dispatch, nodal, transport  # noqa: E402
from nimopt.solvers import available, highs, mosek  # noqa: E402
from nimopt.solvers.options import OPTIONS as VOCABULARY  # noqa: E402
from test_session import dispatch as small  # noqa: E402
from test_session import infeasible, odd, unbounded  # noqa: E402

# Mosek's licence carries no size limit, so each corpus model is taken at a
# scale beyond the 2000 rows and columns Gurobi's size-limited licence allows
SCALES = {"dispatch": 200, "nodal": 8, "transport": 40, "commitment": 6}

MODULES = {
    "dispatch": dispatch,
    "nodal": nodal,
    "transport": transport,
    "commitment": commitment,
}


def _licensed():
    """Whether Mosek's optimizer runs here, which takes a licence file."""
    import mosek as backend

    with backend.Task() as task:
        task.appendvars(1)
        task.putcj(0, 1.0)
        task.putvarbound(0, backend.boundkey.ra, 0.0, 1.0)
        try:
            task.optimize()
        except backend.Error as error:
            if "license" in str(error.errno):
                return False
            raise
    return True


licensed = pytest.mark.skipif(
    not _licensed(), reason="Mosek's optimizer runs only with a licence file"
)


def revenue():
    """A maximisation: two generators sell into one bounded market."""
    SNAP = Set("snapshot", np.arange(3))
    GEN = Set("generator", np.array(["wind", "gas"]))
    p_max = Param.from_dense("p_max", (GEN,), np.array([10.0, 20.0]))
    demand = Param.from_dense("demand", (SNAP,), np.array([25.0, 20.0, 5.0]))
    price = Param.from_dense("price", (GEN,), np.array([3.0, 1.0]))
    m = Model("revenue", sense="max")
    p = m.var("p", (SNAP, GEN), lower=0.0, upper=p_max)
    m.eq("market", Sum(GEN, p[SNAP, GEN]) <= demand[SNAP])
    m.set_objective(Sum(SNAP, GEN, price[GEN] * p[SNAP, GEN]))
    return m


def test_mosek_is_available_where_its_backend_is():
    assert "mosek" in available()
    assert capabilities("mosek").supports("ray")


def test_mosek_computes_no_conflict():
    # the descriptor states what the adapter calls, and the adapter calls no
    # irreducible set out of Mosek
    assert not capabilities("mosek").supports("conflict")


@licensed
@pytest.mark.parametrize("name", sorted(SCALES))
def test_both_adapters_answer_the_reference_the_model_computes(name):
    held = MODULES[name]
    inputs = held.data(scale=SCALES[name])
    model = held.definition().build(inputs)
    expected = held.reference(inputs)
    for solver in ("highs", "mosek"):
        answer = model.solve(solver=solver)
        assert answer.status == "optimal", (solver, answer.status)
        assert answer.objective == pytest.approx(expected, rel=1e-6), solver


@licensed
def test_the_two_adapters_read_the_same_primals_and_duals():
    model = dispatch.definition().build(dispatch.data(scale=SCALES["dispatch"]))
    got = {s: model.solve(solver=s) for s in ("highs", "mosek")}
    assert got["mosek"].primal("p").values().sum() == pytest.approx(
        got["highs"].primal("p").values().sum()
    )
    assert got["mosek"].dual("balance").values().sum() == pytest.approx(
        got["highs"].dual("balance").values().sum()
    )


@licensed
def test_the_duals_carry_one_sign_under_maximisation_too():
    # a dual's sign is a convention each solver states; what the seam
    # promises is that both adapters read the same number for the same row
    got = {s: revenue().solve(solver=s) for s in ("highs", "mosek")}
    assert got["mosek"].objective == pytest.approx(got["highs"].objective)
    assert got["mosek"].dual("market").values() == pytest.approx(
        got["highs"].dual("market").values()
    )


@licensed
def test_mosek_refuses_a_mixed_integer_models_duals_too():
    model = commitment.definition().build(commitment.data(scale=2))
    solved = model.solve(solver="mosek")
    assert solved.status == "optimal"
    with pytest.raises(ValueError, match="refuses duals"):
        solved.dual("demand")


@licensed
def test_an_infeasible_model_is_named_and_its_conflict_is_refused():
    with infeasible().session(solver="mosek") as session:
        assert session.solve().status == "infeasible"
        with pytest.raises(ValueError, match="computes no conflict"):
            session.diagnose()


@licensed
def test_a_model_infeasible_only_through_its_integrality_is_infeasible():
    assert odd().solve(solver="mosek").status == "infeasible"


@licensed
def test_mosek_names_the_direction_an_unbounded_model_runs_off_in():
    # the ray is Mosek's certificate of dual infeasibility, which is a
    # direction the primal runs off in; what holds is that it names a column
    with unbounded().session(solver="mosek") as session:
        assert session.solve().status == "unbounded"
        found = session.diagnose()
    assert found.ray
    assert {held.variable for held in found.ray} == {"x"}


@licensed
def test_the_ray_stands_whatever_a_caller_passes():
    with unbounded().session(solver="mosek", options={"presolve": "off"}) as session:
        assert session.solve().status == "unbounded"
        found = session.diagnose()
    assert found.ray


def test_a_sense_the_mosek_adapter_does_not_know_is_refused():
    with pytest.raises(ValueError, match="sense is 'min' or 'max'"):
        mosek.solve(small().assemble(), "minimize")


def test_a_mosek_status_the_seam_does_not_name_is_refused_rather_than_folded():
    from nimopt.solvers.base import STATUS

    assert set(mosek.OUTCOME.values()) <= set(STATUS)
    assert set(mosek.LIMITS.values()) <= set(STATUS)
    assert not set(mosek.LIMITS) & set(mosek.FAILED)


@licensed
def test_a_limit_mosek_stops_at_is_named_rather_than_raised():
    model = commitment.definition().build(commitment.data(scale=SCALES["commitment"]))
    assert model.solve(solver="mosek", options={"node_limit": 0}).status == "node_limit"


@licensed
def test_a_mixed_integer_model_runs_only_mosek_s_mixed_integer_optimizer():
    # Mosek refuses a continuous optimizer on a model with integer columns
    # rather than solving the relaxation; the adapter names the cause
    model = commitment.definition().build(commitment.data(scale=2))
    with pytest.raises(RuntimeError, match="integer columns"):
        model.solve(solver="mosek", options={"method": "simplex"})


def test_the_mosek_map_covers_the_whole_vocabulary():
    assert set(mosek.OPTION_NAMES) == {o.name for o in VOCABULARY}
    assert set(mosek.OPTION_VALUES) <= set(mosek.OPTION_NAMES)


def test_an_option_the_vocabulary_does_not_carry_is_refused():
    with pytest.raises(ValueError, match="time_limt"):
        small().solve(solver="mosek", options={"time_limt": 60.0})


def test_an_option_mosek_does_not_carry_is_refused_by_name():
    with pytest.raises(ValueError, match="mosek carries no option 'pdlp_tol'"):
        small().solve(solver="mosek", options={"pdlp_tol": 1e-6})


def test_a_choice_mosek_lacks_is_refused_naming_what_it_takes():
    with pytest.raises(ValueError, match="mosek has no 'hipo' for option 'method'"):
        small().solve(solver="mosek", options={"method": "hipo"})


@licensed
@pytest.mark.parametrize("crossover", ["off", "on"])
def test_the_interior_point_answers_with_or_without_a_basis(crossover):
    # with crossover off Mosek defines no basic solution, and the adapter
    # reads the interior one rather than asking for a basis it never built
    held = MODULES["dispatch"]
    inputs = held.data(scale=SCALES["dispatch"])
    answer = (
        held.definition()
        .build(inputs)
        .solve(solver="mosek", options={"method": "barrier", "crossover": crossover})
    )
    assert answer.status == "optimal"
    assert answer.objective == pytest.approx(held.reference(inputs), rel=1e-6)
    assert answer.dual("balance").values().sum() != 0.0


@licensed
def test_the_log_is_silent_unless_asked_for(capsys):
    small().solve(solver="mosek")
    assert capsys.readouterr().out == ""
    small().solve(solver="mosek", options={"log": True})
    assert "Optimizer started" in capsys.readouterr().out


@licensed
def test_the_matrix_crosses_as_the_arrays_the_model_assembled(recwarn):
    # Mosek warns when it copies an array it was handed in the wrong form;
    # the handoff passes the pointer in the width Mosek reads
    small().solve(solver="mosek")
    assert [str(w.message) for w in recwarn] == []


# `threads` is left out for the same reason as in the Gurobi suite: HiGHS
# fixes its thread count at the first solve a process runs
PORTABLE = {
    "time_limit": 60.0,
    "seed": 7,
    "log": False,
    "presolve": "off",
    "method": "simplex",
    "feasibility_tol": 1e-7,
}


@licensed
def test_both_solvers_take_the_same_options_and_answer_the_same_model():
    held = MODULES["dispatch"]
    inputs = held.data(scale=SCALES["dispatch"])
    model = held.definition().build(inputs)
    expected = held.reference(inputs)
    for solver in ("highs", "mosek"):
        answer = model.solve(solver=solver, options=PORTABLE)
        assert answer.status == "optimal", solver
        assert answer.objective == pytest.approx(expected, rel=1e-6), solver


@licensed
def test_the_highs_reference_agrees_with_mosek_on_the_matrix_it_was_given():
    # the two adapters read one `Assembled`; a bound key derived wrongly
    # would move the optimum, and the reference model would show it
    assembled = small().assemble()
    _, from_highs, *_ = highs.solve(assembled, "min")
    _, from_mosek, *_ = mosek.solve(assembled, "min")
    assert from_mosek == pytest.approx(from_highs)
