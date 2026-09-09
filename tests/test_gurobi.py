import numpy as np
import pytest

pytest.importorskip("gurobipy", reason="the Gurobi adapter needs the gurobi extra")

from nimopt import capabilities  # noqa: E402
from nimopt.models import commitment, dispatch, nodal, transport  # noqa: E402
from nimopt.solvers import available, gurobi  # noqa: E402
from nimopt.solvers.options import OPTIONS as VOCABULARY  # noqa: E402
from test_session import dispatch as small  # noqa: E402
from test_session import infeasible, odd, unbounded  # noqa: E402

# the size-limited licence builds and solves up to 2000 rows and 2000 columns,
# so each corpus model is taken at the largest scale that fits under it
SCALES = {"dispatch": 10, "nodal": 3, "transport": 10, "commitment": 4}

MODULES = {
    "dispatch": dispatch,
    "nodal": nodal,
    "transport": transport,
    "commitment": commitment,
}


def test_gurobi_is_available_where_its_backend_is():
    assert "gurobi" in available()
    assert capabilities("gurobi").supports("conflict")


@pytest.mark.parametrize("name", sorted(SCALES))
def test_both_adapters_answer_the_reference_the_model_computes(name):
    held = MODULES[name]
    inputs = held.data(scale=SCALES[name])
    model = held.definition().build(inputs)
    expected = held.reference(inputs)
    for solver in ("highs", "gurobi"):
        answer = model.solve(solver=solver)
        assert answer.status == "optimal", (solver, answer.status)
        assert answer.objective == pytest.approx(expected, rel=1e-6), solver


def test_the_two_adapters_read_the_same_primals_and_duals():
    model = dispatch.definition().build(dispatch.data(scale=SCALES["dispatch"]))
    got = {s: model.solve(solver=s) for s in ("highs", "gurobi")}
    assert got["gurobi"].primal("p").values().sum() == pytest.approx(
        got["highs"].primal("p").values().sum()
    )
    assert got["gurobi"].dual("balance").values().sum() == pytest.approx(
        got["highs"].dual("balance").values().sum()
    )


def test_gurobi_refuses_a_mixed_integer_models_duals_too():
    model = commitment.definition().build(commitment.data(scale=2))
    solved = model.solve(solver="gurobi")
    assert solved.status == "optimal"
    with pytest.raises(ValueError, match="refuses duals"):
        solved.dual("demand")


def test_a_conflict_from_gurobi_is_native_and_reads_as_a_row():
    # a descriptor that drifts from its adapter is the failure the
    # "as shipped" rule exists to prevent
    m = infeasible()
    with m.session(solver="gurobi") as session:
        assert session.solve().status == "infeasible"
        found = session.diagnose()
    assert found.solver == "gurobi"
    assert found.method == "native"
    assert repr(found.conflict[0]) == repr(m.row("balance", snapshot=1))


@pytest.mark.parametrize("solver", ["highs", "gurobi"])
def test_each_adapter_returns_a_set_that_conflicts(solver):
    # two solvers may return different irreducible sets, so what is asserted
    # of each is that removing it makes the model feasible
    from nimopt.solvers import highs

    m = infeasible()
    with m.session(solver=solver) as session:
        session.solve()
        found = session.diagnose()
        assembled = session.assembled
    assert found.conflict
    at = [row.index for row in found.conflict]
    assembled.row_lower[at] = -np.inf
    assembled.row_upper[at] = np.inf
    assert highs.solve(assembled, m.sense)[0] == "optimal"


def test_gurobi_reaches_an_infeasibility_that_is_only_the_integrality():
    # HiGHS computes its conflict over the relaxation and refuses this one;
    # Gurobi's reaches the integer column, and names the row itself
    m = odd()
    with m.session(solver="gurobi") as session:
        assert session.solve().status == "infeasible"
        found = session.diagnose()
    assert [row.constraint for row in found.conflict] == ["odd"]
    assert found.columns == ()


def test_gurobi_names_the_direction_an_unbounded_model_runs_off_in():
    # the ray is not HiGHS's ray; what holds of each is that it names a column
    with unbounded().session(solver="gurobi") as session:
        assert session.solve().status == "unbounded"
        found = session.diagnose()
    assert found.ray
    assert {held.variable for held in found.ray} == {"x"}


def test_a_sense_the_gurobi_adapter_does_not_know_is_refused():
    with pytest.raises(ValueError, match="sense is 'min' or 'max'"):
        gurobi.solve(small().assemble(), "minimize")


def test_a_gurobi_status_the_seam_does_not_name_is_refused_rather_than_folded():
    from nimopt.solvers.base import STATUS

    assert set(gurobi.OUTCOME.values()) <= set(STATUS)
    assert not set(gurobi.OUTCOME) & set(gurobi.FAILED)


def test_a_limit_gurobi_stops_at_is_named_rather_than_raised():
    # the vocabulary carries the limits both solvers express, and a work
    # limit is Gurobi's alone, so a node limit is what a caller reaches here
    model = commitment.definition().build(commitment.data(scale=SCALES["commitment"]))
    assert (
        model.solve(solver="gurobi", options={"node_limit": 0}).status == "node_limit"
    )


def test_a_model_over_the_licence_says_so_rather_than_answering():
    # the size-limited licence stops at 2000 rows and 2000 columns; a bigger
    # model must not come back with a number
    model = dispatch.definition().build(dispatch.data(scale=100))
    assert model.n_columns > 2000
    with pytest.raises(Exception, match="[Ll]icense"):
        model.solve(solver="gurobi")


def test_the_gurobi_map_covers_the_whole_vocabulary():
    assert set(gurobi.OPTION_NAMES) == {o.name for o in VOCABULARY}
    assert set(gurobi.OPTION_VALUES) <= set(gurobi.OPTION_NAMES)


def test_an_option_the_vocabulary_does_not_carry_is_refused():
    with pytest.raises(ValueError, match="time_limt"):
        small().solve(solver="gurobi", options={"time_limt": 60.0})


def test_the_ray_stands_whatever_a_caller_passes():
    # InfUnbdInfo is why the adapter can answer a ray; a caller's options are
    # read before the adapter's own, so they cannot withdraw it
    with unbounded().session(solver="gurobi", options={"presolve": "off"}) as session:
        assert session.solve().status == "unbounded"
        found = session.diagnose()
    assert found.ray
    assert {held.variable for held in found.ray} == {"x"}


# `threads` is left out: HiGHS fixes its thread count at the first solve a
# process runs, and the suite has solved many by the time this runs
PORTABLE = {
    "time_limit": 60.0,
    "seed": 7,
    "log": False,
    "presolve": "off",
    "method": "simplex",
    "feasibility_tol": 1e-7,
}


def test_both_solvers_take_the_same_options_and_answer_the_same_model():
    # what a portable vocabulary claims: one spelling, one meaning
    held = MODULES["dispatch"]
    inputs = held.data(scale=SCALES["dispatch"])
    model = held.definition().build(inputs)
    expected = held.reference(inputs)
    for solver in ("highs", "gurobi"):
        answer = model.solve(solver=solver, options=PORTABLE)
        assert answer.status == "optimal", solver
        assert answer.objective == pytest.approx(expected, rel=1e-6), solver
