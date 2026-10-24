import numpy as np
import pytest

from nimopt import Model, Param, Set, Sum, available, capabilities, options
from nimopt.models import commitment, dispatch, nodal, transport
from nimopt.solvers import gams, highs
from nimopt.solvers.base import STATUS
from nimopt.solvers.options import OPTIONS as VOCABULARY
from test_session import dispatch as small
from test_session import infeasible, odd, unbounded
from test_solution_limit import knapsack

# each corpus model stays under the 2000 rows and 2000 columns the GAMS demo
# license allows; dispatch at scale 10 has 60 rows and 1800 columns
SCALES = {"dispatch": 10, "nodal": 8, "transport": 40, "commitment": 6}

MODULES = {
    "dispatch": dispatch,
    "nodal": nodal,
    "transport": transport,
    "commitment": commitment,
}

SEPARATE = {"separate_process": True}


def _runs(mode):
    """Whether GAMS is on PATH and runs CPLEX here under the options `mode`.

    A GAMS license without CPLEX, and a C++ runtime in this process older than
    the one CPLEX requires, return False. Any other failure raises.
    """
    if not gams.installed():
        return False
    try:
        return small().solve(solver="gams", options=mode).status == "optimal"
    except RuntimeError as error:
        if "licensing" in str(error) or "in this process" in str(error):
            return False
        raise


on_path = pytest.mark.skipif(
    not gams.installed(), reason="the gams adapter needs GAMS on PATH"
)

in_process = pytest.mark.skipif(
    not _runs({}),
    reason="CPLEX does not run in this process: no GAMS license for CPLEX, or a "
    "C++ runtime older than the one CPLEX requires",
)

separate = pytest.mark.skipif(
    not _runs(SEPARATE),
    reason="CPLEX does not run in a separate process: no GAMS license for CPLEX",
)

MODES = [
    pytest.param({}, id="in-process", marks=in_process),
    pytest.param(SEPARATE, id="separate-process", marks=separate),
]


def revenue():
    """A maximisation: two generators sell into one bounded market."""
    SNAP = Set("snapshot", np.arange(3))
    GEN = Set("generator", np.array(["wind", "gas"]))
    p_max = Param.from_dense("p_max", (GEN,), np.array([10.0, 20.0]))
    demand = Param.from_dense("demand", (SNAP,), np.array([25.0, 20.0, 5.0]))
    price = Param.from_dense("price", (GEN,), np.array([3.0, 1.0]))
    m = Model("revenue", sense="max")
    p = m.var("p", (SNAP, GEN), lower=0.0, upper=p_max)
    m.constraint("market", Sum(GEN, p[SNAP, GEN]) <= demand[SNAP])
    m.set_objective(Sum(SNAP, GEN, price[GEN] * p[SNAP, GEN]))
    return m


def free():
    """One free column held from below at -3: the optimum is -3."""
    T = Set("T", np.arange(1))
    one = Param.from_dense("one", (T,), np.ones(1))
    m = Model("free", sense="min")
    x = m.var("x", (T,), lower=-np.inf)
    m.constraint("floor", one[T] * x[T] >= -3.0)
    m.set_objective(Sum(T, one[T] * x[T]))
    return m


def test_the_gams_descriptor_is_readable_without_gams():
    # the adapter reads no conflict and no ray out of GMO; a session raises
    # for both and calls the adapter for neither
    held = capabilities("gams")
    assert held.supports("integrality")
    assert held.supports("duals")
    assert not held.supports("conflict")
    assert not held.supports("ray")
    assert held.rejects("integrality", "duals")


def test_gams_is_not_installed_without_gams_on_path(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert gams.installed() is False
    assert "gams" not in available()


def test_a_solve_without_gams_on_path_raises_naming_path(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="PATH"):
        small().solve(solver="gams")


@on_path
def test_gams_is_available_where_gams_is_on_path():
    assert gams.installed() is True
    assert "gams" in available()


@pytest.mark.parametrize(
    ("platform", "name"),
    [
        ("linux", "libjoatdclib64.so"),
        ("darwin", "libjoatdclib64.dylib"),
        ("win32", "joatdclib64.dll"),
    ],
)
def test_the_library_name_is_the_one_the_gams_c_api_loads(platform, name):
    # apifiles/C/api/gmomcc.c: prefix "lib" except on Windows, suffix "64" on
    # a 64-bit platform, extension .so, .dylib or .dll
    assert gams._library_name(platform) == name


def test_a_platform_without_a_gams_library_raises_naming_it():
    with pytest.raises(RuntimeError, match="'sunos5'"):
        gams._library_name("sunos5")


def test_a_library_that_does_not_load_raises_naming_its_directory(tmp_path):
    (tmp_path / gams._library_name()).write_bytes(b"not a shared library")
    with pytest.raises(RuntimeError, match="does not load into this Python"):
        gams._Library(tmp_path)


def test_the_gams_status_maps_name_only_members_of_the_vocabulary():
    assert set(gams.OUTCOME.values()) <= set(STATUS)
    assert set(gams.LIMITS.values()) <= set(STATUS)
    assert not set(gams.LIMITS) & set(gams.FAILED)


def test_no_status_without_an_optimum_reads_a_feasible_point():
    # a model status that reports infeasibility or unboundedness must not
    # pass its iterate as a primal-feasible point
    unsolved = {name for name, status in gams.OUTCOME.items() if status != "optimal"}
    assert unsolved
    assert not unsolved & set(gams.FEASIBLE)


def test_a_termination_without_a_limit_points_to_the_cplex_log():
    # CPLEX barrier on a week of the European network stops with "not proven
    # optimal due to numerical difficulties": model status 7, solve status 4
    with pytest.raises(RuntimeError, match="reached no limit.*log=True"):
        gams._status("feasible solution", "terminated by solver", {}, 102.0, 2e300)


def test_a_termination_at_two_limits_asks_for_one():
    limits = {"iteration_limit": 0, "node_limit": 0}
    with pytest.raises(RuntimeError, match="set one of the two"):
        gams._status("integer solution", "terminated by solver", limits, 0.0, 0.0)


def test_a_signature_the_library_does_not_confirm_raises_naming_it():
    # a GAMS release whose entry point takes other arguments would be called
    # with the wrong stack; the adapter raises before the first call
    def confirms(name, codes):
        return name != "gmoAddRow"

    with pytest.raises(RuntimeError, match="gmoAddRow"):
        gams._check_entries(confirms)


def test_every_entry_point_is_confirmed_on_a_matching_library():
    seen = []
    gams._check_entries(lambda name, codes: seen.append(name) or True)
    assert "gmoAddRow" in seen and "gevCallSolver" in seen


def test_separate_process_is_the_gams_solvelink_and_a_bool():
    native = {o.name: o.native for o in options("gams")}
    assert native["separate_process"] == "solvelink"
    assert gams.OPTION_VALUES["separate_process"] == {False: 5, True: 2}


def test_an_adapter_without_a_separate_process_rejects_the_option():
    with pytest.raises(ValueError, match="highs has no option 'separate_process'"):
        small().solve(solver="highs", options=SEPARATE)


@on_path
@pytest.mark.parametrize(
    ("mode", "action"),
    [({}, "separate_process=True"), (SEPARATE, "run `gams` in a terminal")],
)
def test_a_failed_solver_call_names_the_action_for_its_mode(monkeypatch, mode, action):
    # GEV returns nonzero where it does not call CPLEX; in this process the
    # cause is often a C++ runtime older than the one CPLEX requires
    library = gams._library(gams.system_directory())
    monkeypatch.setattr(library, "gevCallSolver", lambda *arguments: 1)
    with pytest.raises(RuntimeError, match=action):
        small().solve(solver="gams", options=mode)


@separate
def test_a_separate_process_leaves_no_scratch_files():
    # GEV writes the model and the solution into the scratch directory; at
    # full-year scale they take 900 MB
    scratch = gams._control_file(gams.system_directory()).parent
    before = sorted(path.name for path in scratch.iterdir())
    small().solve(solver="gams", options=SEPARATE)
    assert sorted(path.name for path in scratch.iterdir()) == before


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", sorted(SCALES))
def test_gams_and_highs_return_the_reference_the_model_computes(name, mode):
    held = MODULES[name]
    inputs = held.data(scale=SCALES[name])
    model = held.definition().build(inputs)
    expected = held.reference(inputs)
    for solver, given in (("highs", {}), ("gams", mode)):
        solved = model.solve(solver=solver, options=given)
        assert solved.status == "optimal", (solver, solved.status)
        assert solved.objective == pytest.approx(expected, rel=1e-6), solver


@pytest.mark.parametrize("mode", MODES)
def test_gams_and_highs_read_the_same_primals_and_duals(mode):
    model = dispatch.definition().build(dispatch.data(scale=SCALES["dispatch"]))
    got = {
        "highs": model.solve(solver="highs"),
        "gams": model.solve(solver="gams", options=mode),
    }
    assert got["gams"].primal("p").values().sum() == pytest.approx(
        got["highs"].primal("p").values().sum()
    )
    assert got["gams"].dual("balance").values().sum() == pytest.approx(
        got["highs"].dual("balance").values().sum()
    )


@pytest.mark.parametrize("mode", MODES)
def test_the_duals_have_the_highs_sign_under_maximisation_too(mode):
    from_highs = revenue().solve(solver="highs")
    from_gams = revenue().solve(solver="gams", options=mode)
    assert from_gams.objective == pytest.approx(from_highs.objective)
    assert from_gams.dual("market").values() == pytest.approx(
        from_highs.dual("market").values()
    )


@pytest.mark.parametrize("mode", MODES)
def test_a_free_column_reaches_a_negative_optimum(mode):
    # a lower bound of -inf passed as 0 would stop the column at 0
    solved = free().solve(solver="gams", options=mode)
    assert solved.objective == pytest.approx(-3.0)


@pytest.mark.parametrize("mode", MODES)
def test_a_mixed_integer_models_duals_are_rejected(mode):
    model = commitment.definition().build(commitment.data(scale=2))
    solved = model.solve(solver="gams", options=mode)
    assert solved.status == "optimal"
    with pytest.raises(ValueError, match="reports no duals for it"):
        solved.dual("balance")


@pytest.mark.parametrize("mode", MODES)
def test_gams_bounds_a_continuous_optimum_by_its_own_objective(mode):
    solved = small().solve(solver="gams", options=mode)
    assert solved.status == "optimal"
    assert solved.feasible is True
    assert solved.bound == pytest.approx(solved.objective)
    assert solved.gap == 0.0


@pytest.mark.parametrize("mode", MODES)
def test_gams_bounds_a_mixed_integer_optimum_by_its_dual_bound(mode):
    model = commitment.definition().build(commitment.data(scale=2))
    solved = model.solve(solver="gams", options=mode)
    assert solved.status == "optimal"
    assert np.isfinite(solved.bound)
    assert solved.gap == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("mode", MODES)
def test_an_infeasible_model_is_named_and_its_conflict_is_rejected(mode):
    with infeasible().session(solver="gams", options=mode) as session:
        assert session.solve().status == "infeasible"
        with pytest.raises(ValueError, match="computes no conflict"):
            session.diagnose()


@pytest.mark.parametrize("mode", MODES)
def test_a_model_infeasible_only_through_its_integrality_is_infeasible(mode):
    solved = odd().solve(solver="gams", options=mode)
    assert solved.status == "infeasible"
    assert solved.bound is None


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("presolve", ["choose", "off"])
def test_an_unbounded_model_is_unbounded_and_has_no_ray(presolve, mode):
    # CPLEX presolve proves only that no finite optimum exists; the adapter
    # sets `rerun yes`, and CPLEX solves again without presolve to separate
    # the two
    given = {"presolve": presolve, **mode}
    with unbounded().session(solver="gams", options=given) as session:
        solution = session.solve()
        assert solution.status == "unbounded"
        with pytest.raises(ValueError, match="computes no ray"):
            session.diagnose()
    with pytest.raises(ValueError, match="the objective has no finite optimum"):
        solution.objective


@pytest.mark.parametrize("mode", MODES)
def test_a_node_limit_returns_the_incumbent_under_its_bound(mode):
    # CPLEX explores one node: an incumbent of 2967.24 under a bound near 2990
    solved = knapsack(120, 10).solve(solver="gams", options={"node_limit": 1, **mode})
    assert solved.status == "node_limit"
    assert solved.feasible is True
    assert 0.0 < solved.objective < solved.bound
    assert solved.gap > 0.0


@pytest.mark.parametrize("mode", MODES)
def test_a_time_limit_of_zero_stops_at_the_time_limit(mode):
    solved = knapsack(120, 10).solve(solver="gams", options={"time_limit": 0.0, **mode})
    assert solved.status == "time_limit"


@pytest.mark.parametrize("mode", MODES)
def test_an_iteration_limit_stops_a_continuous_model(mode):
    solved = small().solve(solver="gams", options={"iteration_limit": 0, **mode})
    assert solved.status == "iteration_limit"


@pytest.mark.parametrize("mode", MODES)
def test_an_iteration_limit_stops_a_mixed_integer_model(mode):
    # GAMS reports the same solve status for a node and an iteration limit
    # on a mixed-integer model; the iteration count identifies this one
    solved = knapsack(120, 10).solve(
        solver="gams", options={"iteration_limit": 0, **mode}
    )
    assert solved.status == "iteration_limit"


@pytest.mark.parametrize("mode", MODES)
def test_the_session_holds_no_backend_after_a_solve(mode):
    # the adapter frees GMO and GEV after it reads the values; GMO stores a
    # copy of the whole matrix
    with small().session(solver="gams", options=mode) as session:
        session.solve()
        assert session.backend_open is False


@pytest.mark.parametrize("mode", MODES)
def test_the_log_is_silent_unless_asked_for(capfd, mode):
    small().solve(solver="gams", options=mode)
    assert capfd.readouterr().out == ""
    small().solve(solver="gams", options={"log": True, **mode})
    assert "IBM ILOG CPLEX" in capfd.readouterr().out


PORTABLE = {
    "time_limit": 60.0,
    "seed": 7,
    "log": False,
    "presolve": "off",
    "method": "simplex",
    "feasibility_tol": 1e-7,
    "optimality_tol": 1e-7,
}


@pytest.mark.parametrize("mode", MODES)
def test_gams_and_highs_take_the_same_options_and_return_the_reference(mode):
    # `threads` is passed to CPLEX alone: HiGHS fixes its thread count at the
    # first solve a process runs
    held = MODULES["dispatch"]
    inputs = held.data(scale=SCALES["dispatch"])
    model = held.definition().build(inputs)
    expected = held.reference(inputs)
    for solver, given in (
        ("highs", PORTABLE),
        ("gams", {**PORTABLE, "threads": 1, **mode}),
    ):
        solved = model.solve(solver=solver, options=given)
        assert solved.status == "optimal", solver
        assert solved.objective == pytest.approx(expected, rel=1e-6), solver


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", ["simplex", "barrier"])
def test_each_method_returns_the_reference(method, mode):
    held = MODULES["dispatch"]
    inputs = held.data(scale=SCALES["dispatch"])
    solved = (
        held.definition()
        .build(inputs)
        .solve(solver="gams", options={"method": method, "crossover": "on", **mode})
    )
    assert solved.status == "optimal"
    assert solved.objective == pytest.approx(held.reference(inputs), rel=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_crossover_off_uses_no_deprecated_cplex_option(capfd, mode):
    # CPLEX 22 replaces `barcrossalg -1` with `solutiontype 2` and reports
    # "Depreciated value -1 for BarCrossAlg" in its log
    held = MODULES["dispatch"]
    inputs = held.data(scale=SCALES["dispatch"])
    solved = (
        held.definition()
        .build(inputs)
        .solve(
            solver="gams",
            options={"method": "barrier", "crossover": "off", "log": True, **mode},
        )
    )
    assert solved.status == "optimal"
    assert solved.objective == pytest.approx(held.reference(inputs), rel=1e-6)
    log = capfd.readouterr().out
    assert "IBM ILOG CPLEX" in log
    assert "Depreciated" not in log


@pytest.mark.parametrize("mode", MODES)
def test_a_mip_gap_is_passed_to_cplex(mode):
    # a relative gap of 1 accepts the first incumbent, below the optimum
    exact = knapsack(120, 10).solve(solver="gams", options=mode)
    loose = knapsack(120, 10).solve(solver="gams", options={"mip_gap": 1.0, **mode})
    assert exact.status == "optimal"
    assert loose.status == "optimal"
    assert loose.objective < exact.objective


def test_the_gams_map_covers_the_whole_vocabulary():
    assert set(gams.OPTION_NAMES) == {o.name for o in VOCABULARY}
    assert set(gams.OPTION_VALUES) <= set(gams.OPTION_NAMES)


@on_path
def test_every_option_file_name_is_a_cplex_option_gams_defines():
    # CPLEX under GAMS logs an unknown option and solves on; a name absent
    # from the definition file would be dropped without an error
    defined = gams.system_directory() / "optcplex.def"
    names = {
        line.split()[0].lower()
        for line in defined.read_text().splitlines()
        if line.strip()
    }
    written = {
        native
        for native in gams.OPTION_NAMES.values()
        if native is not None and native not in gams.CALL_OPTIONS
    }
    assert written
    assert written <= names


def test_an_option_cplex_under_gams_lacks_is_rejected_by_name():
    with pytest.raises(ValueError, match="gams has no option 'pdlp_tol'"):
        small().solve(solver="gams", options={"pdlp_tol": 1e-6})


def test_a_choice_cplex_under_gams_lacks_is_rejected_naming_what_it_takes():
    with pytest.raises(ValueError, match="gams has no 'hipo' for option 'method'"):
        small().solve(solver="gams", options={"method": "hipo"})


@in_process
def test_the_highs_reference_agrees_with_gams_on_the_matrix_it_was_given():
    assembled = small().assemble()
    assert gams.solve(assembled, "min").objective == pytest.approx(
        highs.solve(assembled, "min").objective
    )
