import numpy as np
import pytest

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set
from nimopt.solvers import highs
from nimopt.solvers.options import BY_NAME, OPTIONS, Option, checked, translated
from nimopt.solvers.options import OPTIONS as VOCABULARY
from nimopt.term import Sum


def test_the_vocabulary_names_fifteen_options_each_stating_what_it_does():
    assert len(OPTIONS) == 15
    assert all(isinstance(o, Option) for o in OPTIONS)
    assert all(o.does for o in OPTIONS), "every option says what it does"
    assert set(BY_NAME) == {o.name for o in OPTIONS}
    assert "time_limit" in BY_NAME and "mip_gap" in BY_NAME


def test_a_choice_states_its_values_and_a_scalar_states_none():
    assert BY_NAME["presolve"].choices == ("off", "choose", "on")
    assert BY_NAME["method"].choices == ("choose", "simplex", "barrier", "hipo", "pdlp")
    assert BY_NAME["newton_system"].choices == ("choose", "augmented", "normaleq")
    assert BY_NAME["crossover"].choices == ("choose", "off", "on")
    assert BY_NAME["time_limit"].choices == ()


def test_no_options_are_no_options():
    assert checked(None) == {}
    assert checked({}) == {}


def test_an_option_the_vocabulary_does_not_carry_is_refused():
    # the failure this exists to prevent: a typo that solves a different
    # problem and reports success
    with pytest.raises(ValueError, match="time_limt"):
        checked({"time_limt": 60.0})


def test_a_value_of_the_wrong_kind_is_refused():
    with pytest.raises(TypeError, match="takes a number"):
        checked({"time_limit": "soon"})
    with pytest.raises(TypeError, match="takes an int"):
        checked({"threads": 4.5})
    with pytest.raises(TypeError, match="takes a bool"):
        checked({"log": 1})


def test_a_bool_is_not_an_int_here():
    # bool subclasses int, so an unguarded check would take True for a count
    with pytest.raises(TypeError, match="takes an int"):
        checked({"threads": True})


def test_an_int_stands_where_a_number_is_asked_for():
    assert checked({"time_limit": 60}) == {"time_limit": 60}


def test_a_value_outside_a_choice_is_refused_naming_the_set():
    with pytest.raises(ValueError, match=r"\('off', 'choose', 'on'\)"):
        checked({"presolve": "aggressive"})


def test_translation_states_a_solvers_own_names_and_values():
    names = {"time_limit": "TimeLimit", "presolve": "Presolve"}
    values = {"presolve": {"off": 0, "choose": -1, "on": 2}}
    got = translated({"time_limit": 60.0, "presolve": "off"}, names, values)
    assert got == {"TimeLimit": 60.0, "Presolve": 0}


def test_translation_refuses_what_the_vocabulary_refuses():
    with pytest.raises(ValueError, match="nope"):
        translated({"nope": 1}, {}, {})


def tiny():
    """A two-column model that solves in no time."""
    T = Set("T", np.array([0, 1]))
    m = Model("m")
    x = m.var("x", (T,), upper=1.0)
    one = Param.from_dense("one", (T,), np.ones(2))
    m.eq("cap", one[T] * x[T] <= 1.0)
    m.set_objective(Sum(T, one[T] * x[T]))
    return m.assemble()


def test_the_highs_map_covers_the_whole_vocabulary():
    # a vocabulary an adapter half-carries would refuse an option on one
    # solver and take it on the other
    assert set(highs.OPTION_NAMES) == {o.name for o in VOCABULARY}
    assert set(highs.OPTION_VALUES) <= set(highs.OPTION_NAMES)


def test_the_gurobi_map_covers_the_whole_vocabulary():
    from nimopt.solvers import gurobi

    assert set(gurobi.OPTION_NAMES) == {o.name for o in VOCABULARY}
    assert set(gurobi.OPTION_VALUES) <= set(gurobi.OPTION_NAMES)


def test_an_option_a_solver_does_not_carry_is_refused_by_name():
    from nimopt.solvers import gurobi

    with pytest.raises(ValueError, match="gurobi carries no option 'newton_system'"):
        translated(
            {"newton_system": "augmented"},
            gurobi.OPTION_NAMES,
            gurobi.OPTION_VALUES,
            solver="gurobi",
        )


def test_a_choice_a_solver_lacks_is_refused_naming_what_it_takes():
    from nimopt.solvers import gurobi

    with pytest.raises(ValueError, match="gurobi has no 'hipo' for option 'method'"):
        translated(
            {"method": "hipo"},
            gurobi.OPTION_NAMES,
            gurobi.OPTION_VALUES,
            solver="gurobi",
        )
    # a choice both carry still crosses, under gurobi's own value
    held = translated(
        {"crossover": "off"}, gurobi.OPTION_NAMES, gurobi.OPTION_VALUES, solver="gurobi"
    )
    assert held == {"Crossover": 0}


def test_the_interior_point_options_reach_highs():
    # HiGHS accepts the HiPO and PDLP settings whichever method runs, so a
    # solve carrying them reports the optimum the tiny model has
    result = highs.solve(
        tiny(),
        "min",
        {"newton_system": "augmented", "crossover": "off", "pdlp_tol": 1e-6},
    )
    assert result.status == "optimal"


def test_hipo_runs_where_highs_carries_it_and_is_refused_where_it_does_not():
    # the PyPI wheel lacks HiPO's extras library and would run simplex while
    # logging an error; a HiGHS built with the library runs HiPO
    if highs.hipo_available():
        result = highs.solve(tiny(), "min", {"method": "hipo", "crossover": "off"})
        assert result.status == "optimal"
    else:
        with pytest.raises(RuntimeError, match="extras library"):
            highs.solve(tiny(), "min", {"method": "hipo"})


def test_an_option_highs_does_not_carry_is_refused():
    with pytest.raises(ValueError, match="time_limt"):
        highs.solve(tiny(), "min", {"time_limt": 60.0})


def test_an_option_highs_carries_reaches_it():
    result = highs.solve(tiny(), "min", {"time_limit": 60.0, "presolve": "off"})
    assert result.status == "optimal"


def test_log_is_off_unless_a_caller_asks_and_reaches_the_solver_when_it_does(capfd):
    # HiGHS writes its log to the process's own stdout rather than Python's,
    # so `capfd` is what reads it and `capsys` would not
    highs.solve(tiny(), "min")
    assert capfd.readouterr().out == "", "a solve is silent by default"
    highs.solve(tiny(), "min", {"log": True})
    assert capfd.readouterr().out != "", "log=True reaches the solver"


def test_a_thread_count_highs_cannot_take_says_why():
    # HiGHS fixes its thread count at the first solve a process runs; the
    # suite has solved by now, so this asks for a count it will not take
    with pytest.raises(RuntimeError, match="thread count at the first solve"):
        highs.solve(tiny(), "min", {"threads": 3})


def test_the_vocabulary_reads_back_without_naming_a_solver():
    from nimopt import options

    assert options() == VOCABULARY
    assert all(o.native == "" for o in options())


def test_reading_it_for_a_solver_fills_that_solvers_own_names():
    from nimopt import options

    held = {o.name: o for o in options("highs")}
    assert held["time_limit"].native == "time_limit"
    assert held["mip_gap"].native == "mip_rel_gap"
    assert dict(held["method"].native_choices)["barrier"] == "ipm"
    assert held["time_limit"].does == BY_NAME["time_limit"].does
    lacking = {o.name: o for o in options("gurobi")}
    assert lacking["newton_system"].native is None
    assert "hipo" not in dict(lacking["method"].native_choices)


def test_a_solver_that_is_not_an_adapter_is_refused():
    from nimopt import options

    with pytest.raises(ValueError, match="the solvers are"):
        options("cplex")


def test_reading_it_for_a_solver_needs_no_backend_installed():
    # a descriptor is readable whether or not a backend is here, and so is
    # the option list a caller reads before installing one
    from nimopt import options

    assert len(options("gurobi")) == len(VOCABULARY)


def test_every_native_highs_name_resolves_in_highs():
    # a name HiGHS renames upstream would misconfigure a solve silently;
    # setOptionValue answering kError is what catches it here
    highspy = pytest.importorskip("highspy")
    from nimopt import options

    engine = highspy.Highs()
    engine.setOptionValue("output_flag", False)
    probe = {"float": 1.0, "int": 1, "bool": False, "choice": None}
    unresolved = []
    for option in options("highs"):
        value = (
            dict(option.native_choices)[option.choices[0]]
            if option.kind == "choice"
            else probe[option.kind]
        )
        if engine.setOptionValue(option.native, value) != highspy.HighsStatus.kOk:
            unresolved.append(option.native)
    assert unresolved == [], unresolved


def test_every_native_gurobi_name_resolves_in_gurobi():
    gp = pytest.importorskip("gurobipy")
    from nimopt import options

    known = {p.lower() for p in dir(gp.GRB.Param) if not p.startswith("_")}
    unresolved = [
        o.native
        for o in options("gurobi")
        if o.native is not None and o.native.lower() not in known
    ]
    assert unresolved == [], unresolved


def test_the_mosek_map_covers_the_whole_vocabulary():
    from nimopt.solvers import mosek

    assert set(mosek.OPTION_NAMES) == {o.name for o in VOCABULARY}
    assert set(mosek.OPTION_VALUES) <= set(mosek.OPTION_NAMES)


def test_mosek_spells_each_shared_choice_in_its_own_enum():
    # a choice both carry crosses under Mosek's own member name, which is
    # what the adapter resolves against Mosek's enums
    from nimopt.solvers import mosek

    held = translated(
        {"method": "barrier", "presolve": "off", "crossover": "off"},
        mosek.OPTION_NAMES,
        mosek.OPTION_VALUES,
        solver="mosek",
    )
    assert held == {
        "optimizer": "intpnt",
        "presolve_use": "off",
        "intpnt_basis": "never",
    }
