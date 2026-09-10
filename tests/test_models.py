import importlib

import numpy as np
import pytest

from nimopt import Definition, Model
from nimopt.models import MODELS

SCALES_WITH_DATA = (
    "dispatch",
    "expansion",
    "recourse",
    "transport",
    "sector",
    "storage",
    "nodal",
    "commitment",
    "profiled",
)


def module(name):
    return importlib.import_module(f"nimopt.models.{name}")


def test_the_corpus_names_the_models_it_ships():
    assert MODELS == (
        "commitment",
        "dispatch",
        "expansion",
        "fleet",
        "nodal",
        "profiled",
        "recourse",
        "sector",
        "storage",
        "transport",
    )


@pytest.mark.parametrize("name", MODELS)
def test_each_model_states_a_definition_its_data_and_its_reference(name):
    held = module(name)
    assert isinstance(held.definition(), Definition)
    assert isinstance(held.data(), dict)
    assert isinstance(held.reference(held.data()), float)


@pytest.mark.parametrize("name", MODELS)
def test_each_model_explains_itself_with_nothing_bound(name):
    # the slice the documentation takes: a definition explains before data
    explained = module(name).definition().explain()
    assert explained.built is False
    assert explained.sets and explained.variables and explained.constraints


@pytest.mark.parametrize("name", MODELS)
def test_each_model_solves_to_the_objective_its_reference_computes(name):
    # the reference is arithmetic over the inputs; a test comparing nimopt to
    # nimopt would pass against a model that is wrong in the same way twice
    held = module(name)
    inputs = held.data()
    built = held.definition().build(inputs)
    assert isinstance(built, Model)
    answer = built.solve()
    assert answer.status == "optimal"
    assert answer.objective == pytest.approx(held.reference(inputs))


@pytest.mark.parametrize("name", SCALES_WITH_DATA)
def test_each_model_grows_with_its_scale(name):
    held = module(name)
    small = held.definition().build(held.data())
    large = held.definition().build(held.data(scale=3))
    assert large.n_columns > small.n_columns


@pytest.mark.parametrize("name", SCALES_WITH_DATA)
def test_a_model_at_scale_still_answers_its_reference(name):
    held = module(name)
    inputs = held.data(scale=3)
    answer = held.definition().build(inputs).solve()
    assert answer.objective == pytest.approx(held.reference(inputs))


def test_the_baseline_shape_is_a_merit_order():
    inputs = module("dispatch").data()
    # the fleet is cheapest first; the load is met by wind, then solar, then
    # gas, and the reference computes what that costs
    assert module("dispatch").reference(inputs) == pytest.approx(1920.0)
    # data() suffixes each copy of the fleet, so scale 1 is one copy of it
    assert list(inputs["generator"]) == ["wind0", "solar0", "gas0"]


def test_the_arithmetic_a_reference_shares_is_a_merit_order():
    from nimopt.models._arithmetic import merit_order

    capacity = np.array([100.0, 60.0, 200.0])
    cost = np.array([1.0, 2.0, 50.0])
    # 80 from the cheapest alone; 120 fills wind then takes 20 of solar
    assert merit_order(capacity, cost, np.array([80.0])) == pytest.approx(80.0)
    assert merit_order(capacity, cost, np.array([120.0])) == pytest.approx(140.0)


def test_a_load_the_fleet_cannot_meet_is_refused_by_the_arithmetic():
    from nimopt.models._arithmetic import merit_order

    with pytest.raises(ValueError, match="total 3.0 against a load of 9.0"):
        merit_order(np.array([3.0]), np.array([1.0]), np.array([9.0]))


def test_a_sparse_network_takes_a_column_per_arc():
    held = module("transport")
    inputs = held.data()
    built = held.definition().build(inputs)
    arcs = len(inputs["cost"][1])
    product = len(inputs["P"]) * len(inputs["W"])
    assert built.n_columns == arcs
    assert built.n_columns < product


def test_a_warehouse_no_arc_reaches_states_no_demand_row():
    # the last warehouse is outside every band, so its demand row is not
    # stated and absent names the rule that dropped it
    held = module("transport")
    inputs = held.data()
    absence = held.definition().build(inputs).absent("demand")
    assert absence.expected == len(inputs["W"])
    assert absence.standing == len(inputs["W"]) - 1
    assert [d.rule for d in absence.dropped_rows] == ["term-does-not-reach"]
    assert absence.dropped_rows[0].coordinate == {"W": inputs["W"][-1]}


def test_mixed_density_is_dense_in_one_axis_and_sparse_in_the_other():
    held = module("sector")
    inputs = held.data()
    built = held.definition().build(inputs)
    regions, techs, hours = (len(inputs[k]) for k in ("R", "K", "T"))
    sited = len(inputs["sited"][1])
    # every sited pair covers every hour, and the pairs are a subset
    assert built.n_columns == sited
    assert sited % hours == 0
    assert sited // hours < regions * techs


def test_a_cyclic_state_of_charge_states_every_hour():
    # the row at the first hour reads the last, so no hour is dropped
    held = module("storage")
    inputs = held.data()
    built = held.definition().build(inputs)
    assert built.constraints["state_of_charge"].n_rows == len(inputs["T"]) * len(
        inputs["S"]
    )
    assert built.absent("state_of_charge").dropped_rows == ()


def test_a_lossy_store_with_no_spread_to_work_on_stays_idle():
    # the reference is the hourly merit order, which is the optimum only
    # while shifting energy never pays. The premise is checked, not assumed:
    # a round trip returns 0.8835 of what it takes, and the fleet's dearest
    # unit at 55.0 saves 48.59 against the 50.0 the cheapest charges.
    held = module("storage")
    answer = held.definition().build(held.data()).solve()
    assert abs(answer.primal("charge").to_dense()).max() == pytest.approx(0.0)
    assert abs(answer.primal("discharge").to_dense()).max() == pytest.approx(0.0)


def test_a_lookup_turns_a_row_over_generators_into_a_row_over_buses():
    held = module("nodal")
    inputs = held.data()
    built = held.definition().build(inputs)
    # the balance is free over the dimensions the lookup introduces
    assert built.constraints["balance"].frame == ("B", "T")
    assert built.constraints["balance"].n_rows == len(inputs["B"]) * len(inputs["T"])


def test_a_generator_contributes_only_to_the_bus_it_sits_at():
    # every generator not at a bus is a term the lookup drops from that row
    held = module("nodal")
    absence = held.definition().build(held.data()).absent("balance")
    assert {d.rule for d in absence.dropped_terms} == {"absent-coefficient"}
    assert absence.dropped_rows == ()


def test_many_small_declarations_state_one_variable_each():
    held = module("fleet")
    inputs = held.data()
    built = held.definition().build(inputs)
    # one variable per unit, against one variable over the product
    assert len(built.variables) == len(held.units(1))
    assert built.n_columns == len(held.units(1)) * len(inputs["T"])


def test_a_fleet_declares_one_variable_per_unit_at_any_scale():
    held = module("fleet")
    built = held.definition(3).build(held.data(scale=3))
    assert len(built.variables) == len(held.units(3))
    assert built.solve().objective == pytest.approx(held.reference(held.data(scale=3)))


def test_the_committed_model_carries_binary_columns():
    held = module("commitment")
    built = held.definition().build(held.data())
    flags = built.integrality()
    assert flags.sum() == built.variables["on"].n_columns
    assert flags.sum() * 2 == built.n_columns


def test_a_commitment_is_checked_against_every_on_off_subset():
    # the reference enumerates commitments rather than solving one; nothing
    # couples one snapshot to the next, so the enumeration is exact
    held = module("commitment")
    inputs = held.data()
    answer = held.definition().build(inputs).solve()
    assert answer.objective == pytest.approx(held.reference(inputs))
    assert held.reference(inputs) == pytest.approx(13800.0)


def test_a_profile_bounds_each_generator_hour_by_hour():
    held = module("profiled")
    inputs = held.data()
    built = held.definition().build(inputs)
    _, upper = built.column_bounds()
    # the profile is stated over (G, T) and the variable over (T, G), so the
    # bounds are the profile transposed onto the column order
    assert list(upper) == list(inputs["profile"].T.ravel())


def test_every_model_the_corpus_names_can_be_imported():
    for name in MODELS:
        assert module(name).__doc__, name


def test_every_model_has_a_documentation_page():
    # the page is where an agent meets the model; one without a page ships
    # a shape nothing points at
    from docs_blocks import DOCS

    missing = [n for n in MODELS if not (DOCS / "models" / f"{n}.md").is_file()]
    assert missing == [], missing


def test_every_page_the_corpus_ships_is_listed_in_the_sidebar():
    from pathlib import Path

    import nimopt as no

    sidebar = (Path(no.__file__).parents[2] / "website" / "sidebars.ts").read_text()
    missing = [n for n in MODELS if f'"models/{n}"' not in sidebar]
    assert missing == [], missing


def test_no_model_is_named_by_the_package_root():
    # importing one model loads one; importing the package must not load eight
    import subprocess
    import sys

    code = "import nimopt, sys; print([m for m in sys.modules if 'nimopt.models' in m])"
    got = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert got.stdout.strip() == "[]"


def test_every_reference_is_a_number_the_solve_agrees_with():
    # the corpus's whole point, asserted over every model at once
    for name in MODELS:
        held = module(name)
        inputs = held.data()
        answer = held.definition().build(inputs).solve()
        assert answer.status == "optimal", name
        assert answer.objective == pytest.approx(held.reference(inputs)), name


def test_a_ramp_row_is_not_stated_at_the_hour_that_has_no_predecessor():
    # one model uses both lag rules: the charge row wraps and keeps every
    # hour, and the ramp row loses the first
    held = module("storage")
    inputs = held.data()
    built = held.definition().build(inputs)
    generators, hours = len(inputs["G"]), len(inputs["T"])
    assert built.constraints["ramp"].n_rows == generators * (hours - 1)
    absence = built.absent("ramp")
    assert absence.expected - absence.standing == generators
    assert {d.rule for d in absence.dropped_rows} == {"term-does-not-reach"}


def test_every_limit_the_store_carries_is_a_row():
    # a limit an agent reads a dual for is a row, not a bound
    held = module("storage")
    built = held.definition().build(held.data())
    assert set(built.constraints) == {
        "balance",
        "state_of_charge",
        "generation_limit",
        "ramp",
        "charge_limit",
        "discharge_limit",
        "energy_limit",
    }
