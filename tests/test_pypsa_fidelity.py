import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
DATA = Path(__file__).parent.parent / "benchmarks" / "data"
NETWORK = DATA / "elec_s_10.npz"


@pytest.fixture(scope="module")
def reference():
    return json.loads((DATA / "elec_s_10_reference.json").read_text())


def test_the_column_space_matches_but_for_the_objective_constant(reference):
    from pypsa_network import build

    model = build(NETWORK)
    # linopy adds the objective's constant as a column of its own, with no
    # matrix entry; nimopt reports the constant as a number
    assert model.n_columns == reference["cols"] - 1


def _check(model, reference, names):
    """Each named family's rows and nonzeros, against PyPSA's own."""
    got = {}
    for name in names:
        constraint = model.constraints[name]
        want = reference["families"][name]
        got[name] = (constraint.n_rows, constraint.nnz, want["rows"], want["nnz"])
    wrong = {k: v for k, v in got.items() if v[:2] != v[2:]}
    assert not wrong, wrong


NOMINAL_BOUNDS = (
    "Generator-ext-p_nom-lower",
    "Generator-ext-p_nom-upper",
    "Line-ext-s_nom-lower",
    "Link-ext-p_nom-lower",
    "Store-ext-e_nom-lower",
)


def test_the_nominal_bound_rows_match(reference):
    from pypsa_network import build

    model = build(NETWORK)
    _check(model, reference, NOMINAL_BOUNDS)
    # lines, links and stores have an infinite nominal maximum; PyPSA builds
    # no upper row for them and neither does this
    assert "Line-ext-s_nom-upper" not in model.constraints
    assert "Link-ext-p_nom-upper" not in model.constraints
    assert "Store-ext-e_nom-upper" not in model.constraints


FIXED_OPERATIONAL = (
    "Generator-fix-p-lower",
    "Generator-fix-p-upper",
    "StorageUnit-fix-p_dispatch-lower",
    "StorageUnit-fix-p_dispatch-upper",
    "StorageUnit-fix-p_store-lower",
    "StorageUnit-fix-p_store-upper",
    "StorageUnit-fix-state_of_charge-lower",
    "StorageUnit-fix-state_of_charge-upper",
)


def test_the_fixed_operational_rows_match(reference):
    from pypsa_network import build

    model = build(NETWORK)
    _check(model, reference, FIXED_OPERATIONAL)
    # eleven generators and two storage units carry a fixed capacity
    rows = sum(model.constraints[name].n_rows for name in FIXED_OPERATIONAL)
    assert rows == 2 * 11 * 36 + 6 * 2 * 36


EXTENDABLE_OPERATIONAL = (
    "Generator-ext-p-lower",
    "Generator-ext-p-upper",
    "Line-ext-s-lower",
    "Line-ext-s-upper",
    "Link-ext-p-lower",
    "Link-ext-p-upper",
    "Store-ext-e-lower",
    "Store-ext-e-upper",
)


def test_the_extendable_operational_rows_match(reference):
    from pypsa_network import build

    model = build(NETWORK)
    _check(model, reference, EXTENDABLE_OPERATIONAL)


def test_a_zero_availability_states_a_row_without_a_capacity_term(reference):
    from pypsa_network import build

    model = build(NETWORK)
    upper = model.constraints["Generator-ext-p-upper"]
    # thirty extendable generators over thirty-six hours, each row holding the
    # generator's own term; 156 of them see zero availability and hold no
    # capacity term beside it
    assert upper.n_rows == 30 * 36
    assert upper.nnz == 30 * 36 + (30 * 36 - 156)


NETWORK_ROWS = ("Bus-nodal_balance", "Kirchhoff-Voltage-Law")


def test_the_network_rows_match(reference):
    from pypsa_network import build

    model = build(NETWORK)
    _check(model, reference, NETWORK_ROWS)


def test_every_bus_and_hour_states_a_balance_row(reference):
    from pypsa_network import build

    model = build(NETWORK)
    # no bus has every component; the rows are declared, not derived from
    # the terms that happen to cover them
    assert model.constraints["Bus-nodal_balance"].n_rows == 30 * 36


TEMPORAL = ("StorageUnit-energy_balance", "Store-energy_balance")


def test_the_temporal_rows_match(reference):
    from pypsa_network import build

    model = build(NETWORK)
    _check(model, reference, TEMPORAL)


def test_a_zero_store_efficiency_states_no_charging_term(reference):
    from pypsa_network import build

    model = build(NETWORK)
    # a storage unit stores at zero efficiency here; its charging term has
    # no coefficient and the row has four terms, not five
    assert model.constraints["StorageUnit-energy_balance"].nnz == 4 * 2 * 36


def test_the_global_limit_row_matches(reference):
    from pypsa_network import build

    model = build(NETWORK)
    _check(model, reference, ("GlobalConstraint-CO2Limit",))
    # ten generators burn a carrier that emits, over thirty-six hours
    assert model.constraints["GlobalConstraint-CO2Limit"].nnz == 10 * 36


def test_the_whole_matrix_agrees_with_pypsa(reference):
    from pypsa_network import build

    assembled = build(NETWORK).assemble()
    assert assembled.n_rows == reference["rows"]
    assert assembled.n_cols == reference["cols"] - 1
    assert int(assembled.values.size) == reference["nnz"]


def test_every_family_pypsa_states_is_stated_here(reference):
    from pypsa_network import build

    model = build(NETWORK)
    assert set(model.constraints) == set(reference["families"])


def test_the_model_solves_to_what_pypsa_solves_to(reference):
    from pypsa_network import build, constant

    model = build(NETWORK)
    answer = model.solve()
    assert answer.status == "optimal"
    # PyPSA reports the objective less what the existing capacity of its
    # extendable components already costs, which it holds as a constant
    reported = answer.objective - constant(NETWORK)
    assert abs(reported - reference["objective"]) < 1e-6 * abs(reference["objective"])
    assert abs(constant(NETWORK) - reference["constant"]) < 1.0


LARGE = DATA / "large" / "eu_24.npz"
LARGE_REFERENCE = DATA / "large" / "eu_24_reference.json"
needs_large = pytest.mark.skipif(
    not LARGE.exists(), reason=f"generate {LARGE} with benchmarks/pypsa_reference.py"
)

BALANCE_BLOCKS = (
    "Bus-nodal_balance",
    "Bus-meshed-30-nodal_balance",
    "Bus-meshed-100-nodal_balance",
    "Bus-meshed-400-nodal_balance",
)


@pytest.fixture(scope="module")
def large_reference():
    return json.loads(LARGE_REFERENCE.read_text())


@needs_large
def test_a_link_reaches_every_bus_it_joins(large_reference):
    from pypsa_network import build

    model = build(LARGE)
    want = sum(
        large_reference["families"][name]["nnz"]
        for name in BALANCE_BLOCKS
        if name in large_reference["families"]
    )
    # 838 links reach a third bus here, 348 a fourth and 114 a fifth
    assert model.constraints["Bus-nodal_balance"].nnz == want


LARGE_FIXED = (
    "Link-fix-p-lower",
    "Link-fix-p-upper",
    "Store-fix-e-lower",
    "Store-fix-e-upper",
)


@needs_large
def test_the_fixed_link_and_store_rows_match(large_reference):
    from pypsa_network import build

    _check(build(LARGE), large_reference, LARGE_FIXED)


@needs_large
def test_the_temporal_rows_match_at_scale(large_reference):
    from pypsa_network import build

    _check(build(LARGE), large_reference, TEMPORAL)


@needs_large
def test_a_unit_no_inflow_reaches_states_no_spill(large_reference):
    from pypsa_network import build

    balance = build(LARGE).constraints["StorageUnit-energy_balance"]
    # every one of the 56 units contributes its charge at this hour and the
    # last, and its dispatch; only the 30 with an inflow contribute a spill,
    # and only the 26 that store at a nonzero efficiency a charging term
    hours = 24
    assert balance.nnz == (56 * 2 + 56 + 30 + 26) * hours


@needs_large
def test_a_store_that_does_not_cycle_states_no_predecessor_at_the_first_hour(
    large_reference,
):
    from pypsa_network import build

    balance = build(LARGE).constraints["Store-energy_balance"]
    # two of the 278 stores do not cycle, and each contributes one term fewer
    assert balance.nnz == 278 * 24 * 3 - 2


@needs_large
def test_the_energy_sum_rows_match(large_reference):
    from pypsa_network import build

    families = ("Generator-e_sum_min", "Generator-e_sum_max")
    _check(build(LARGE), large_reference, families)


@needs_large
def test_the_operational_limit_row_matches(large_reference):
    from pypsa_network import build

    model = build(LARGE)
    _check(model, large_reference, ("GlobalConstraint-co2_sequestration_limit",))
    # the limit reads the sequestered store's energy at the last hour alone
    assert model.constraints["GlobalConstraint-co2_sequestration_limit"].nnz == 1


@needs_large
def test_a_carbon_limit_modelled_as_a_store_states_no_row(large_reference):
    from pypsa_network import build

    # this network caps carbon through a co2_atmosphere store and not a row;
    # PyPSA builds no CO2Limit family and neither does this
    assert "GlobalConstraint-CO2Limit" not in build(LARGE).constraints
    assert "GlobalConstraint-CO2Limit" not in large_reference["families"]


@needs_large
def test_the_whole_matrix_agrees_with_pypsa_at_scale(large_reference):
    from pypsa_network import build

    assembled = build(LARGE).assemble()
    assert assembled.n_rows == large_reference["rows"]
    assert assembled.n_cols == large_reference["cols"] - 1
    assert int(assembled.values.size) == large_reference["nnz"]


@needs_large
def test_every_pypsa_family_belongs_to_one_stated_group():
    from pypsa_fidelity import compare

    got = compare("eu_24", root=DATA / "large")
    for name, sides in got["families"].items():
        assert sides["rows"][0] == sides["rows"][1], (name, sides)
        assert sides["nnz"][0] == sides["nnz"][1], (name, sides)


def test_the_small_network_still_agrees_family_by_family():
    from pypsa_fidelity import compare

    got = compare("elec_s_10", root=DATA)
    wrong = {
        name: sides
        for name, sides in got["families"].items()
        if sides["rows"][0] != sides["rows"][1] or sides["nnz"][0] != sides["nnz"][1]
    }
    assert not wrong, wrong
