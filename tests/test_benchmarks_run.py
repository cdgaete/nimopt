import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))


def test_the_transport_benchmark_reports_a_matrix_and_a_peak():
    from bench_transport import measure

    got = measure(n_plants=200, n_warehouses=100, arcs_per_plant=10)
    # one column per arc, and each arc appears in one supply and one demand row
    assert got["cols"] == 200 * 10
    assert got["nnz"] == 2 * 200 * 10
    assert got["rows"] == 200 + 100
    assert got["build_ms"] > 0


def test_the_transport_model_solves():
    from bench_transport import measure

    got = measure(n_plants=200, n_warehouses=100, arcs_per_plant=10, solve=True)
    assert got["status"] == "optimal"
    assert got["objective"] > 0


def test_the_storage_benchmark_states_the_rows_the_horizon_allows():
    from bench_storage import measure

    got = measure(n_generators=10, n_storage=2, n_hours=168)
    # gen over (G, T) plus charge, discharge and stored energy over (S, T)
    assert got["cols"] == 10 * 168 + 3 * 2 * 168
    # the ramp row at the first hour has no predecessor and is absent; the
    # cyclic charge row reads the last hour and keeps every row
    ramp_rows = 10 * (168 - 1)
    storage_rows = 2 * 168
    assert got["rows"] == 168 + storage_rows + 10 * 168 + ramp_rows + 3 * storage_rows


def test_the_storage_model_solves_and_its_state_of_charge_is_cyclic():
    from bench_storage import (
        CHARGE_EFFICIENCY,
        DISCHARGE_EFFICIENCY,
        build,
        profiles,
    )

    availability, demand, price = profiles(10, 168)
    model = build(10, 2, 168, availability, demand, price)
    got = model.solve()
    assert got.status == "optimal"

    soc = got.primal("soc").to_dense()
    charge = got.primal("charge").to_dense()
    discharge = got.primal("discharge").to_dense()
    # every hour's balance holds, the first against the last
    residual = (
        soc
        - np.roll(soc, 1, axis=1)
        - CHARGE_EFFICIENCY * charge
        + discharge / DISCHARGE_EFFICIENCY
    )
    assert np.allclose(residual, 0.0, atol=1e-6), residual.max()
    # the batteries are worth having: they move energy across the day
    assert discharge.sum() > 0.0


def test_a_build_peaks_within_eight_times_the_matrix_it_produces():
    from bench_storage import measure as storage
    from bench_transport import measure as transport

    # peak spans building the model and assembling it, so the model, the
    # vectors the solver takes beside the matrix, and one constraint's
    # working set are all live at the top. Measured 7.30x on the transport
    # rung and 7.37x on the storage one, both stable across runs; 8.0
    # catches a regression rather than run-to-run noise.
    got = transport(n_plants=10000, n_warehouses=2000, arcs_per_plant=40)
    assert got["ratio"] < 8.0, got
    got = storage(n_generators=40, n_storage=8, n_hours=720)
    assert got["ratio"] < 8.0, got


def test_a_rung_measured_without_a_solve_reports_its_shape():
    pytest.importorskip(
        "linopy", reason="the comparison benchmarks need the bench extra"
    )
    from bench_vs_linopy import compare

    sizes = dict(n_plants=20, n_warehouses=10, arcs_per_plant=3)
    sides = compare("transport", sizes, traced=False, solve=False)
    for side in ("nimopt", "linopy"):
        got = sides[side]
        assert got["rows"] > 0 and got["cols"] > 0 and got["nnz"] > 0, side
        # a rung too large to solve still reports what it built
        assert "objective" not in got, side


def test_agreement_refuses_a_shape_mismatch_whether_or_not_it_solved():
    pytest.importorskip(
        "linopy", reason="the comparison benchmarks need the bench extra"
    )
    from bench_vs_linopy import agree

    stated = {"rows": 10, "cols": 5, "nnz": 20}
    apart = {"rows": 11, "cols": 5, "nnz": 20}
    with pytest.raises(ValueError, match="different problems"):
        agree({"nimopt": stated, "linopy": apart}, {"snapshots": 1})
    # two sides that agree on shape and were not solved are accepted
    assert agree({"nimopt": stated, "linopy": dict(stated)}, {"snapshots": 1}) == stated


def test_the_european_case_reports_the_sparsity_it_measured():
    from bench_pypsa import ARRAYS, nimopt

    if not ARRAYS.exists():
        pytest.skip(f"generate {ARRAYS} with benchmarks/pypsa_reference.py")
    case = nimopt(24)
    got = case.describe(case.build())
    assert (got["rows"], got["cols"], got["nnz"]) == (162582, 79457, 382520)
    assert abs(got["nnz"] / got["cols"] - 4.81) < 0.01
    # the 2,537 nominal columns less those whose capacity term drops at every
    # hour the component is unavailable, which is the rule that leaves the
    # extendable generator's upper rows short of two nonzeros a row
    assert got["dense_cols"] == 2314


def test_the_process_peak_is_not_below_a_phase_it_contains():
    pytest.importorskip(
        "linopy", reason="the comparison benchmarks need the bench extra"
    )
    from bench_vs_linopy import transport
    from compare import measure

    got = measure(
        transport(n_plants=200, n_warehouses=100, arcs_per_plant=10), traced=False
    )
    reached = [value for name, value in got.items() if name.endswith("_rss_peak_mb")]
    assert reached
    # getrusage counts kibibytes; a process peak below a phase it contains is
    # the unit and not the memory
    assert got["process_rss_mb"] >= max(reached)
