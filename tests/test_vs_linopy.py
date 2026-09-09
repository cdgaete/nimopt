import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

pytest.importorskip("linopy", reason="the comparison benchmarks need the bench extra")

SMALL = {
    "transport": dict(n_plants=20, n_warehouses=8, arcs_per_plant=3),
    "storage": dict(n_generators=4, n_storage=2, n_hours=24),
    "transport_milp": dict(n_plants=20, n_warehouses=8, arcs_per_plant=3),
}


@pytest.mark.parametrize("case", ["transport", "storage", "transport_milp"])
def test_the_two_builders_state_the_same_problem(case):
    from bench_vs_linopy import CASES
    from compare import measure

    sides = {
        side: measure(factory(**SMALL[case])) for side, factory in CASES[case].items()
    }
    nimopt, linopy = sides["nimopt"], sides["linopy"]
    for field in ("rows", "cols", "nnz"):
        assert nimopt[field] == linopy[field], (field, nimopt, linopy)
    assert nimopt["objective"] == pytest.approx(linopy["objective"], rel=1e-6)


def test_a_readback_reaches_the_same_values():
    from bench_vs_linopy import CASES
    from compare import measure

    sides = {
        side: measure(factory(**SMALL["transport"]))
        for side, factory in CASES["transport"].items()
    }
    assert sides["nimopt"]["primal_sum"] == pytest.approx(
        sides["linopy"]["primal_sum"], rel=1e-6
    )
    assert sides["nimopt"]["dual_sum"] == pytest.approx(
        sides["linopy"]["dual_sum"], rel=1e-6
    )


def test_a_case_measured_in_a_child_process_reports_the_same_shape():
    from bench_vs_linopy import CASES
    from compare import measure, run

    here = measure(CASES["transport"]["nimopt"](**SMALL["transport"]))
    there = run("transport", "nimopt", SMALL["transport"])
    for field in ("rows", "cols", "nnz"):
        assert here[field] == there[field], (field, here, there)
    assert there["build_rss_mb"] >= 0.0
    assert there["build_ms"] > 0.0


def test_a_sparse_subset_costs_nimopt_far_less_to_build_than_a_masked_product():
    from bench_vs_linopy import compare

    # a variable spanning the arcs against one spanning the full product with
    # a mask over it: 11.4 MB against 110.4 MB resident at this rung, so a
    # third of linopy's figure is a bound a regression breaks, not noise
    sides = compare(
        "transport",
        dict(n_plants=2000, n_warehouses=500, arcs_per_plant=20),
        traced=False,
    )
    assert sides["nimopt"]["build_rss_mb"] < sides["linopy"]["build_rss_mb"] / 3, sides
