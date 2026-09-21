import os
import re

import pytest

from nimopt.models import transport

# PDLP on a GPU needs an NVIDIA driver and a HiGHS built from source with CUDA;
# a machine without them fails, so these tests run only when asked to
enabled = pytest.mark.skipif(
    os.environ.get("NIMOPT_TEST_PDLP") != "1",
    reason="PDLP tests are disabled; set NIMOPT_TEST_PDLP=1 on a machine with an "
    "NVIDIA driver and a HiGHS built with CUDA",
)


@enabled
def test_pdlp_runs_through_the_highs_options_and_solves_to_the_optimum(capfd):
    data = transport.data(scale=2)
    model = transport.definition().build(data)
    solution = model.solve(options={"method": "pdlp", "pdlp_tol": 1e-8, "log": True})
    # HiGHS writes a PDLP iteration count in its summary only after a PDLP run
    assert re.search(r"PDLP\s+iterations", capfd.readouterr().out)
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(transport.reference(data), rel=1e-6)


@enabled
def test_pdlp_returns_the_row_duals_simplex_returns():
    data = transport.data(scale=2)
    model = transport.definition().build(data)
    pdlp = model.solve(options={"method": "pdlp", "pdlp_tol": 1e-8})
    simplex = model.solve(options={"method": "simplex"})
    for name in ("supply", "demand"):
        assert pdlp.dual(name).values() == pytest.approx(
            simplex.dual(name).values(), abs=1e-6
        )
