import subprocess
import sys
from pathlib import Path

import nimblend as nb

import nimopt as no


def test_nimblend_source_never_names_nimopt():
    root = Path(nb.__file__).parent
    offenders = [
        str(path) for path in root.rglob("*.py") if "nimopt" in path.read_text()
    ]
    assert offenders == [], offenders


def test_importing_nimblend_does_not_import_nimopt():
    code = "import nimblend, sys; print('nimopt' in sys.modules)"
    got = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert got.stdout.strip() == "False"


SCIPY = Path(no.__file__).parent / "solvers" / "gurobi.py"


def test_scipy_is_named_only_by_the_adapter_whose_solver_takes_it():
    """The model's own handoff is numpy: scipy is named where a solver's
    matrix interface takes it, and in no other module."""
    root = Path(no.__file__).parent
    offenders = [
        str(path)
        for path in root.rglob("*.py")
        if path != SCIPY and "scipy" in path.read_text()
    ]
    assert offenders == [], offenders


def test_the_adapter_that_names_scipy_drives_a_solver_that_takes_it():
    # the allowance above is load-bearing rather than permissive: it exists
    # because Gurobi's matrix interface takes a SciPy sparse matrix
    assert "scipy" in SCIPY.read_text()


def test_reading_the_adapter_does_not_import_scipy():
    # scipy is a cost of handing a model to Gurobi, not of reading what the
    # adapter declares, which holds only while the import sits inside the
    # function that needs it
    code = (
        "from nimopt.solvers import adapter; adapter('gurobi'); "
        "import sys; print('scipy' in sys.modules)"
    )
    got = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert got.stdout.strip() == "False"


def test_the_reserved_names_cannot_be_ordinary_set_names():
    assert no.COLUMN.startswith("__") and no.COLUMN.endswith("__")
    assert no.ROW.startswith("__") and no.ROW.endswith("__")
    assert no.COLUMN != no.ROW
