"""Regression test for Bug 2: stale var-index cache after variable deletion.

The direct solver's Rust matrix builder cached a name->column map keyed by
`id(model)` in a module-level dict that was never invalidated. Because the MCP
mutates ONE persistent model object across edits, deleting a variable changed the
column layout while `id(model)` stayed the same, so a later build reused stale
columns and silently corrupted the matrix.

The stale map is only consulted on the non-vectorized fallback path (a constraint
with no free sets and multiple terms, e.g. `z == Sum(P, x[P])`), so the model
below is built to take that path.
"""

import numpy as np

import nimopt as no
from nimopt.solvers.highs_direct import _build_matrices_rust


def _build(with_dummy):
    m = no.Model(sense="maximize")
    P = no.Set("P", ["p1", "p2", "p3"])
    if with_dummy:
        # scalar variable declared FIRST -> occupies column 0; deleting it later
        # shifts every following variable's column.
        m.var("dummy", lb=0, ub=7)
    x = m.var("x", [P], lb=0, ub=5)
    z = m.var("z", lb=0, ub=1000)
    # no free sets + two terms -> non-vectorized fallback that uses the var-index map
    m.eq("zdef", z == no.Sum(P, x[P]))
    m.eq("zcap", z <= 9)
    m.set_objective(z)
    return m


def test_delete_variable_rebuilds_matrix_columns():
    # Fresh, never-had-dummy reference. Keep the model object alive for the whole
    # test so CPython cannot recycle its id() for the mutated model below.
    ref_model = _build(with_dummy=False)
    ref = _build_matrices_rust(ref_model)

    # Bug path: build WITH dummy (first build seeds any per-model cache), delete
    # the dummy on the same object (mirrors MCP delete_variable), rebuild.
    m = _build(with_dummy=True)
    _build_matrices_rust(m)
    del m.variables["dummy"]
    after = _build_matrices_rust(m)

    assert after["n_vars"] == ref["n_vars"]
    np.testing.assert_array_equal(after["indptr"], ref["indptr"])
    np.testing.assert_array_equal(after["indices"], ref["indices"])
    np.testing.assert_allclose(after["data"], ref["data"])

    assert ref_model is not None  # keep reference alive
