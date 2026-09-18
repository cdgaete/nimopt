import sys
from pathlib import Path

import numpy as np
import pytest

from nimopt.model import Model
from nimopt.param import Param
from nimopt.sets import Set, product, subset
from nimopt.term import Sum

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))


def transport_parts(n_plants=200, n_warehouses=100, density=1.0, seed=0):
    """The sets and parameters a transport model is stated over.

    The data is built here rather than inside the model so that attributing
    a build's allocations measures the build alone. The sets fix the rows and
    the columns, and the coefficient's density alone sets the nonzeros, so
    two densities vary the matrix content and nothing else.
    """
    P = Set("P", np.array([f"p{i}" for i in range(n_plants)]))
    W = Set("W", np.array([f"w{i}" for i in range(n_warehouses)]))
    grid = np.stack(
        np.meshgrid(np.arange(n_plants), np.arange(n_warehouses), indexing="ij"), 0
    ).reshape(2, -1)
    keep = np.random.default_rng(seed).random(grid.shape[1]) < density
    # one entry per plant, so every supply row stands at every density
    keep[::n_warehouses] = True
    columns = {
        "P": np.array([f"p{i}" for i in grid[0][keep]]),
        "W": np.array([f"w{j}" for j in grid[1][keep]]),
    }
    cost = Param.from_long("c", (P, W), columns, np.ones(int(keep.sum())))
    supply = Param.from_dense("s", (P,), np.full(n_plants, 100.0))
    return P, W, cost, supply


def build_transport(parts):
    """A transport model over `parts`, declared but not assembled."""
    P, W, cost, supply = parts
    m = Model("transport")
    x = m.var("x", (P, W))
    m.constraint("supply", Sum(W, cost[P, W] * x[P, W]) <= supply[P])
    m.set_objective(Sum(P, W, cost[P, W] * x[P, W]))
    return m


def test_nimopt_allocates_nothing_per_nonzero():
    from ownership import attribute

    full = transport_parts(density=1.0)
    tenth = transport_parts(density=0.1)
    dense = attribute(lambda: build_transport(full).assemble())
    sparse = attribute(lambda: build_transport(tenth).assemble())
    added = build_transport(full).nnz - build_transport(tenth).nnz

    # the matrix costs an int32 column and a float64 value per entry
    assert dense["nimblend"] > sparse["nimblend"] + 12 * added, (dense, sparse)
    assert dense["nimopt"] == pytest.approx(sparse["nimopt"], rel=0.01), (dense, sparse)


def test_nimopt_allocates_the_solver_vectors_and_nothing_else():
    from ownership import attribute

    parts = transport_parts()
    model = build_transport(parts)
    owned = attribute(lambda: build_transport(parts).assemble())
    # lower, upper and integrality per column; the bounds per row
    ceiling = 24 * model.n_columns + 64 * model.n_rows
    assert owned["nimopt"] < ceiling, (owned["nimopt"], ceiling)


def test_the_matrix_bytes_belong_to_nimblend():
    from ownership import attribute

    parts = transport_parts()
    assembled = build_transport(parts).assemble()
    matrix = (
        assembled.indices.nbytes + assembled.values.nbytes + assembled.indptr.nbytes
    )
    owned = attribute(lambda: build_transport(parts).assemble())
    assert owned["nimblend"] >= matrix, (owned["nimblend"], matrix)


def test_the_peak_grows_only_with_what_the_matrix_costs():
    from ownership import peak_bytes

    full = transport_parts(density=1.0)
    tenth = transport_parts(density=0.1)
    added = build_transport(full).nnz - build_transport(tenth).nnz
    grew = peak_bytes(lambda: build_transport(full).assemble()) - peak_bytes(
        lambda: build_transport(tenth).assemble()
    )
    # the matrix costs 16 bytes an entry and the merge that builds it holds its
    # ravelled keys and a mask beside them: 46.5 bytes an entry, the same to a
    # hundredth of a byte on every run. A float64 transient held where the peak
    # is reads 54.5, so this catches one whether or not the build frees it.
    assert grew / added < 50.0, grew / added


def network_parts(n_buses=200, n_links=400, n_hours=24, density=1.0, seed=0):
    """Return the sets and the incidence a network balance is declared over.

    The rows are every bus and hour, and the columns are every link and hour.
    Neither count moves with the incidence. The number of link-hours in the
    incidence sets the nonzeros. Each one places the link's flow in the two
    buses it joins.
    """
    B = Set("B", np.array([f"b{i}" for i in range(n_buses)]))
    L = Set("L", np.array([f"l{i}" for i in range(n_links)]))
    T = Set("T", np.arange(n_hours))
    tail = np.arange(n_links) % n_buses
    head = (np.arange(n_links) + 1) % n_buses
    link = np.repeat(np.arange(n_links), n_hours)
    hour = np.tile(np.arange(n_hours), n_links)
    keep = np.random.default_rng(seed).random(link.size) < density
    link, hour = link[keep], hour[keep]
    columns = {
        "B": np.array([f"b{i}" for i in np.concatenate([tail[link], head[link]])]),
        "L": np.array([f"l{i}" for i in np.concatenate([link, link])]),
        "T": np.concatenate([hour, hour]),
    }
    values = np.concatenate([-np.ones(link.size), np.full(link.size, 0.9)])
    inc = Param.from_long("inc", (B, L, T), columns, values)
    zero = Param.from_dense("zero", (B, T), np.zeros((n_buses, n_hours)))
    cost = Param.from_dense("cost", (L, T), np.ones((n_links, n_hours)))
    return B, L, T, inc, zero, cost


def build_network(parts):
    """A network balance over `parts`, declared but not assembled."""
    B, L, T, inc, zero, cost = parts
    m = Model("network")
    flow = m.var("flow", (L, T), lower=-np.inf)
    m.constraint(
        "balance",
        Sum(L, inc[B, L, T] * flow[L, T]) == zero[B, T],
        over=product((B, T)),
    )
    m.set_objective(Sum(L, T, cost[L, T] * flow[L, T]))
    return m


def test_an_incidence_allocates_nothing_per_nonzero_in_nimopt():
    from ownership import attribute

    full = network_parts(density=1.0)
    tenth = network_parts(density=0.1)
    dense = attribute(lambda: build_network(full).assemble())
    sparse = attribute(lambda: build_network(tenth).assemble())
    added = build_network(full).nnz - build_network(tenth).nnz

    # the matrix costs an int32 column and a float64 value per entry
    assert dense["nimblend"] > sparse["nimblend"] + 12 * added, (dense, sparse)
    # nimopt holds a vector per row and per column and both are fixed here, so
    # what it allocates must not track the nonzeros the incidence adds. A byte
    # an entry is already below the four an int32 column would cost.
    assert (dense["nimopt"] - sparse["nimopt"]) / added < 1.0, (dense, sparse)


def piecewise_parts(n_units, n_points=4):
    """Return the sets and breakpoints of a curve declared for one unit.

    The breakpoint data is four entries for `u0` whatever `n_units` is, so
    the generated model is the same at every extent. Only the extent of the
    unit set moves.
    """
    U = Set("U", np.array([f"u{i}" for i in range(n_units)]))
    K = Set("K", np.arange(n_points))
    columns = {"U": np.array(["u0"] * n_points), "K": np.arange(n_points)}
    x = np.linspace(0.0, 30.0, n_points)
    xp = Param.from_long("xp", (U, K), columns, x)
    yp = Param.from_long("yp", (U, K), columns, x * 2.0)
    held = subset((U,), {"U": np.array(["u0"])})
    return U, K, xp, yp, held


def build_curve(parts, declared):
    """A model over `parts`, with the piecewise declaration or without it."""
    U, K, xp, yp, held = parts
    m = Model("curve")
    p = m.var("p", (U,), subset=held, upper=30.0)
    c = m.var("c", (U,), subset=held, lower=-1e4)
    if declared:
        m.piecewise(
            "pw",
            x=p[U],
            x_points=xp[U, K],
            y=c[U],
            y_points=yp[U, K],
            sign=">=",
            method="incremental",
        )
    m.set_objective(Sum(U, c[U]))
    return m


def declaration_bytes(n_units, runs=3):
    """Return the peak the piecewise declaration adds at `n_units`.

    Both sides build the same sets and the same data outside the traced call,
    so the members of the unit set cost the same on each, and the difference
    is the declaration alone. The least of `runs` peaks is returned on each
    side because a peak varies between runs.
    """
    from ownership import peak_bytes

    parts = piecewise_parts(n_units)
    with_it = min(
        peak_bytes(lambda: build_curve(parts, True).assemble()) for _ in range(runs)
    )
    without = min(
        peak_bytes(lambda: build_curve(parts, False).assemble()) for _ in range(runs)
    )
    return with_it - without


def test_a_piecewise_declaration_allocates_nothing_per_absent_member():
    # the breakpoints are four entries for one unit at both extents, and the
    # generated model is identical, so what the declaration costs must not
    # track the members that have no breakpoint
    small = build_curve(piecewise_parts(10), True).assemble()
    large = build_curve(piecewise_parts(10000), True).assemble()
    assert (small.n_rows, small.n_cols) == (large.n_rows, large.n_cols)
    assert small.values.size == large.values.size

    grew = declaration_bytes(10000) / declaration_bytes(10)
    assert grew < 3.0, grew
