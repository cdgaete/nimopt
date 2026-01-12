"""Tests for lazy solution storage."""

import tempfile

import pytest

import nimopt as no
from nimopt.solution_lazy import load_solution, save_solution
from nimopt.solvers import HiGHSSolver


@pytest.fixture
def solved_transport():
    """Create and solve transport model."""
    i = no.Set('i', ['seattle', 'sandiego'])
    j = no.Set('j', ['newyork', 'chicago', 'topeka'])
    supply = no.Param('supply', [i], [350, 600])
    demand = no.Param('demand', [j], [325, 300, 275])
    cost = no.Param('cost', [i, j], [[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]])

    m = no.Model(name='transport', sense='minimize')
    x = m.var('x', [i, j], lb=0)
    m.eq('supply', no.Sum(j, x[i, j]) <= supply[i])
    m.eq('demand', no.Sum(i, x[i, j]) >= demand[j])
    m.objective = no.Sum(i, j, cost[i, j] * x[i, j])

    with tempfile.NamedTemporaryFile(suffix='.lp', delete=False) as f:
        lp_file = f.name
    m.to_lp(lp_file)

    solver = HiGHSSolver()
    solver.read_lp(lp_file)
    solver.solve()
    return m, solver


def test_save_and_load(solved_transport):
    """Test save and load roundtrip."""
    m, solver = solved_transport

    with tempfile.TemporaryDirectory() as tmpdir:
        sol = save_solution(solver, m, tmpdir)
        assert sol.variables == ['x']
        assert sol.constraints == ['supply', 'demand']

        # Reload
        sol2 = load_solution(tmpdir)
        assert sol2.variables == ['x']
        assert sol2.constraints == ['supply', 'demand']


def test_lazy_variable_access(solved_transport):
    """Test lazy loading of variable data."""
    m, solver = solved_transport

    with tempfile.TemporaryDirectory() as tmpdir:
        save_solution(solver, m, tmpdir)
        sol = load_solution(tmpdir)

        x = sol.var('x')
        assert x.dims == ['i', 'j']
        assert x.shape == (2, 3)
        assert x.values.shape == (2, 3)
        assert x[0, 1] == pytest.approx(300)  # seattle -> chicago


def test_lazy_constraint_access(solved_transport):
    """Test lazy loading of constraint data."""
    m, solver = solved_transport

    with tempfile.TemporaryDirectory() as tmpdir:
        save_solution(solver, m, tmpdir)
        sol = load_solution(tmpdir)

        supply = sol.con('supply')
        assert supply.dims == ['i']
        assert supply.shape == (2,)

        demand = sol.con('demand')
        assert demand.dims == ['j']
        assert len(demand.duals) == 3
