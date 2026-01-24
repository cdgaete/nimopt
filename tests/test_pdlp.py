"""Tests for PDLP solver."""

import pytest
import numpy as np
import nimopt as no
from nimopt import Sum
from nimopt.solvers import PDLPSolver, SolverStatus


def test_pdlp_transport():
    """Test PDLP on classic transport problem."""
    i = no.Set('i', ['seattle', 'sandiego'])
    j = no.Set('j', ['newyork', 'chicago', 'topeka'])

    a = no.Param('a', [i], [350, 600])
    b = no.Param('b', [j], [325, 300, 275])
    d = no.Param('d', [i, j], [[2.5, 1.7, 1.8],
                               [2.5, 1.8, 1.4]])

    m = no.Model(sense='minimize')
    x = m.var('x', [i, j], lb=0)
    z = m.var('z')

    m.eq('cost', z == Sum(i, j, d[i,j] * x[i,j]))
    m.eq('supply', Sum(j, x[i,j]) <= a[i])
    m.eq('demand', Sum(i, x[i,j]) >= b[j])
    m.set_objective(z)

    solver = PDLPSolver(backend='ortools', tolerance=1e-4)
    solver.load_model(m)
    result = solver.solve()

    assert result.status == SolverStatus.OPTIMAL
    assert abs(result.objective_value - 1707.5) < 1.0


def test_pdlp_maximize():
    """Test PDLP with maximization."""
    x = no.Set('x', ['a', 'b'])
    c = no.Param('c', [x], [3, 2])
    
    m = no.Model(sense='maximize')
    y = m.var('y', [x], lb=0, ub=10)
    m.set_objective(Sum(x, c[x] * y[x]))
    
    solver = PDLPSolver(backend='ortools')
    solver.load_model(m)
    result = solver.solve()
    
    assert result.status == SolverStatus.OPTIMAL
    assert abs(result.objective_value - 50) < 0.1


def test_pdlp_solution_extraction():
    """Test that solution values can be extracted."""
    i = no.Set('i', ['a', 'b'])
    m = no.Model(sense='minimize')
    x = m.var('x', [i], lb=0, ub=5)
    m.set_objective(Sum(i, x[i]))
    
    solver = PDLPSolver(backend='ortools')
    solver.load_model(m)
    result = solver.solve()
    
    vals = solver.get_variable_values()
    names = solver.get_variable_names()
    
    assert len(vals) == 2
    assert len(names) == 2
    assert all(v >= 0 for v in vals)


def test_pdlp_backend_selection():
    """Test backend selection logic."""
    solver = PDLPSolver(backend='ortools')
    assert solver.backend == 'ortools'
    
    # Auto should select ortools on CPU-only system
    solver = PDLPSolver(backend='auto')
    assert solver.backend in ['ortools', 'cupdlp']
