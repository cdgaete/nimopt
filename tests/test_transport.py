"""Test the transport problem."""



def test_transport_basic(tmp_path):
    import nimopt as no

    i = no.Set('i', ['seattle', 'sandiego'])
    j = no.Set('j', ['newyork', 'chicago', 'topeka'])

    a = no.Param('a', [i], [350, 600])
    b = no.Param('b', [j], [325, 300, 275])
    d = no.Param('d', [i, j], [[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]])

    m = no.Model(name='transport', sense='minimize')
    x = m.var('x', [i, j], lb=0)
    z = m.var('z')

    m.eq('cost', z == no.Sum(i, j, d[i, j] * x[i, j]))
    m.set_objective(z)
    m.eq('supply', no.Sum(j, x[i, j]) <= a[i])
    m.eq('demand', no.Sum(i, x[i, j]) >= b[j])

    lp_file = tmp_path / 'transport.lp'
    m.to_lp(str(lp_file))

    assert lp_file.exists()
    content = lp_file.read_text()

    assert 'Minimize' in content
    assert 'Subject To' in content
    assert 'x_seattle_newyork' in content
    assert 'supply' in content
    assert 'demand' in content


def test_param_nimblend():
    import nimblend as nb

    import nimopt as no

    i = no.Set('i', ['a', 'b'])
    j = no.Set('j', [1, 2, 3])

    p = no.Param('p', [i, j], [[1, 2, 3], [4, 5, 6]])

    # Param uses nimblend Array internally
    assert isinstance(p.array, nb.Array)
    assert p.array.dims == ['i', 'j']
    assert p.shape == (2, 3)

    # Indexing
    assert p['a', 1] == 1.0
    assert p['b', 3] == 6.0


def test_set_free_tracking():
    import nimopt as no

    i = no.Set('i', ['a', 'b'])
    j = no.Set('j', [1, 2])

    m = no.Model()
    x = m.var('x', [i, j])

    # VarRef tracks free sets
    ref = x[i, j]
    assert len(ref.free_sets) == 2

    ref2 = x['a', j]
    assert len(ref2.free_sets) == 1


def test_sum_reduces_dimensions():
    import nimopt as no

    i = no.Set('i', ['a', 'b'])
    j = no.Set('j', [1, 2, 3])

    m = no.Model()
    x = m.var('x', [i, j])

    # Sum over j leaves i free
    expr = no.Sum(j, x[i, j])
    assert len(expr.free_sets) == 1
    assert expr.free_sets[0].name == 'i'

    # Sum over both leaves nothing free
    expr2 = no.Sum(i, j, x[i, j])
    assert len(expr2.free_sets) == 0


def test_model_constraint_count():
    import nimopt as no

    i = no.Set('i', ['a', 'b', 'c'])
    j = no.Set('j', [1, 2])

    m = no.Model()
    x = m.var('x', [i, j], lb=0)
    p = no.Param('p', [i], [10, 20, 30])

    # This generates 3 constraints (one per i)
    m.eq('limit', no.Sum(j, x[i, j]) <= p[i])

    assert m.n_constraints == 3
