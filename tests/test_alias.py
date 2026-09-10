import numpy as np
import pytest

from nimopt import Alias, Definition, Model, Param, Set, Sum


def test_an_alias_shares_the_sets_labels_and_coordinate():
    N = Set("N", np.array(["a", "b", "c"]))
    NP = Alias("NP", N)
    assert NP.name == "NP"
    assert len(NP) == 3
    assert NP.coord is N.coord
    assert NP.labels is N.labels


def test_a_parameter_over_a_set_and_its_alias_is_two_dimensional():
    N = Set("N", np.array(["a", "b", "c"]))
    NP = Alias("NP", N)
    d = Param.from_dense("d", (N, NP), np.arange(9.0).reshape(3, 3))
    assert d.dims == ("N", "NP")
    assert d.nnz == 9


def test_a_constraint_sums_over_the_alias():
    N = Set("N", np.array(["a", "b", "c"]))
    NP = Alias("NP", N)
    m = Model("alias")
    x = m.var("x", (N, NP))
    d = Param.from_dense("d", (N, NP), np.arange(9.0).reshape(3, 3))
    rhs = Param.from_dense("r", (N,), np.ones(3))
    m.constraint("row", Sum(NP, d[N, NP] * x[N, NP]) <= rhs[N])
    assembled = m.assemble()
    assert m.n_rows == 3
    assert m.n_columns == 9
    # row 0 carries the first row of d, one coefficient per member of NP
    row0 = assembled.values[assembled.indptr[0] : assembled.indptr[1]]
    assert np.array_equal(row0, np.array([0.0, 1.0, 2.0]))


def test_rename_and_transpose_supply_the_transpose():
    N = Set("N", np.array(["a", "b", "c"]))
    NP = Alias("NP", N)
    values = np.arange(9.0).reshape(3, 3)
    d = Param.from_dense("d", (N, NP), values)
    flipped = Param(
        "dT", (N, NP), d.array.rename({"N": "NP", "NP": "N"}).transpose("N", "NP")
    )
    assert np.array_equal(flipped.array.to_dense(), values.T)


def test_an_alias_is_lagged_like_a_set():
    N = Set("N", np.array(["a", "b", "c"]))
    NP = Alias("NP", N)
    lagged = NP - 1
    assert lagged.name == "NP"
    assert lagged.shift == 1
    assert lagged.mode == "drop"
    assert (NP.cyclic - 1).mode == "wrap"


def test_an_alias_refuses_a_reference_naming_the_base_set():
    N = Set("N", np.array(["a", "b", "c"]))
    NP = Alias("NP", N)
    d = Param.from_dense("d", (N, NP), np.arange(9.0).reshape(3, 3))
    with pytest.raises(ValueError, match="declared over"):
        d[N, N]


def test_a_definition_declares_an_alias_over_one_of_its_sets():
    d = Definition("network")
    N = d.set("N")
    NP = d.alias("NP", N)
    assert NP.base is N
    assert d.aliases == {"NP": NP}


def test_an_alias_names_a_base_the_definition_declares():
    d = Definition("network")
    other = Set("Q")
    with pytest.raises(ValueError, match="which is not a set of definition"):
        d.alias("QP", other)


def test_an_alias_shares_the_one_namespace_every_symbol_is_named_in():
    d = Definition("network")
    N = d.set("N")
    d.alias("NP", N)
    with pytest.raises(ValueError, match="already declared as an? alias"):
        d.param("NP", (N,))
    with pytest.raises(ValueError, match="already declared as a set"):
        d.alias("N", N)


def test_an_alias_binds_with_the_set_it_names():
    d = Definition("network", sense="min")
    N = d.set("N")
    NP = d.alias("NP", N)
    flow = d.var("flow", (N, NP), lower=0.0)
    limit = d.param("limit", (N, NP))
    d.constraint("cap", flow[N, NP] <= limit[N, NP])
    d.set_objective(Sum(N, NP, limit[N, NP] * flow[N, NP]))
    m = d.build({"N": np.array(["a", "b"]), "limit": np.ones((2, 2))})
    assert m.n_columns == 4
    assert m.constraints["cap"].n_rows == 4


def test_a_definition_with_an_alias_round_trips_through_its_file():
    from nimopt import loads

    d = Definition("network", sense="min")
    N = d.set("N")
    NP = d.alias("NP", N)
    flow = d.var("flow", (N, NP), lower=0.0)
    limit = d.param("limit", (N, NP))
    d.constraint("cap", flow[N, NP] <= limit[N, NP])
    text = d.to_yaml()
    assert "aliases:\n  NP: N\n" in text
    assert loads(text).to_yaml() == text


def test_a_model_writes_the_aliases_it_is_declared_over_apart_from_its_sets():
    N = Set("N", np.array(["a", "b"]))
    NP = Alias("NP", N)
    m = Model("network")
    x = m.var("x", (N, NP))
    limit = Param.from_dense("limit", (N, NP), np.ones((2, 2)))
    m.constraint("cap", x[N, NP] <= limit[N, NP])
    text = m.to_yaml()
    assert "sets: [N]\naliases:\n  NP: N\n" in text
    assert "NP of N" in repr(m.explain())


def test_an_alias_takes_no_data_of_its_own():
    d = Definition("network", sense="min")
    N = d.set("N")
    NP = d.alias("NP", N)
    d.var("flow", (N, NP), lower=0.0)
    with pytest.raises(ValueError, match=r"data names undeclared \['NP'\]"):
        d.build({"N": np.array(["a", "b"]), "NP": np.array(["a", "b"])})


def test_a_file_naming_an_alias_base_it_does_not_declare_is_refused():
    from nimopt import loads

    with pytest.raises(ValueError, match="names set 'Q', which the file does not"):
        loads("version: 2\nname: d\nsense: min\nsets: [N]\naliases:\n  NP: Q\n")


def test_a_model_writes_the_base_of_every_alias_it_is_declared_over():
    from nimopt import loads

    B = Set("B", np.array(["b1", "b2"]))
    B2 = Alias("B2", B)
    m = Model("net")
    f = m.var("f", (B2,))
    limit = Param.from_dense("limit", (B2,), np.ones(2))
    m.constraint("cap", f[B2] <= limit[B2])
    text = m.to_yaml()
    assert "sets: [B]" in text
    assert loads(text).to_yaml() == text
