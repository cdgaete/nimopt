import numpy as np

from nimopt import Model, Param, Set, Sum, product, subset


def test_a_capacity_column_is_bound_by_a_time_varying_availability():
    # one generator over two hours: available fully at t0, at half at t1.
    # demand is 100 in both hours, so the half hour sets capacity at 200.
    G = Set("G", np.array(["g0"]))
    T = Set("T", np.arange(2))
    m = Model("expansion")
    cap = m.var("cap", (G,))
    gen = m.var("gen", (G, T))

    ones_gt = Param.from_dense("ones_gt", (G, T), np.ones((1, 2)))
    avail = Param.from_dense("avail", (G, T), np.array([[1.0, 0.5]]))
    invest = Param.from_dense("invest", (G,), np.array([10.0]))
    run = Param.from_dense("run", (G, T), np.ones((1, 2)))
    load = Param.from_dense("load", (T,), np.full(2, 100.0))

    m.constraint("cap_link", ones_gt[G, T] * gen[G, T] - avail[G, T] * cap[G] <= 0.0)
    m.constraint("balance", Sum(G, ones_gt[G, T] * gen[G, T]) == load[T])
    m.set_objective(Sum(G, invest[G] * cap[G]) + Sum(G, T, run[G, T] * gen[G, T]))

    assembled = m.assemble()
    # two cap_link rows and two balance rows; one cap column and two gen
    assert (assembled.n_rows, assembled.n_cols) == (4, 3)
    # two entries in each cap_link row, one in each balance row
    assert int(assembled.values.size) == 6

    answer = m.solve()
    assert answer.status == "optimal"
    # 200 MW at 10/MW, plus 200 MWh at 1/MWh
    assert answer.objective == 2200.0


def test_a_link_column_reaches_the_two_buses_it_joins():
    # l0 carries power from b0 to b1 at 90% efficiency; demand sits at b1
    B = Set("B", np.array(["b0", "b1"]))
    G = Set("G", np.array(["g0"]))
    L = Set("L", np.array(["l0"]))
    T = Set("T", np.arange(2))
    m = Model("network")
    at_bus = subset(
        (B, G, T),
        {
            "B": np.array(["b0", "b0"]),
            "G": np.array(["g0", "g0"]),
            "T": np.array([0, 1]),
        },
    )
    gen = m.var("gen", (B, G, T), subset=at_bus)
    flow = m.var("flow", (L, T))

    ones_bgt = Param.from_dense("ones_bgt", (B, G, T), np.ones((2, 1, 2)))
    cost = Param.from_dense("cost", (B, G, T), np.ones((2, 1, 2)))
    grid = np.zeros((2, 1, 2))
    grid[0, 0, :] = -1.0
    grid[1, 0, :] = 0.9
    inc = Param.from_dense("inc", (B, L, T), grid)
    load = Param.from_dense("load", (B, T), np.array([[0.0, 0.0], [10.0, 10.0]]))

    m.constraint(
        "balance",
        Sum(G, ones_bgt[B, G, T] * gen[B, G, T]) + Sum(L, inc[B, L, T] * flow[L, T])
        == load[B, T],
        over=product((B, T)),
    )
    m.set_objective(Sum(B, G, T, cost[B, G, T] * gen[B, G, T]))

    assembled = m.assemble()
    # every bus and hour states a row; two gen columns and two flow
    assert (assembled.n_rows, assembled.n_cols) == (4, 4)
    # the generator's two entries and the link's four
    assert int(assembled.values.size) == 6

    answer = m.solve()
    assert answer.status == "optimal"
    # 10 MW at b1 costs 10/0.9 at b0, in each of two hours
    assert abs(answer.objective - 2 * 10.0 / 0.9) < 1e-9
