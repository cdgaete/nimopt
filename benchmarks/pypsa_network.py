"""A PyPSA network restated in nimopt, from the arrays it was extracted to.

PyPSA leaves every variable free and states each operating limit as a row, so
the columns here carry no finite bound either. The one exception is a storage
unit's spill, which PyPSA bounds by the inflow and states no row for.

The variables carry PyPSA's own names, so a family checked here and the family
it is checked against are reached by the same key.
"""

import numpy as np

from nimopt import Model, Param, Set, Sum, product, subset

FREE = {"lower": -np.inf, "upper": np.inf}


def _first_hours(arrays, hours):
    """Every array cut to the first `hours` snapshots.

    A grid over the snapshots is its last axis, a weighting is one value an
    hour, and everything else stands as it is.
    """
    held = len(arrays["snapshots"])
    if hours > held:
        raise ValueError(f"the extraction carries {held} snapshots; {hours} asked for")
    cut = {}
    for name, value in arrays.items():
        if name == "snapshots" or name.startswith("weighting_"):
            cut[name] = value[:hours]
        elif name.endswith("_t"):
            cut[name] = value[:, :hours]
        else:
            cut[name] = value
    return cut


NOMINAL = (
    ("generators", "Generator", "p_nom", "G"),
    ("lines", "Line", "s_nom", "L"),
    ("links", "Link", "p_nom", "K"),
    ("stores", "Store", "e_nom", "E"),
)


class Data:
    """The arrays a network was extracted to, reached by name.

    `snapshots` takes the first that many hours, so one extraction serves
    every horizon a sweep measures: each grid over the snapshots is cut, and
    so is every weighting.
    """

    def __init__(self, path, snapshots=None):
        arrays = dict(np.load(path, allow_pickle=False))
        if snapshots is not None:
            arrays = _first_hours(arrays, snapshots)
        self._arrays = arrays

    def __getattr__(self, name):
        try:
            return self.__dict__["_arrays"][name]
        except KeyError:
            raise AttributeError(f"the extraction carries no {name!r}") from None

    def __contains__(self, name):
        return name in self._arrays


def sets(data):
    """The model's sets, keyed by the letter the restatement names them by."""
    return {
        "B": Set("B", data.buses),
        "G": Set("G", data.generators_names),
        "L": Set("L", data.lines_names),
        "K": Set("K", data.links_names),
        "E": Set("E", data.stores_names),
        "U": Set("U", data.storage_units_names),
        "T": Set("T", data.snapshots),
        "C": Set("C", np.arange(data.cycles.shape[1])),
    }


def align(grid, names, order):
    """A `(component, snapshot)` grid rearranged onto `order`, missing as zero."""
    at = {name: row for row, name in enumerate(names)}
    out = np.zeros((len(order), grid.shape[1]))
    for row, name in enumerate(order):
        if name in at:
            out[row] = grid[at[name]]
    return out


def extendable(dim, names, flag):
    """The members a nominal capacity is a column for."""
    return subset((dim,), {dim.name: names[flag.astype(bool)]})


def columns(model, data, dims):
    """Declare every variable PyPSA states, by its own name."""
    made = {}
    nominal = (
        ("Generator-p_nom", "G", "generators", "p_nom_extendable"),
        ("Line-s_nom", "L", "lines", "s_nom_extendable"),
        ("Link-p_nom", "K", "links", "p_nom_extendable"),
        ("Store-e_nom", "E", "stores", "e_nom_extendable"),
    )
    for name, key, component, flag in nominal:
        dim = dims[key]
        names = getattr(data, f"{component}_names")
        flag = getattr(data, f"{component}_{flag}")
        made[name] = model.var(
            name, (dim,), subset=extendable(dim, names, flag), **FREE
        )

    for name, dim in (
        ("Generator-p", dims["G"]),
        ("Line-s", dims["L"]),
        ("Link-p", dims["K"]),
        ("Store-e", dims["E"]),
        ("Store-p", dims["E"]),
        ("StorageUnit-p_dispatch", dims["U"]),
        ("StorageUnit-p_store", dims["U"]),
        ("StorageUnit-state_of_charge", dims["U"]),
    ):
        made[name] = model.var(name, (dim, dims["T"]), **FREE)

    inflow = align(
        data.storage_units_inflow_t,
        data.storage_units_inflow_t_names,
        data.storage_units_names,
    )
    # a unit no water reaches spills nothing, so it is no column
    spilling = np.abs(inflow).sum(axis=1) > 0.0
    made["StorageUnit-spill"] = model.var(
        "StorageUnit-spill",
        (dims["U"], dims["T"]),
        subset=member_hours(
            dims["U"],
            dims["T"],
            data.storage_units_names,
            spilling,
            inflow.shape[1],
        ),
        lower=0.0,
        upper=Param.from_dense("inflow", (dims["U"], dims["T"]), inflow),
    )
    return made


def over_members(name, dim, names, values):
    """A parameter over the members carrying a finite value."""
    live = np.isfinite(values)
    return Param.from_long(name, (dim,), {dim.name: names[live]}, values[live])


def nominal_bounds(model, data, dims, made):
    """The rows a nominal capacity is bounded by, where the bound is finite.

    A component whose nominal maximum is infinite is bounded on one side
    only, and PyPSA states no row for the other.
    """
    for component, label, nom, key in NOMINAL:
        dim = dims[key]
        names = getattr(data, f"{component}_names")
        variable = made[f"{label}-{nom}"]
        for side, sense in (("min", ">="), ("max", "<=")):
            values = getattr(data, f"{component}_{nom}_{side}")
            if not np.isfinite(values).any():
                continue
            suffix = "lower" if side == "min" else "upper"
            name = f"{label}-ext-{nom}-{suffix}"
            bound = over_members(name, dim, names, values)
            if sense == ">=":
                model.constraint(name, variable[dim] >= bound[dim])
            else:
                model.constraint(name, variable[dim] <= bound[dim])


def over_grid(name, dims, keys, names, grid, keep):
    """A parameter over the kept members and every snapshot."""
    rows = np.flatnonzero(keep)
    member = np.repeat(names[rows], grid.shape[1])
    snapshot = np.tile(np.arange(grid.shape[1]), rows.size)
    columns = {dims[keys[0]].name: member, dims[keys[1]].name: snapshot}
    return Param.from_long(
        name, (dims[keys[0]], dims[keys[1]]), columns, grid[rows].ravel()
    )


def fixed_operational(model, data, dims, made):
    """The rows a component of fixed capacity operates within.

    PyPSA states a row for each side of each fixed component, so the limit is
    a right-hand side rather than a bound and the row exists wherever the
    component's capacity is not a variable.
    """
    hours = len(data.snapshots)
    ones = np.ones(hours)

    for label, component, key, nom, operating, stem in FIXED_OPERATIONAL:
        names = getattr(data, f"{component}_names")
        keep = ~getattr(data, f"{component}_{nom}_extendable").astype(bool)
        if not keep.any():
            continue
        existing = getattr(data, f"{component}_{nom}")
        variable = made[operating]
        dim = dims[key]
        attr = nom.split("_")[0]
        for suffix, sense, side in (("lower", ">=", "min"), ("upper", "<=", "max")):
            name = f"{label}-fix-{attr}-{suffix}"
            per_hour = profile(data, component, f"{stem}_{side}_pu", names, hours)
            grid = per_hour * existing[:, None]
            bound = over_grid(name, dims, (key, "T"), names, grid, keep)
            relation = (
                variable[dim, dims["T"]] >= bound[dim, dims["T"]]
                if sense == ">="
                else variable[dim, dims["T"]] <= bound[dim, dims["T"]]
            )
            model.constraint(name, relation)

    store = ~data.storage_units_p_nom_extendable.astype(bool)
    su_nom = data.storage_units_p_nom
    su_hours = data.storage_units_max_hours
    for attr, low, high in (
        ("p_dispatch", np.zeros_like(su_nom), data.storage_units_p_max_pu * su_nom),
        ("p_store", np.zeros_like(su_nom), -data.storage_units_p_min_pu * su_nom),
        ("state_of_charge", np.zeros_like(su_nom), su_hours * su_nom),
    ):
        variable = made[f"StorageUnit-{attr}"]
        for suffix, sense, values in (("lower", ">=", low), ("upper", "<=", high)):
            name = f"StorageUnit-fix-{attr}-{suffix}"
            grid = np.outer(values, ones)
            bound = over_grid(
                name, dims, ("U", "T"), data.storage_units_names, grid, store
            )
            relation = (
                variable[dims["U"], dims["T"]] >= bound[dims["U"], dims["T"]]
                if sense == ">="
                else variable[dims["U"], dims["T"]] <= bound[dims["U"], dims["T"]]
            )
            model.constraint(name, relation)


def member_hours(dim, hours_dim, names, keep, hours):
    """The members a family states rows for, crossed with every snapshot."""
    rows = np.flatnonzero(keep)
    member = np.repeat(names[rows], hours)
    snapshot = np.tile(np.arange(hours), rows.size)
    return subset((dim, hours_dim), {dim.name: member, hours_dim.name: snapshot})


def per_unit(name, dim, hours_dim, names, grid, keep):
    """A per-unit coefficient over the kept members, its zeros absent.

    A zero per-unit value states no capacity term, so the entry is absent
    rather than stored: an absent coefficient contributes no nonzero, and a
    stored zero would contribute one PyPSA does not hold.
    """
    rows = np.flatnonzero(keep)
    member = np.repeat(names[rows], grid.shape[1])
    snapshot = np.tile(np.arange(grid.shape[1]), rows.size)
    values = grid[rows].ravel()
    live = values != 0.0
    return Param.from_long(
        name,
        (dim, hours_dim),
        {dim.name: member[live], hours_dim.name: snapshot[live]},
        values[live],
    )


def extendable_operational(model, data, dims, made):
    """The rows an extendable component operates within.

    Each row bounds the operating variable by a per-unit share of the
    capacity it is a column for, so the capacity's column reaches every hour
    that reads it.
    """
    hours = len(data.snapshots)
    time = dims["T"]

    for label, component, key, nom, operating in (
        ("Generator", "generators", "G", "p_nom", "Generator-p"),
        ("Line", "lines", "L", "s_nom", "Line-s"),
        ("Link", "links", "K", "p_nom", "Link-p"),
        ("Store", "stores", "E", "e_nom", "Store-e"),
    ):
        dim = dims[key]
        names = getattr(data, f"{component}_names")
        flag = "s_nom_extendable" if label == "Line" else f"{nom}_extendable"
        keep = getattr(data, f"{component}_{flag}").astype(bool)
        rows = member_hours(dim, time, names, keep, hours)
        capacity = made[f"{label}-{nom}"]
        variable = made[operating]

        if label == "Line":
            limit = profile(data, component, "s_max_pu", names, hours)
            grids = {"lower": -limit, "upper": limit}
        else:
            stem = "e" if label == "Store" else "p"
            grids = {
                "lower": profile(data, component, f"{stem}_min_pu", names, hours),
                "upper": profile(data, component, f"{stem}_max_pu", names, hours),
            }

        attr = nom.split("_")[0]
        for side, grid in grids.items():
            name = f"{label}-ext-{attr}-{side}"
            share = per_unit(name, dim, time, names, grid, keep)
            body = variable[dim, time] - share[dim, time] * capacity[dim]
            relation = body >= 0.0 if side == "lower" else body <= 0.0
            model.constraint(name, relation, over=rows)


KIRCHHOFF_SCALE = 1e5


BUS_COLUMNS = (
    "generators_bus",
    "stores_bus",
    "storage_units_bus",
    "loads_bus",
    "lines_bus0",
    "lines_bus1",
    "links_bus0",
    "links_bus1",
    "links_bus2",
    "links_bus3",
    "links_bus4",
)


def attached(data):
    """Which buses carry a component, as a mask over the bus set.

    A bus nothing sits on states no balance, because there is nothing there
    to balance.
    """
    carrying = set()
    for column in BUS_COLUMNS:
        labels = getattr(data, column, None)
        if labels is None:
            continue
        carrying |= {str(name) for name in labels if str(name) != ""}
    return np.array([str(name) in carrying for name in data.buses])


def profile(data, component, attr, names, hours):
    """A per-unit attribute over `(member, snapshot)`.

    A member the network varies over the snapshots takes its own row; every
    other takes its static value across the horizon. Reading the static
    column alone would give the right nonzeros and the wrong numbers.
    """
    grid = np.outer(getattr(data, f"{component}_{attr}"), np.ones(hours))
    varying = getattr(data, f"{component}_{attr}_t", None)
    if varying is None:
        return grid
    at = {str(name): row for row, name in enumerate(names)}
    for row, name in enumerate(getattr(data, f"{component}_{attr}_t_names")):
        grid[at[str(name)]] = varying[row]
    return grid


def varies(data, component, attr):
    """The members whose attribute the network varies over the snapshots."""
    return {str(n) for n in getattr(data, f"{component}_{attr}_t_names", [])}


def link_arrivals(data, steady=False):
    """Each bus a link arrives at, with the amount reaching it.

    A link names `bus1` and may name `bus2` onward, each with an efficiency
    of its own. A network whose links reach two buses carries empty labels
    beyond that and places nothing there.
    """
    steady_only = varies(data, "links", "efficiency") if steady else set()
    keep = np.array([str(n) not in steady_only for n in data.links_names])
    first = np.where(keep, data.links_bus1, "")
    arrives = [(first, data.links_efficiency)]
    for nth in (2, 3, 4):
        buses = getattr(data, f"links_bus{nth}", None)
        if buses is None:
            break
        arrives.append((buses, getattr(data, f"links_efficiency{nth}")))
    return arrives


def incidence(name, bus_dim, dim, at_bus, members, values):
    """A coefficient placing each member's column in the bus rows reading it."""
    return Param.from_long(
        name, (bus_dim, dim), {bus_dim.name: at_bus, dim.name: members}, values
    )


def _branch_incidence(name, bus_dim, dim, names, leaves, arrives):
    """A branch leaves one bus and arrives at each of `arrives`.

    `arrives` pairs a bus label per branch with the per-unit amount reaching
    it, so a link naming a third, fourth and fifth bus places a term at each.
    A branch that does not name one of them carries an empty label there and
    places nothing.
    """
    at_bus = [leaves]
    member = [names]
    values = [-np.ones(len(names))]
    for buses, amount in arrives:
        live = np.array([str(b) != "" for b in buses])
        at_bus.append(buses[live])
        member.append(names[live])
        values.append(np.asarray(amount)[live])
    return incidence(
        name,
        bus_dim,
        dim,
        np.concatenate(at_bus),
        np.concatenate(member),
        np.concatenate(values),
    )


def _hourly_arrivals(data, bus_dim, dim, time_dim):
    """The arrivals of the links whose efficiency the network varies.

    Those alone need a coefficient per hour; the rest place one entry each.
    """
    hours = len(data.snapshots)
    moving = varies(data, "links", "efficiency")
    if not moving:
        return None
    grid = profile(data, "links", "efficiency", data.links_names, hours)
    at_bus, member, hour, value = [], [], [], []
    live = np.array([str(n) in moving for n in data.links_names])
    rows = np.flatnonzero(live)
    for row in rows:
        target = str(data.links_bus1[row])
        if target == "":
            continue
        at_bus.extend([target] * hours)
        member.extend([str(data.links_names[row])] * hours)
        hour.extend(range(hours))
        value.extend(grid[row])
    return Param.from_long(
        "link_ends_hourly",
        (bus_dim, dim, time_dim),
        {
            bus_dim.name: np.array(at_bus),
            dim.name: np.array(member),
            time_dim.name: np.array(hour),
        },
        np.array(value),
    )


def _load_grid(data, buses):
    """Every bus's load over the snapshots, zero where no load sits.

    A load the network states once stands at that value across the horizon,
    and one it varies takes its own row. Reading the varying ones alone would
    drop every load stated as a single number.
    """
    hours = len(data.snapshots)
    per_load = profile(data, "loads", "p_set", data.loads_names, hours)
    grid = np.zeros((len(buses), hours))
    at = {str(name): row for row, name in enumerate(buses)}
    for row, bus in enumerate(data.loads_bus):
        grid[at[str(bus)]] += per_load[row]
    return grid


def network_rows(model, data, dims, made):
    """The rows a network states: a balance at every bus, a law on every cycle.

    The balance is stated over every bus and hour, because no bus carries
    every component and a bus a term does not reach still balances.
    """
    bus, time = dims["B"], dims["T"]
    ones_g = np.ones(len(data.generators_names))
    terms = [
        incidence(
            "at_bus", bus, dims["G"], data.generators_bus, data.generators_names, ones_g
        )[bus, dims["G"]]
        * made["Generator-p"][dims["G"], time],
        _branch_incidence(
            "line_ends",
            bus,
            dims["L"],
            data.lines_names,
            data.lines_bus0,
            [(data.lines_bus1, np.ones(len(data.lines_names)))],
        )[bus, dims["L"]]
        * made["Line-s"][dims["L"], time],
        _branch_incidence(
            "link_ends",
            bus,
            dims["K"],
            data.links_names,
            data.links_bus0,
            link_arrivals(data, steady=True),
        )[bus, dims["K"]]
        * made["Link-p"][dims["K"], time],
    ]
    reduced = [
        Sum(dims["G"], terms[0]),
        Sum(dims["L"], terms[1]),
        Sum(dims["K"], terms[2]),
    ]
    hourly = _hourly_arrivals(data, bus, dims["K"], time)
    if hourly is not None:
        flow = made["Link-p"][dims["K"], time]
        reduced.append(Sum(dims["K"], hourly[bus, dims["K"], time] * flow))
    for name, sign, dim_key, component, variable in (
        ("dispatch_at", 1.0, "U", "storage_units", "StorageUnit-p_dispatch"),
        ("store_at", -1.0, "U", "storage_units", "StorageUnit-p_store"),
        ("store_p_at", 1.0, "E", "stores", "Store-p"),
    ):
        dim = dims[dim_key]
        members = getattr(data, f"{component}_names")
        at = getattr(data, f"{component}_bus")
        values = np.full(len(members), sign)
        coefficient = incidence(name, bus, dim, at, members, values)
        reduced.append(Sum(dim, coefficient[bus, dim] * made[variable][dim, time]))

    balance = reduced[0]
    for part in reduced[1:]:
        balance = balance + part
    load = Param.from_dense("load", (bus, time), _load_grid(data, data.buses))
    rows = member_hours(bus, time, data.buses, attached(data), len(data.snapshots))
    model.constraint("Bus-nodal_balance", balance == load[bus, time], over=rows)

    cycles = data.cycles * (data.lines_x_pu_eff * KIRCHHOFF_SCALE)[:, None]
    line, cycle = dims["L"], dims["C"]
    rows, cols = np.nonzero(cycles)
    law = Param.from_long(
        "kirchhoff",
        (cycle, line),
        {cycle.name: cols, line.name: data.lines_names[rows]},
        cycles[rows, cols],
    )
    model.constraint(
        "Kirchhoff-Voltage-Law",
        Sum(line, law[cycle, line] * made["Line-s"][line, time]) == 0.0,
    )


def _carried(model, name, dim, time, made, variable, values, cyclic, names):
    """The terms reading the hour before, cyclic where the member is.

    A member the network does not cycle has no predecessor at the first hour,
    so its term is lagged plainly and states nothing there.
    """
    parts = []
    for kept, lag in ((cyclic, time.cyclic - 1), (~cyclic, time - 1)):
        if not kept.any():
            continue
        which = "cyclic" if kept is cyclic else "plain"
        coefficient = per_unit(f"{name}-{which}", dim, time, names, values, kept)
        parts.append(coefficient[dim, time] * made[variable][dim, lag])
    return parts


def temporal_rows(model, data, dims, made):
    """The rows a state of charge carries from one hour to the next.

    A cyclic member reads the last hour at the first; one that does not cycle
    states no term there. Each coefficient carries its own sign, and a term
    whose coefficient is absent contributes nothing, so the rows are stated
    rather than derived from the terms that reach them.
    """
    time = dims["T"]
    hours = len(data.snapshots)
    weight = data.weighting_stores

    def grid(name, dim, values):
        return Param.from_dense(name, (dim, time), values)

    store = dims["E"]
    names = data.stores_names
    n_stores = len(names)
    now = grid("store_now", store, np.full((n_stores, hours), -1.0))
    power = grid("store_power", store, np.outer(-np.ones(n_stores), weight))
    body = now[store, time] * made["Store-e"][store, time]
    for part in _carried(
        model,
        "store_before",
        store,
        time,
        made,
        "Store-e",
        profile(data, "stores", "standing_loss", names, hours) * -1.0 + 1.0,
        data.stores_e_cyclic.astype(bool),
        names,
    ):
        body = body + part
    body = body + power[store, time] * made["Store-p"][store, time]
    model.constraint("Store-energy_balance", body == 0.0, over=product((store, time)))

    unit = dims["U"]
    names = data.storage_units_names
    count = len(names)
    dispatch = grid(
        "su_dispatch",
        unit,
        np.outer(-1.0 / data.storage_units_efficiency_dispatch, weight),
    )
    charging = per_unit(
        "su_store",
        unit,
        time,
        names,
        np.outer(data.storage_units_efficiency_store, weight),
        np.ones(count, dtype=bool),
    )
    soc_now = grid("su_now", unit, np.full((count, hours), -1.0))
    inflow = align(
        data.storage_units_inflow_t,
        data.storage_units_inflow_t_names,
        names,
    )
    # a unit no water reaches spills nothing, so it states no spill term
    spilling = np.abs(inflow).sum(axis=1) > 0.0
    spilled = per_unit(
        "su_spill", unit, time, names, np.outer(-np.ones(count), weight), spilling
    )
    body = dispatch[unit, time] * made["StorageUnit-p_dispatch"][unit, time]
    body = body + charging[unit, time] * made["StorageUnit-p_store"][unit, time]
    body = body + soc_now[unit, time] * made["StorageUnit-state_of_charge"][unit, time]
    for part in _carried(
        model,
        "su_before",
        unit,
        time,
        made,
        "StorageUnit-state_of_charge",
        np.ones((count, hours)),
        data.storage_units_cyclic_state_of_charge.astype(bool),
        names,
    ):
        body = body + part
    body = body + spilled[unit, time] * made["StorageUnit-spill"][unit, time]
    model.constraint(
        "StorageUnit-energy_balance",
        body
        == Param.from_dense("su_inflow", (unit, time), -inflow * weight)[unit, time],
        over=product((unit, time)),
    )


def emission_rate(data):
    """Each generator's emissions per unit of output, zero where it emits none.

    A carrier that emits nothing contributes no term, so the rate is left at
    zero there rather than divided by an efficiency it does not need.
    """
    carriers = list(data.carriers)
    per_carrier = {name: data.carriers_co2[i] for i, name in enumerate(carriers)}
    co2 = np.array([per_carrier.get(c, 0.0) for c in data.generators_carrier])
    rate = np.zeros_like(co2)
    emits = co2 > 0.0
    rate[emits] = co2[emits] / data.generators_efficiency[emits]
    return rate


def energy_sum_rows(model, data, dims, made):
    """The rows a generator's output over the whole horizon is bounded by.

    A row exists where the bound is finite, as a nominal bound row does.
    """
    generator, time = dims["G"], dims["T"]
    names = data.generators_names
    grid = np.outer(np.ones(len(names)), data.weighting_generators)
    output = made["Generator-p"]
    for side, sense in (("min", ">="), ("max", "<=")):
        values = getattr(data, f"generators_e_sum_{side}")
        live = np.isfinite(values)
        if not live.any():
            continue
        name = f"Generator-e_sum_{side}"
        weighted = per_unit(name, generator, time, names, grid, live)
        body = Sum(time, weighted[generator, time] * output[generator, time])
        bound = Param.from_long(
            f"{name}-bound", (generator,), {generator.name: names[live]}, values[live]
        )
        model.constraint(
            name,
            body >= bound[generator] if sense == ">=" else body <= bound[generator],
        )


def _primary_energy(model, data, dims, made, label, carrier, sense, constant):
    """A limit on what burning a carrier emits, over everything that burns it."""
    generator, time = dims["G"], dims["T"]
    grid = np.outer(emission_rate(data), data.weighting_generators)
    keep = np.ones(len(data.generators_names), dtype=bool)
    rate = per_unit(label, generator, time, data.generators_names, grid, keep)
    output = made["Generator-p"]
    body = Sum(generator, time, rate[generator, time] * output[generator, time])
    model.constraint(label, body <= constant if sense == "<=" else body >= constant)


def _operational_limit(model, data, dims, made, label, carrier, sense, constant):
    """A limit on what a carrier's stores hold when the horizon ends."""
    store, time = dims["E"], dims["T"]
    live = np.flatnonzero(data.stores_carrier == carrier)
    if live.size == 0:
        raise ValueError(
            f"{label!r} limits carrier {carrier!r}, which no store carries"
        )
    last = len(data.snapshots) - 1
    coefficient = Param.from_long(
        label,
        (store, time),
        {
            store.name: data.stores_names[live],
            time.name: np.full(live.size, last),
        },
        -np.ones(live.size),
    )
    held = made["Store-e"]
    body = Sum(store, time, coefficient[store, time] * held[store, time])
    model.constraint(label, body >= constant if sense == ">=" else body <= constant)


# a carbon budget PyPSA carries on a store of its own states no row here
GLOBAL_TYPES = {
    "primary_energy": _primary_energy,
    "operational_limit": _operational_limit,
    "co2_atmosphere": None,
}


def global_rows(model, data, dims, made):
    """The rows the network's own global constraints state."""
    for name, kind, carrier, sense, constant in zip(
        data.global_names,
        data.global_type,
        data.global_carrier,
        data.global_sense,
        data.global_constant,
    ):
        if kind not in GLOBAL_TYPES:
            raise ValueError(
                f"global constraint {name!r} is of type {kind!r}, which this "
                f"restatement does not state"
            )
        state = GLOBAL_TYPES[kind]
        if state is None:
            continue
        label = f"GlobalConstraint-{name}"
        state(model, data, dims, made, label, carrier, sense, constant)


FIXED_OPERATIONAL = (
    ("Generator", "generators", "G", "p_nom", "Generator-p", "p"),
    ("Link", "links", "K", "p_nom", "Link-p", "p"),
    ("Store", "stores", "E", "e_nom", "Store-e", "e"),
)

MARGINAL = (
    ("generators", "G", "Generator-p"),
    ("links", "K", "Link-p"),
    ("stores", "E", "Store-p"),
    ("storage_units", "U", "StorageUnit-p_dispatch"),
)


def objective(model, data, dims, made):
    """What the model minimises: capacity built, and energy run.

    A capital cost prices each nominal variable and a marginal cost prices
    each operating one, weighted by its snapshot. A cost of zero states no
    term, so the coefficient carries its zeros as absent.
    """
    parts = []
    for component, label, nom, key in NOMINAL:
        dim = dims[key]
        names = getattr(data, f"{component}_names")
        keep = getattr(data, f"{component}_{nom}_extendable").astype(bool)
        cost = getattr(data, f"{component}_capital_cost")
        live = keep & (cost != 0.0)
        if not live.any():
            continue
        price = Param.from_long(
            f"{label}-capital", (dim,), {dim.name: names[live]}, cost[live]
        )
        parts.append(Sum(dim, price[dim] * made[f"{label}-{nom}"][dim]))

    for component, key, variable in MARGINAL:
        cost = getattr(data, f"{component}_marginal_cost")
        if not (cost != 0.0).any():
            continue
        dim = dims[key]
        names = getattr(data, f"{component}_names")
        grid = np.outer(cost, data.weighting_objective)
        keep = np.ones(len(names), dtype=bool)
        price = per_unit(f"{variable}-marginal", dim, dims["T"], names, grid, keep)
        run = made[variable]
        parts.append(Sum(dim, dims["T"], price[dim, dims["T"]] * run[dim, dims["T"]]))

    total = parts[0]
    for part in parts[1:]:
        total = total + part
    model.set_objective(total)


def constant(path):
    """What the existing capacity of the extendable components already costs.

    PyPSA holds this apart from the objective it reports, so a comparison
    against that number subtracts it.
    """
    data = Data(path)
    total = 0.0
    for component, _, nom, _ in NOMINAL:
        keep = getattr(data, f"{component}_{nom}_extendable").astype(bool)
        cost = getattr(data, f"{component}_capital_cost")
        existing = getattr(data, f"{component}_{nom}")
        total += float((cost[keep] * existing[keep]).sum())
    return total


def model_from(data):
    """The model `data` states, declared and ready to assemble."""
    dims = sets(data)
    model = Model("pypsa")
    made = columns(model, data, dims)
    nominal_bounds(model, data, dims, made)
    fixed_operational(model, data, dims, made)
    extendable_operational(model, data, dims, made)
    network_rows(model, data, dims, made)
    temporal_rows(model, data, dims, made)
    energy_sum_rows(model, data, dims, made)
    global_rows(model, data, dims, made)
    objective(model, data, dims, made)
    return model


def build(path, snapshots=None):
    """The network at `path` as a nimopt model, over its first `snapshots`."""
    return model_from(Data(path, snapshots))
