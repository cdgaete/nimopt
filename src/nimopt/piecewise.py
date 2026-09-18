"""A piecewise-linear relation between two expressions, as generated declarations."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from nimopt.coefficient import Coefficient
from nimopt.param import Param
from nimopt.sets import Set, coords_of, displayed, subset
from nimopt.symbol import read_at_its_sets
from nimopt.term import Expression, Sum

METHODS = ("incremental", "tangent")
SIGNS = ("==", "<=", ">=")
KINDS = ("sets", "parameters", "variables", "constraints")
TOLERANCE = 1e-10


class Piecewise:
    """A piecewise-linear relation of `y` to `x`, through breakpoints.

    `x` lies on the curve through `x_points` and `y_points`. `sign` relates
    `y` to the curve. `generated` maps each kind of declaration to the names
    a model generated for this declaration. Each entry is empty until a model
    generates them.

    Raises TypeError for an `x`, `y` or `active` that is not an expression,
    and for points that are not a coefficient. Raises ValueError for a name
    that is not an identifier, an unknown method or sign, expressions over
    different sets, points over sets other than the sets of `x` and one
    breakpoint set, and a `tangent` declaration with sign `==` or `active`.
    """

    def __init__(
        self,
        name: str,
        x: Any,
        x_points: Any,
        y: Any,
        y_points: Any,
        sign: str,
        method: str,
        active: Any = None,
    ) -> None:
        self.name = str(name)
        if not self.name.isidentifier() or self.name == "Sum":
            raise ValueError(
                f"piecewise name {self.name!r} is not a Python identifier; "
                f"declare an identifier other than Sum"
            )
        if method not in METHODS:
            raise ValueError(
                f"method of piecewise {self.name!r} is one of {METHODS}; got {method!r}"
            )
        if sign not in SIGNS:
            raise ValueError(
                f"sign of piecewise {self.name!r} is one of {SIGNS}; got {sign!r}"
            )
        self.method = method
        self.sign = sign
        self.x = self._expression(x, "x")
        self.y = self._expression(y, "y")
        self.active = None if active is None else self._expression(active, "active")
        self.x_points = self._points(x_points, "x_points")
        self.y_points = self._points(y_points, "y_points")
        self._same_sets(self.y, "y")
        if self.active is not None:
            self._same_sets(self.active, "active")
        outside = [d for d in self.x_points.dims if d not in self.x.frame]
        if len(outside) != 1:
            raise ValueError(
                f"x_points of piecewise {self.name!r} is over "
                f"{self.x_points.dims} and x is over {self.x.frame}; give "
                f"x_points exactly one set that x is not over"
            )
        self.breakpoints = outside[0]
        if set(self.y_points.dims) != set(self.x_points.dims):
            raise ValueError(
                f"y_points of piecewise {self.name!r} is over "
                f"{self.y_points.dims} and x_points is over "
                f"{self.x_points.dims}; give both over the same sets"
            )
        if method == "tangent" and sign == "==":
            raise ValueError(
                f"piecewise {self.name!r} uses method 'tangent' with sign "
                f"'=='; use sign '<=' or '>=', or method 'incremental'"
            )
        if method == "tangent" and self.active is not None:
            raise ValueError(
                f"piecewise {self.name!r} uses method 'tangent' with active; "
                f"use method 'incremental'"
            )
        if method == "tangent" and self.x.constant != 0.0:
            raise ValueError(
                f"x of piecewise {self.name!r} has the constant "
                f"{self.x.constant} and the method is 'tangent'; subtract the "
                f"constant from x_points, or use method 'incremental'"
            )
        if self.active is not None and self.active.constant != 0.0:
            raise ValueError(
                f"active of piecewise {self.name!r} has the constant "
                f"{self.active.constant}; give active as an expression with no "
                f"constant"
            )
        self.generated = {kind: () for kind in KINDS}

    def __repr__(self) -> str:
        return (
            f"Piecewise({self.name!r}, {self.method!r}, {self.sign!r}, "
            f"over {self.breakpoints!r})"
        )

    def _expression(self, given: Any, slot: str) -> Expression:
        given = read_at_its_sets(given)
        if not isinstance(given, Expression):
            raise TypeError(
                f"{slot} of piecewise {self.name!r} is an expression; got "
                f"{type(given).__name__}"
            )
        return given

    def _points(self, given: Any, slot: str) -> Coefficient:
        if not isinstance(given, Coefficient):
            raise TypeError(
                f"{slot} of piecewise {self.name!r} is a parameter read at its "
                f"sets; got {type(given).__name__}"
            )
        return given

    def _same_sets(self, given: Expression, slot: str) -> None:
        if set(given.frame) != set(self.x.frame):
            raise ValueError(
                f"{slot} of piecewise {self.name!r} is over {given.frame} and "
                f"x is over {self.x.frame}; give both over the same sets"
            )

    def check_domain(self) -> None:
        """Raise ValueError where y or active spans other coordinates than x.

        The expressions materialise to compare their coordinates. Every
        variable they read requires its columns.
        """
        _, rows = self.x.materialise()
        for slot, given in (("y", self.y), ("active", self.active)):
            if given is None:
                continue
            _, other = given.materialise()
            other = other.transpose(*rows.dims)
            differing = rows.difference(other).union(other.difference(rows))
            if differing.size:
                at = {
                    name: displayed(column[0])
                    for name, column in differing.labels().items()
                }
                raise ValueError(
                    f"{slot} of piecewise {self.name!r} and x differ at {at}; "
                    f"declare {slot} and x over the same members"
                )

    def names(self) -> dict[str, tuple[str, ...]]:
        """Return the names this declaration generates, by kind."""
        n = self.name
        if self.method == "tangent":
            return {
                "sets": (f"{n}_segment",),
                "parameters": (
                    f"{n}_slope",
                    f"{n}_intercept",
                    f"{n}_x_low",
                    f"{n}_x_high",
                ),
                "variables": (),
                "constraints": (f"{n}_tangent", f"{n}_x_min", f"{n}_x_max"),
            }
        constraints = (
            f"{n}_x",
            f"{n}_y",
            f"{n}_order_bound",
            f"{n}_fill_order",
            f"{n}_order_link",
        )
        if self.active is not None:
            constraints += (f"{n}_active",)
        return {
            "sets": (f"{n}_segment",),
            "parameters": (
                f"{n}_members",
                f"{n}_x_step",
                f"{n}_y_step",
                f"{n}_x_first",
                f"{n}_y_first",
            ),
            "variables": (f"{n}_fill", f"{n}_order"),
            "constraints": constraints,
        }

    def generated_names(self) -> frozenset[str]:
        """Return every name a model generated for this declaration."""
        return frozenset(name for kind in KINDS for name in self.generated[kind])


def _sets_of(*held: Any) -> dict[str, Any]:
    """Return the set of each dimension the expressions and coefficients read."""
    found = {}
    for item in held:
        if isinstance(item, Expression):
            for term in item.terms:
                for s in term.variable.sets:
                    found.setdefault(s.name, s)
                if term.coefficient is not None:
                    for s in term.coefficient.sets:
                        found.setdefault(s.name, s)
        elif isinstance(item, Coefficient):
            for s in item.sets:
                found.setdefault(s.name, s)
    return found


def _relation(body: Expression, sign: str, rhs: Any) -> Any:
    """Return `body` compared with `rhs` under `sign`."""
    if sign == "<=":
        return body <= rhs
    if sign == ">=":
        return body >= rhs
    return body == rhs


class _Grid:
    """The breakpoints as arrays over the entity sets and the breakpoint set.

    `entities` contains one member per entity with breakpoints. `segments`
    contains one member per pair of consecutive breakpoints, over the entity
    sets and the generated segment set. `x_step`, `y_step`, `x_start` and
    `y_start` contain one value per segment, in the order of `segments`.

    Raises ValueError for points with no breakpoint, breakpoints present at
    different coordinates, an entity with one breakpoint, an absent
    breakpoint before a present one, a value that is not finite, and
    breakpoints that are not strictly monotonic.
    """

    def __init__(
        self, declaration: Piecewise, sets: Mapping[str, Any], segment: str
    ) -> None:
        self.name = declaration.name
        b = declaration.breakpoints
        self.breakpoints = b
        self.sets = dict(sets)
        self.entity = tuple(d for d in declaration.x_points.dims if d != b)
        self.entity_sets = tuple(self.sets[d] for d in self.entity)
        order = (*self.entity, b)
        self.x = self._read(declaration.x_points, order, "x_points")
        self.y = self._read(declaration.y_points, order, "y_points")
        present = self.x.domain(order)
        if present.size == 0:
            raise ValueError(
                f"piecewise {self.name!r} has no breakpoint; give two or more "
                f"breakpoints to at least one entity"
            )
        self._check_pairing(present, self.y.domain(order))
        self._check_runs(present)
        behind_x = self.x.shift({b: 1})
        behind_y = self.y.shift({b: 1})
        self.pairs = present.intersect(behind_x.domain(order))
        self.x_step = (self.x - behind_x).restrict(self.pairs).values()
        self.y_step = (self.y - behind_y).restrict(self.pairs).values()
        self.x_start = behind_x.restrict(self.pairs).values()
        self.y_start = behind_y.restrict(self.pairs).values()
        self._check_monotonic()
        self.segment = Set(segment, self.sets[b].labels[1:])
        self.sets[segment] = self.segment
        self.points = (*self.entity, segment)
        labels = self.pairs.labels()
        columns = {d: labels[d] for d in self.entity}
        columns[segment] = labels[b]
        self.segments = subset((*self.entity_sets, self.segment), columns)
        self.entities = self.x.domain(self.entity)

    def _read(self, points: Any, order: tuple[str, ...], slot: str) -> Any:
        array = points.materialise()
        if not np.isfinite(array.values()).all():
            raise ValueError(
                f"{slot} of piecewise {self.name!r} contains a value that is "
                f"not finite; give finite breakpoints"
            )
        return array.transpose(*order)

    def _at(self, index: npt.NDArray[np.int32], column: int) -> tuple[int, ...]:
        """Return the entity positions in column `column` of `index`."""
        return tuple(int(index[axis][column]) for axis in range(len(self.entity)))

    def _reject(self, at: tuple[int, ...], condition: str, action: str) -> None:
        entity = {s.name: displayed(s.labels[k]) for s, k in zip(self.entity_sets, at)}
        raise ValueError(f"piecewise {self.name!r} {condition} at {entity}; {action}")

    def _check_pairing(self, present: Any, other: Any) -> None:
        """Raise ValueError where x_points and y_points differ in presence."""
        differing = present.difference(other).union(other.difference(present))
        if differing.size:
            self._reject(
                self._at(differing.coordinates(), 0),
                "has x_points and y_points present at different breakpoints",
                "give both a value at the same breakpoints",
            )

    def _check_runs(self, present: Any) -> None:
        """Raise ValueError for an entity with one breakpoint or with a gap.

        The breakpoints of an entity are the first members of the breakpoint
        set. The count, the least position and the greatest position check
        both conditions.
        """
        b = self.breakpoints
        counted = present.array(np.ones(present.size)).sum(b)
        count = counted.values()
        index = counted.coordinates(self.entity)
        single = np.flatnonzero(count == 1.0)
        if single.size:
            self._reject(
                self._at(index, single[0]), "has one breakpoint", "give two or more"
            )
        at_b = present.array(present.coordinates()[-1].astype(np.float64))
        gap = np.flatnonzero(
            (at_b.min(b).values() != 0.0) | (at_b.max(b).values() != count - 1.0)
        )
        if gap.size:
            self._reject(
                self._at(index, gap[0]),
                "has an absent breakpoint before a present one",
                "leave only the last breakpoints absent",
            )

    def _check_monotonic(self) -> None:
        """Raise ValueError for an entity whose x_points are not monotonic."""
        b = self.breakpoints
        step = self.pairs.array(self.x_step)
        least = step.min(b)
        greatest = step.max(b)
        unordered = np.flatnonzero(
            ~((least.values() > 0.0) | (greatest.values() < 0.0))
        )
        if unordered.size:
            self._reject(
                self._at(least.coordinates(self.entity), unordered[0]),
                "has x_points that are not strictly monotonic",
                "give strictly increasing or strictly decreasing breakpoints; "
                "nimopt does not support special ordered sets",
            )

    def slopes(self) -> npt.NDArray[np.float64]:
        """Return the slope of each segment, in the order of `segments`."""
        return self.y_step / self.x_step

    def check_curvature(self, sign: str) -> None:
        """Raise ValueError for an entity whose curvature does not match `sign`."""
        b = self.breakpoints
        slope = self.pairs.array(self.slopes())
        behind = slope.shift({b: 1})
        both = self.pairs.intersect(behind.domain(slope.dims))
        totals = self.pairs.array(self.x_step).sum(b)
        direction = self.entities.array(np.sign(totals.values()))
        change = (slope - behind).restrict(both) * direction
        if sign == ">=":
            found = change.min(b)
            wrong = np.flatnonzero(found.values() < -TOLERANCE)
            shape = "convex"
        else:
            found = change.max(b)
            wrong = np.flatnonzero(found.values() > TOLERANCE)
            shape = "concave"
        if wrong.size:
            self._reject(
                self._at(found.coordinates(self.entity), wrong[0]),
                f"has points that are not {shape}, required by sign {sign!r}",
                "use method 'incremental'",
            )

    def first(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return x and y at the first breakpoint, one value per entity."""
        at = {self.breakpoints: self.sets[self.breakpoints].labels[0]}
        return self.x.sel(at).values(), self.y.sel(at).values()

    def bounds(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the least and the greatest x, one value per entity."""
        b = self.breakpoints
        return self.x.min(b).values(), self.x.max(b).values()

    def spread(self, name: str, dims: tuple[str, ...], array: Any) -> Param | float:
        """Return a parameter over `dims` from an array over some of them.

        The array is replicated across each dimension of `dims` it does not
        have. Over no dimension, the one value is returned as a float.
        """
        if not dims:
            return float(array.values()[0])
        missing = tuple(d for d in dims if d not in array.dims)
        if missing:
            array = array.expand(missing, coords_of(self.sets[d] for d in missing))
        return Param(name, tuple(self.sets[d] for d in dims), array.transpose(*dims))

    def per_entity(
        self, name: str, dims: tuple[str, ...], values: npt.NDArray[np.float64]
    ) -> Param | float:
        """Return a parameter over `dims` from one value per entity."""
        return self.spread(name, dims, self.entities.array(values))

    def per_segment(
        self, name: str, dims: tuple[str, ...], values: npt.NDArray[np.float64]
    ) -> Param | float:
        """Return a parameter over `dims` from one value per entity and segment."""
        return self.spread(name, dims, self.segments.array(values))

    def at(self, held: Param | float, dims: tuple[str, ...]) -> Any:
        """Return `held` read at the sets of `dims`, and a float unchanged."""
        if isinstance(held, float):
            return held
        return held[tuple(self.sets[d] for d in dims)]


def _taken(model: Any, names: Mapping[str, tuple[str, ...]]) -> list[str]:
    """Return the generated names the model declares already."""
    parameters = model._parameters()
    sets, aliases = model._dimensions(parameters)
    declared = {
        *model.variables,
        *model.constraints,
        *(p.name for p in parameters),
        *(s.name for s in (*sets, *aliases)),
    }
    return [name for kind in KINDS for name in names[kind] if name in declared]


def _tangent(
    model: Any, declaration: Piecewise, grid: _Grid, names: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Declare one row per segment and the two rows that bound `x`."""
    d = declaration
    slope_name, intercept_name, low_name, high_name = names["parameters"]
    tangent_name, min_name, max_name = names["constraints"]
    slopes = grid.slopes()
    slope = grid.per_segment(slope_name, grid.points, slopes)
    body = d.y - grid.at(slope, grid.points) * d.x
    intercept = grid.per_segment(
        intercept_name, body.frame, grid.y_start - slopes * grid.x_start
    )
    low, high = grid.bounds()
    x_low = grid.per_entity(low_name, d.x.frame, low)
    x_high = grid.per_entity(high_name, d.x.frame, high)
    rows = (
        model.constraint(
            tangent_name,
            _relation(body, d.sign, grid.at(intercept, body.frame)),
        ),
        model.constraint(min_name, d.x >= grid.at(x_low, d.x.frame)),
        model.constraint(max_name, d.x <= grid.at(x_high, d.x.frame)),
    )
    return (), tuple(row.name for row in rows)


def _incremental(
    model: Any, declaration: Piecewise, grid: _Grid, names: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Declare one fill and one order column per segment, and their rows."""
    d = declaration
    members_name, x_step_name, y_step_name, x_first_name, y_first_name = names[
        "parameters"
    ]
    fill_name, order_name = names["variables"]
    x_name, y_name, bound_name, fill_order_name, link_name, *active_name = names[
        "constraints"
    ]
    dims = (*d.x.frame, grid.segment.name)
    columns = tuple(grid.sets[k] for k in dims)
    after = (*columns[:-1], grid.segment + 1)
    members = grid.per_segment(members_name, dims, np.ones(grid.segments.size))
    fill = model.var(fill_name, columns, subset=members, upper=1.0)
    order = model.var(order_name, columns, subset=members, upper=1.0, integer=True)
    x_step = grid.per_segment(x_step_name, grid.points, grid.x_step)
    y_step = grid.per_segment(y_step_name, grid.points, grid.y_step)
    x_body = d.x - Sum(grid.segment, grid.at(x_step, grid.points) * fill[columns])
    y_body = d.y - Sum(grid.segment, grid.at(y_step, grid.points) * fill[columns])
    first_x, first_y = grid.first()
    if d.active is None:
        x_first = grid.per_entity(x_first_name, d.x.frame, first_x)
        y_first = grid.per_entity(y_first_name, d.y.frame, first_y)
        x_rhs = grid.at(x_first, d.x.frame)
        y_rhs = grid.at(y_first, d.y.frame)
    else:
        x_first = grid.per_entity(x_first_name, grid.entity, first_x)
        y_first = grid.per_entity(y_first_name, grid.entity, first_y)
        x_body = x_body - grid.at(x_first, grid.entity) * d.active
        y_body = y_body - grid.at(y_first, grid.entity) * d.active
        x_rhs = y_rhs = 0.0
    rows = [
        model.constraint(x_name, x_body == x_rhs),
        model.constraint(y_name, _relation(y_body, d.sign, y_rhs)),
        model.constraint(bound_name, fill[columns] - order[columns] <= 0.0),
        model.constraint(fill_order_name, fill[after] - fill[columns] <= 0.0),
        model.constraint(link_name, order[after] - fill[columns] <= 0.0),
    ]
    if d.active is not None:
        rows.append(model.constraint(active_name[0], fill[columns] - d.active <= 0.0))
    return (fill.name, order.name), tuple(row.name for row in rows)


def generate(model: Any, declaration: Piecewise) -> None:
    """Declare the sets, parameters, variables and constraints of `declaration`.

    The name checks, the coordinate check and the breakpoint checks run
    before any declaration. Raises ValueError for a generated name the model
    declares, for a `y` or an `active` over other coordinates than `x`, and
    for a breakpoint check that fails. Sets `declaration.generated`.
    """
    names = declaration.names()
    taken = _taken(model, names)
    if taken:
        raise ValueError(
            f"piecewise {declaration.name!r} generates {taken}, already "
            f"declared in model {model.name!r}; rename the piecewise declaration"
        )
    declaration.check_domain()
    sets = _sets_of(
        declaration.x,
        declaration.y,
        declaration.active,
        declaration.x_points,
        declaration.y_points,
    )
    grid = _Grid(declaration, sets, names["sets"][0])
    if declaration.method == "tangent":
        grid.check_curvature(declaration.sign)
        variables, constraints = _tangent(model, declaration, grid, names)
    else:
        variables, constraints = _incremental(model, declaration, grid, names)
    declaration.generated = {
        "sets": names["sets"],
        "parameters": names["parameters"],
        "variables": variables,
        "constraints": constraints,
    }
