"""The declared bounds and the pins of the variables a session changes."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt

from nimopt.param import Param
from nimopt.term import Expression
from nimopt.variable import Variable

_NOT_A_TARGET = "a target is a variable or one reference to one variable"

_Layer = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
_Pins = npt.NDArray[np.float64]


def target_of(
    variables: Mapping[str, Variable], target: Any
) -> tuple[Variable, npt.NDArray[np.int64], bool]:
    """Return the variable of a target, its positions and whether it is whole."""
    if isinstance(target, Variable):
        variable = target
        fixed: Mapping[str, Any] = {}
    else:
        term = _one_plain_term(target)
        variable = term.variable
        fixed = term.fixed
    held = variables.get(variable.name)
    if held is None:
        raise ValueError(
            f"the model has no variable {variable.name!r}; "
            f"use one of {tuple(variables)}"
        )
    return held, held.columns_of(fixed), not fixed


def _one_plain_term(target: Any) -> Any:
    """Return the one plain term of a target, or raise TypeError."""
    if not isinstance(target, Expression):
        raise TypeError(f"{_NOT_A_TARGET}; got {type(target).__name__}")
    if len(target.terms) != 1 or target.constant != 0.0:
        raise TypeError(f"{_NOT_A_TARGET}; got {target!r}")
    term = target.terms[0]
    plain = (
        term.coefficient is None
        and not term.summed
        and not term.shifts
        and term.where is None
        and term.scale == 1.0
    )
    if not plain:
        raise TypeError(f"{_NOT_A_TARGET}; got {target!r}")
    return term


def _columns_at(variable: Variable) -> slice:
    """Return the columns of `variable` in the vectors given to the constructor."""
    n = variable.n_columns
    start = variable.start
    if start is None:
        raise ValueError(
            f"variable {variable.name!r} has no columns in the given vectors; "
            f"declare it with Model.var"
        )
    return slice(start, start + n)


def _pin_values(
    variable: Variable, values: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return `values` after the finite and the integral checks of a pin."""
    if not np.isfinite(values).all():
        raise ValueError(
            f"the pin of variable {variable.name!r} is not finite; give a finite value"
        )
    if variable.integer:
        not_integral = values != np.floor(values)
        if not_integral.any():
            first = float(values[not_integral][0])
            raise ValueError(
                f"the pin {first!r} of variable {variable.name!r} is not "
                f"integral; a column of an integer variable takes an integer"
            )
    return values


def _bound_side(
    variable: Variable, side: float | Param, whole: bool, at: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Return the positions and the values of one side of `set_bound`."""
    if isinstance(side, Param):
        if not whole:
            raise ValueError(
                f"the bound of variable {variable.name!r} is a parameter over "
                f"a slice of it; bound the whole variable, or bound this "
                f"slice at a number"
            )
        positions, values = variable.align(side)
        if positions.size != variable.n_columns:
            raise ValueError(
                f"the bound {side.name!r} has a value at {positions.size} of "
                f"the {variable.n_columns} columns of variable "
                f"{variable.name!r}; give the bound a value at every member"
            )
        return np.asarray(positions, dtype=np.int64), np.asarray(
            values, dtype=np.float64
        )
    return at, np.full(at.size, float(side), dtype=np.float64)


def _reject_not_a_number(variable: Variable, side: float | Param, which: str) -> None:
    """Raise ValueError for a bound that is NaN."""
    named = "" if not isinstance(side, Param) else f" {side.name!r}"
    raise ValueError(
        f"the {which} bound{named} of variable {variable.name!r} is not a "
        f"number; give a finite value or an infinity"
    )


class Bounds:
    """The declared bounds and the pins of the variables a session changes.

    A variable with no entry costs nothing. The two vectors given to the
    constructor are read and not written.
    """

    def __init__(
        self,
        lower: npt.NDArray[np.float64],
        upper: npt.NDArray[np.float64],
    ) -> None:
        self._lower = lower
        self._upper = upper
        self._declared: dict[str, _Layer] = {}
        self._pins: dict[str, npt.NDArray[np.float64]] = {}

    def bounds_of(
        self, variable: Variable
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the final lower and upper bound of each column of `variable`."""
        held = self._declared.get(variable.name)
        if held is None:
            at = _columns_at(variable)
            lower = self._lower[at].copy()
            upper = self._upper[at].copy()
        else:
            lower = held[0].copy()
            upper = held[1].copy()
        pins = self._pins.get(variable.name)
        if pins is not None:
            pinned = np.flatnonzero(~np.isnan(pins))
            lower[pinned] = pins[pinned]
            upper[pinned] = pins[pinned]
        return lower, upper

    def fix(
        self,
        variable: Variable,
        positions: npt.ArrayLike,
        value: float | Param,
        whole: bool,
    ) -> None:
        """Pin the columns of `variable` at `positions` to `value`."""
        if isinstance(value, Param):
            if not whole:
                raise ValueError(
                    f"the pin of variable {variable.name!r} is a parameter "
                    f"over a slice of it; pin the whole variable, or pin this "
                    f"slice at a number"
                )
            at, given = variable.align(value)
            pins_at = np.asarray(at, dtype=np.int64)
            values = _pin_values(variable, np.asarray(given, dtype=np.float64))
        else:
            pins_at = np.asarray(positions, dtype=np.int64)
            number = float(_pin_values(variable, np.asarray([value]))[0])
            values = np.full(pins_at.size, number, dtype=np.float64)
        if pins_at.size == 0:
            return
        self._entry(variable)
        pins = self._pins.get(variable.name)
        if pins is None:
            pins = np.full(variable.n_columns, np.nan, dtype=np.float64)
            self._pins[variable.name] = pins
        pins[pins_at] = values

    def unfix(self, variable: Variable, positions: npt.ArrayLike) -> None:
        """Remove the pin of the columns of `variable` at `positions`."""
        pins = self._pins.get(variable.name)
        if pins is None:
            raise ValueError(
                f"variable {variable.name!r} has no fixed column; "
                f"fix a column before unfixing it"
            )
        pins[np.asarray(positions, dtype=np.int64)] = np.nan
        if np.isnan(pins).all():
            del self._pins[variable.name]

    def set_bound(
        self,
        variable: Variable,
        positions: npt.ArrayLike,
        lower: float | Param | None,
        upper: float | Param | None,
        whole: bool,
    ) -> None:
        """Replace the declared bound of `variable` at `positions`."""
        if lower is None and upper is None:
            raise ValueError(
                f"no bound is given for variable {variable.name!r}; "
                f"give a lower bound, an upper bound, or both"
            )
        at = np.asarray(positions, dtype=np.int64)
        writes = []
        for which, side in (("lower", lower), ("upper", upper)):
            if side is None:
                continue
            at_side, values = _bound_side(variable, side, whole, at)
            if np.isnan(values).any():
                _reject_not_a_number(variable, side, which)
            writes.append((0 if which == "lower" else 1, at_side, values))
        declared = self._entry(variable)
        for which, at_side, values in writes:
            declared[which][at_side] = values

    @contextmanager
    def changing(self, variable: Variable) -> Iterator[None]:
        """Restore the entry of `variable` where the body raises."""
        saved = self._saved(variable)
        try:
            yield
        except BaseException:
            self._restore(variable, saved)
            raise

    def _saved(self, variable: Variable) -> tuple[_Layer | None, _Pins | None]:
        """Return copies of the two layers of `variable`, None where absent."""
        declared = self._declared.get(variable.name)
        pins = self._pins.get(variable.name)
        return (
            None if declared is None else (declared[0].copy(), declared[1].copy()),
            None if pins is None else pins.copy(),
        )

    def _restore(
        self, variable: Variable, saved: tuple[_Layer | None, _Pins | None]
    ) -> None:
        """Write back the two layers of `variable` that `_saved` returned."""
        declared, pins = saved
        if declared is None:
            self._declared.pop(variable.name, None)
        else:
            self._declared[variable.name] = (declared[0].copy(), declared[1].copy())
        if pins is None:
            self._pins.pop(variable.name, None)
        else:
            self._pins[variable.name] = pins.copy()

    def _entry(self, variable: Variable) -> _Layer:
        """Return the declared layer of `variable`, copied at the first change."""
        held = self._declared.get(variable.name)
        if held is None:
            at = _columns_at(variable)
            held = (self._lower[at].copy(), self._upper[at].copy())
            self._declared[variable.name] = held
        return held
