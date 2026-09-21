"""A model as a file: its structure in YAML, and its data inline or beside it."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from nimopt.definition import Definition
from nimopt.model import Model
from nimopt.names import check_addressable, check_one_kind
from nimopt.param import Param
from nimopt.sets import as_members, member_text, named_condition
from nimopt.syntax import read, render
from nimopt.term import Relation

VERSION = 4
VERSIONS = (2, 3, 4)
WRITTEN = (3, 4)
KEYS = (
    "version",
    "name",
    "sense",
    "sets",
    "aliases",
    "parameters",
    "variables",
    "constraints",
    "piecewise",
    "objective",
    "data",
)
VARIABLE_KEYS = ("sets", "subset", "lower", "upper", "integer")
CONSTRAINT_KEYS = ("relation", "where", "over")
PIECEWISE_KEYS = (
    "x",
    "x_points",
    "y",
    "y_points",
    "sign",
    "method",
    "active",
    "relaxed",
    "where",
)
PIECEWISE_REQUIRED = PIECEWISE_KEYS[:-3]
SET_KEYS = ("dtype", "members")
_HEAD_3 = """\
# --- Reading this file --------------------------------------------------
# A nimopt model file, format version 3. The keys are written in this
# order, and no other key is accepted: version, name, sense, sets,
# aliases, parameters, variables, constraints, objective, data. Only
# version, name and sense are required.
"""
_HEAD_4 = """\
# --- Reading this file --------------------------------------------------
# A nimopt model file, format version 4. The keys are written in this
# order, and no other key is accepted: version, name, sense, sets,
# aliases, parameters, variables, constraints, piecewise, objective,
# data. Only version, name and sense are required.
"""
_BODY = """\
#
# sense       'min' or 'max', applied to the objective.
# sets        The names of the index sets. Their members are listed
#             under data.
# aliases     name: set. A second name for a set, with the same
#             members, so that a parameter or a variable can be
#             indexed by one set twice.
# parameters  name: [set, ...]. The index sets of a data array. Its
#             values are listed under data.
# objective   An expression over the variables, with no comparison
#             operator. A constant term in the objective is a fixed
#             cost: it is reported with the solver's objective value
#             and is not passed to the solver.
# data        The file name of an .npz archive in the same directory
#             as this file, or an inline mapping. It contains one entry
#             per set (its members) and one entry per parameter (its
#             values), in one of two forms: a dense array over the
#             full product of the parameter's index sets, or a table
#             with one column per index set followed by 'value'. A
#             table contains only the entries it lists. A coordinate
#             that is not listed is not a zero; it is a coefficient
#             that does not exist. The constraint row rules below
#             depend on that difference.
#             A set of datetime64 or timedelta64 members is written as
#             a mapping of 'dtype' and 'members' instead of a list:
#               T:
#                 dtype: datetime64[s]
#                 members: ['2030-01-01T00:00:00', '2030-01-01T01:00:00']
#             A datetime64 member is its ISO 8601 string and a
#             timedelta64 member is its integer count of the unit in
#             the dtype. A label column of a table is written in the
#             same text.
#
# variables   name: {sets, subset, lower, upper, integer}
#   sets      The index sets of the variable. Required.
#   subset    A parameter name, or a list of set names. The variable
#             exists only at those coordinates. Default: the full
#             product of its index sets.
#   lower     A number or a parameter name. Default: 0.
#   upper     A number or a parameter name. Default: .inf.
#   integer   true for an integer variable. Default: continuous.
#
# constraints name: {relation, where, over}
#   relation  Required. The whole constraint: an expression, one of
#             <= >= ==, and a right-hand side. The right-hand side is
#             a number or a parameter indexed by the constraint's free
#             index sets. Any term that contains a variable has been
#             moved to the left-hand side.
#   over      The rows, declared explicitly: a parameter name or a list
#             of set names. Every one of those rows exists. A term
#             that has entries at only some of those rows contributes
#             to the rows where it has entries. The right-hand side
#             must have a value at every row; otherwise building the
#             model raises an error.
#   where     The derived rows, restricted to this domain: a parameter
#             name or a list of set names.
#   Giving both over and where is an error. With neither, the rows
#   are derived: the coordinates at which every term has an entry,
#   restricted to the coordinates at which the right-hand side has a
#   value. A term whose parameter has no entry at a coordinate removes
#   that row.
#
"""
_PIECEWISE = """\
# piecewise   name: {x, x_points, y, y_points, sign, method, active,
#                    relaxed, where}
#             A piecewise-linear relation of y to x. Loading the file
#             generates its variables and constraints; they are not
#             written.
#   x         Required. An expression.
#   x_points  Required. A parameter indexed by some or all of the
#             index sets of x and by one breakpoint set. An absent
#             entry is a breakpoint that does not exist; only the last
#             breakpoints of an entity may be absent.
#   y         Required. An expression over the index sets of x.
#   y_points  Required. A parameter over the index sets of x_points,
#             with entries at the same coordinates.
#   sign      Required. <=, >= or ==: y compared with the curve. x is
#             always on the curve.
#   method    Required. incremental: one continuous and one integer
#             variable per segment. tangent: one row per segment; the
#             points are convex under >= and concave under <=. auto:
#             tangent where it applies and no active is given, else
#             incremental.
#   active    A binary variable over the index sets of x, or a sum of
#             them, incremental only. Where it is 0, x is 0 and y is
#             compared with 0.
#   relaxed   true accepts an active between 0 and 1, which scales the
#             curve by its value. Absent is false.
#   where     The entities the declaration covers: a parameter name or
#             a list of set names, over the sets of x_points other than
#             the breakpoint set. Absent is every entity.
#   The generated names are the piecewise name, an underscore and one
#   of: segment, members, x_step, y_step, x_first, y_first, fill,
#   order, x, y, order_bound, fill_order, order_link, active, slope,
#   intercept, x_low, x_high, tangent, x_min, x_max.
#
"""
_SYNTAX = """\
#   Name[s, ...]           a parameter or a variable indexed by its
#                          sets; a bare Name where it has no sets
#   Name['label']          that index fixed at one member; a
#                          datetime64 member is its quoted ISO 8601
#                          string and a timedelta64 member is a quoted
#                          count and numpy unit code such as '3 h'
#   s - k      s + k       that index shifted by k positions; the term
#                          has no entry where the shift leaves the set
#   s.cyclic - k           shifted, wrapping around at the ends
#   Sum(s, ..., body)      body summed over the named sets; the sets
#                          that remain are the constraint's free sets
#   Sum(s, body, where=D)  the same, restricted to the coordinates
#                          of D: a parameter, or a tuple of sets
#                          such as (s, t)
#   + - * / **, unary minus, and parentheses.
#   One comparison operator per relation: write each side of a range
#   as a separate constraint.
# ------------------------------------------------------------------------
"""
INSTRUCTIONS_V3 = (
    _HEAD_3 + _BODY + "# Expression syntax, in relation and objective:\n" + _SYNTAX
)
INSTRUCTIONS = (
    _HEAD_4
    + _BODY
    + _PIECEWISE
    + "# Expression syntax, in relation, objective and piecewise:\n"
    + _SYNTAX
)


class _Dumper(yaml.SafeDumper):
    """Block mappings, and a list of scalars on one line."""


def _sequence(dumper: Any, data: Any) -> Any:
    flow = all(isinstance(item, (str, int, float, bool)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _sequence)


def instructions_for(version: Any) -> str:
    """Return the comment block that explains a file of `version`."""
    return INSTRUCTIONS if version == 4 else INSTRUCTIONS_V3


def yaml_text(mapping: Mapping[str, Any], instructions: bool = False) -> str:
    """Return `mapping` as YAML text, its keys in the order given.

    `instructions=True` prefixes the comment block that explains the format
    of the mapping's version. The block is a YAML comment, and the loader
    ignores it.
    """
    text = yaml.dump(
        mapping,
        Dumper=_Dumper,
        sort_keys=False,
        width=float("inf"),
        default_flow_style=False,
    )
    if not instructions:
        return text
    return instructions_for(mapping.get("version", VERSION)) + text


def written_version(version: Any) -> int:
    """Return `version` as a version a file is written in.

    Raises ValueError for a version other than 3 and 4.
    """
    if isinstance(version, bool) or version not in WRITTEN:
        raise ValueError(f"a file is written in version 3 or 4; got {version!r}")
    return int(version)


def _domain(held: Any, owner: str, slot: str) -> Any:
    if held is None:
        return None
    name = named_condition(held, owner, slot)
    if isinstance(name, str):
        return name
    return list(name)


def _bound(bound: Any, default: float) -> str | float | None:
    if isinstance(bound, Param):
        return bound.name
    return None if float(bound) == default else float(bound)


def _parts(held: Any, version: int = VERSION) -> tuple[Any, ...]:
    """Return the sets, aliases, parameters, variables, constraints, objective.

    Version 4 omits what a model's piecewise declarations generated, and
    reads the parameters of the declarations. Version 3 contains every
    declaration of a model, and raises ValueError for a definition with a
    piecewise declaration.
    """
    if isinstance(held, Definition):
        if version == 3 and held.piecewise_declarations:
            first = next(iter(held.piecewise_declarations))
            raise ValueError(
                f"definition {held.name!r} contains piecewise {first!r}; "
                f"write version 4 or build the model first"
            )
        constraints = [
            (name, relation, where, over)
            for name, (relation, where, over) in held.constraints.items()
        ]
        return (
            tuple(held.sets.values()),
            tuple(held.aliases.values()),
            tuple(held.parameters.values()),
            tuple(held.variables.values()),
            constraints,
            held.objective,
        )
    if isinstance(held, Model):
        declared = version == 4
        skip = held._generated() if declared else frozenset()
        parameters = held._parameters(as_declared=declared)
        constraints = [
            (name, c.relation, c.where, c.over)
            for name, c in held.constraints.items()
            if name not in skip
        ]
        sets, aliases = held._dimensions(parameters, as_declared=declared)
        return (
            sets,
            aliases,
            parameters,
            tuple(v for name, v in held.variables.items() if name not in skip),
            constraints,
            held.objective,
        )
    raise TypeError(
        f"a file contains a definition or a model; got {type(held).__name__}"
    )


def _long(
    parameter: Any,
) -> tuple[dict[str, npt.NDArray[Any]], npt.NDArray[np.float64]]:
    """Return a parameter's entries as label columns and values."""
    array = parameter.materialise()
    positions = array.coordinates()
    labels = {
        dim: held.coord.to_index(positions[j])
        for j, (dim, held) in enumerate(zip(parameter.dims, parameter.sets))
    }
    return labels, array.values()


def _arrays(model: Any, version: int = VERSION) -> dict[str, npt.NDArray[Any]]:
    """Return every set's members and every parameter's array, keyed by name.

    The sets and parameters are those a file of `version` declares. A
    parameter over its full product is a dense grid. A parameter over less
    is a structured table of one field per dimension and a `value` field.
    """
    sets, _, parameters, *_ = _parts(model, version)
    out = {s.name: np.asarray(s.labels) for s in sets}
    for parameter in parameters:
        array = parameter.materialise()
        if array.domain(parameter.dims).is_full:
            out[parameter.name] = array.to_dense()
            continue
        labels, values = _long(parameter)
        dtype = [(dim, labels[dim].dtype) for dim in parameter.dims]
        dtype.append(("value", np.dtype(np.float64)))
        table = np.empty(values.size, dtype=dtype)
        for dim in parameter.dims:
            table[dim] = labels[dim]
        table["value"] = values
        out[parameter.name] = table
    for name, array in out.items():
        if array.dtype.hasobject:
            raise ValueError(
                f"{name!r} is an object array, and a file stores no object "
                f"array without pickling; give its labels one dtype"
            )
    return out


def _piecewise_entry(declaration: Any) -> dict[str, Any]:
    """Return the entry of one piecewise declaration."""
    entry = {
        "x": render(declaration.x),
        "x_points": render(declaration.x_points),
        "y": render(declaration.y),
        "y_points": render(declaration.y_points),
        "sign": declaration.sign,
        "method": declaration.method,
    }
    if declaration.active is not None:
        entry["active"] = render(declaration.active)
    if declaration.relaxed:
        entry["relaxed"] = True
    if declaration.where is not None:
        entry["where"] = _domain(
            declaration.where, f"piecewise {declaration.name!r}", "where="
        )
    return entry


def structure(held: Any, version: int = VERSION) -> dict[str, Any]:
    """Return the mapping of a definition's or a model's file, without data.

    Raises ValueError for a version other than 3 and 4.
    """
    version = written_version(version)
    parts = _parts(held, version)
    sets, aliases, parameters, variables, constraints, objective = parts
    symbols = (
        ("set", {s.name for s in sets}),
        ("alias", {a.name for a in aliases}),
        ("parameter", {p.name for p in parameters}),
        ("variable", {v.name for v in variables}),
    )
    for at, (kind, names) in enumerate(symbols):
        for name in sorted(names):
            check_addressable(name, kind)
            check_one_kind(name, kind, symbols[:at])
    out = {
        "version": version,
        "name": held.name,
        "sense": held.sense,
        "sets": [s.name for s in sets],
    }
    if aliases:
        out["aliases"] = {a.name: a.base.name for a in aliases}
    out["parameters"] = {p.name: list(p.dims) for p in parameters}
    out["variables"] = {}
    out["constraints"] = {}
    for v in variables:
        entry = {"sets": list(v.dims)}
        for slot, value in (
            ("subset", _domain(v.subset, f"variable {v.name!r}", "subset=")),
            ("lower", _bound(v.lower, 0.0)),
            ("upper", _bound(v.upper, np.inf)),
        ):
            if value is not None:
                entry[slot] = value
        if v.integer:
            entry["integer"] = True
        out["variables"][v.name] = entry
    for name, relation, where, over in constraints:
        entry = {"relation": render(relation)}
        for slot, value in (
            ("where", _domain(where, f"constraint {name!r}", "where=")),
            ("over", _domain(over, f"constraint {name!r}", "over=")),
        ):
            if value is not None:
                entry[slot] = value
        out["constraints"][name] = entry
    declarations = held.piecewise_declarations if version == 4 else {}
    if declarations:
        out["piecewise"] = {
            name: _piecewise_entry(d) for name, d in declarations.items()
        }
    if objective is not None:
        out["objective"] = render(objective)
    return out


def _names(names: Iterable[str]) -> str:
    """Return the names quoted and separated by commas."""
    return ", ".join(repr(name) for name in names)


def _only(entry: Mapping[str, Any], keys: Iterable[str], what: str) -> None:
    unknown = sorted(set(entry) - set(keys))
    if unknown:
        word = "key" if len(unknown) == 1 else "keys"
        raise ValueError(
            f"{what} contains the unknown {word} {_names(unknown)}; "
            f"write only {_names(keys)}"
        )


def _named_set(definition: Any, name: str, what: str) -> Any:
    if name not in definition.sets:
        raise ValueError(
            f"{what} refers to the undeclared set {name!r}; declare it under sets"
        )
    return definition.sets[name]


def _named_dimension(definition: Any, name: str, what: str) -> Any:
    """Return the set or the alias called `name`."""
    if name in definition.aliases:
        return definition.aliases[name]
    return _named_set(definition, name, what)


def _named_parameter(definition: Any, name: str, what: str) -> Any:
    if name not in definition.parameters:
        raise ValueError(
            f"{what} refers to the undeclared parameter {name!r}; declare it "
            f"under parameters"
        )
    return definition.parameters[name]


def _read_domain(definition: Any, given: Any, what: str) -> Any:
    if given is None:
        return None
    if isinstance(given, str):
        return _named_parameter(definition, given, what)
    if isinstance(given, list):
        return tuple(_named_dimension(definition, name, what) for name in given)
    raise ValueError(
        f"{what} is a parameter's name or a list of set names; got {given!r}"
    )


def _read_bound(definition: Any, given: Any, what: str) -> Any:
    if isinstance(given, str):
        return _named_parameter(definition, given, what)
    if isinstance(given, bool) or not isinstance(given, (int, float)):
        raise ValueError(f"{what} is a number or a parameter's name; got {given!r}")
    return float(given)


def _definition(spec: Mapping[str, Any]) -> Definition:
    """Return the definition a file's structure section declares."""
    for key in ("name", "sense"):
        if key not in spec:
            raise ValueError(f"the file declares no {key!r}; add it")
    d = Definition(spec["name"], sense=spec["sense"])
    for name in spec.get("sets") or []:
        d.set(name)
    for name, base in (spec.get("aliases") or {}).items():
        d.alias(name, _named_set(d, base, f"alias {name!r}"))
    for name, dims in (spec.get("parameters") or {}).items():
        what = f"parameter {name!r}"
        d.param(name, tuple(_named_dimension(d, dim, what) for dim in dims))
    for name, entry in (spec.get("variables") or {}).items():
        what = f"variable {name!r}"
        _only(entry, VARIABLE_KEYS, what)
        d.var(
            name,
            tuple(_named_dimension(d, dim, what) for dim in entry["sets"]),
            subset=_read_domain(d, entry.get("subset"), f"{what} subset"),
            lower=_read_bound(d, entry.get("lower", 0.0), f"{what} lower"),
            upper=_read_bound(d, entry.get("upper", np.inf), f"{what} upper"),
            integer=bool(entry.get("integer", False)),
        )
    symbols = {**d.sets, **d.aliases, **d.parameters, **d.variables}
    for name, entry in (spec.get("constraints") or {}).items():
        what = f"constraint {name!r}"
        _only(entry, CONSTRAINT_KEYS, what)
        relation = read(entry["relation"], symbols)
        if not isinstance(relation, Relation):
            raise ValueError(
                f"{what} reads to no comparison: {entry['relation']!r} is an "
                f"expression with no sense; write one of <=, >= or =="
            )
        d.constraint(
            name,
            relation,
            where=_read_domain(d, entry.get("where"), f"{what} where"),
            over=_read_domain(d, entry.get("over"), f"{what} over"),
        )
    for name, entry in (spec.get("piecewise") or {}).items():
        what = f"piecewise {name!r}"
        _only(entry, PIECEWISE_KEYS, what)
        missing = [key for key in PIECEWISE_REQUIRED if key not in entry]
        if missing:
            raise ValueError(
                f"{what} declares no {_names(missing)}; write "
                f"{_names(PIECEWISE_REQUIRED)}"
            )
        active = entry.get("active")
        d.piecewise(
            name,
            x=read(entry["x"], symbols),
            x_points=read(entry["x_points"], symbols),
            y=read(entry["y"], symbols),
            y_points=read(entry["y_points"], symbols),
            sign=entry["sign"],
            method=entry["method"],
            active=None if active is None else read(active, symbols),
            relaxed=bool(entry.get("relaxed", False)),
            where=_read_domain(d, entry.get("where"), f"{what} where"),
        )
    if "objective" in spec:
        objective = read(spec["objective"], symbols)
        if isinstance(objective, Relation):
            raise ValueError(
                f"the objective reads to a comparison: {spec['objective']!r}; "
                f"write an expression with no comparison operator"
            )
        d.set_objective(objective)
    return d


def _spec(text: str) -> dict[str, Any]:
    spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise ValueError(f"a model file is a YAML mapping; got {type(spec).__name__}")
    version = spec.get("version")
    if version not in VERSIONS:
        accepted = " or ".join(str(v) for v in VERSIONS)
        raise ValueError(
            f"the file declares version {version!r}; pass a file of version {accepted}"
        )
    _only(spec, KEYS, "a model file")
    if "piecewise" in spec and version < 4:
        raise ValueError(
            f"the file declares version {version} and contains piecewise; "
            f"declare version 4"
        )
    return spec


def _written(value: Any) -> Any:
    """Return one label as the value its inline entry contains.

    A datetime64 label returns its ISO 8601 string. A timedelta64 label
    returns its integer count. Every other label returns its Python value.
    """
    if value.dtype.kind == "M":
        return member_text(value)
    if value.dtype.kind == "m":
        return int(value.astype(np.int64))
    return value.item()


def to_inline(model: Any, version: int = VERSION) -> dict[str, Any]:
    """Return a model's data as the block its file of `version` contains."""
    out = {}
    for name, array in _arrays(model, version).items():
        if array.dtype.names is not None:
            columns = list(array.dtype.names)
            out[name] = {
                "columns": columns,
                "rows": [
                    [_written(array[c][k]) for c in columns] for k in range(array.size)
                ],
            }
        elif array.dtype.kind in "Mm":
            out[name] = {
                "dtype": str(array.dtype),
                "members": [_written(value) for value in array],
            }
        else:
            out[name] = array.tolist()
    return out


def _columns(name: str, dims: Sequence[str], columns: Sequence[str]) -> None:
    """Raise where a long table's columns are not the dimensions then `value`."""
    expected = list(dims) + ["value"]
    if list(columns) != expected:
        raise ValueError(
            f"parameter {name!r} is given columns {list(columns)}; a table "
            f"lists the dimensions then value: {expected}"
        )


def _table(
    name: str, dims: Sequence[str], columns: Sequence[str], rows: Sequence[Any]
) -> tuple[dict[str, npt.NDArray[Any]], npt.NDArray[np.float64]]:
    """Return a long table as the pair `build` takes, with its columns checked."""
    _columns(name, dims, columns)
    expected = list(dims) + ["value"]
    held = list(zip(*rows)) if rows else [[] for _ in expected]
    labels = {dim: np.asarray(held[j]) for j, dim in enumerate(dims)}
    return labels, np.asarray(held[-1], dtype=np.float64)


def _members(name: str, entry: Mapping[str, Any]) -> npt.NDArray[Any]:
    """Return a set's members from the mapping form of an inline entry.

    The mapping declares `dtype` and `members`. Raises ValueError for a
    missing key, for a dtype that is not a datetime64 or a timedelta64 dtype,
    and for a member that does not convert exactly to that dtype.
    """
    what = f"set {name!r}"
    _only(entry, SET_KEYS, what)
    missing = " and ".join(repr(key) for key in SET_KEYS if key not in entry)
    if missing:
        raise ValueError(f"{what} declares no {missing}; write dtype and members")
    given = entry["dtype"]
    message = (
        f"{what} declares the dtype {given!r}; write a datetime64 or a "
        f"timedelta64 dtype, or write the members as a list"
    )
    try:
        dtype = np.dtype(given)
    except TypeError:
        raise ValueError(message) from None
    if dtype.kind not in "Mm":
        raise ValueError(message)
    return as_members(entry["members"], dtype, what)


def from_inline(block: Mapping[str, Any], definition: Definition) -> dict[str, Any]:
    """Return the mapping `build` takes, from a file's inline block."""
    out = {}
    for name, value in block.items():
        parameter = definition.parameters.get(name)
        if parameter is not None and isinstance(value, dict):
            _only(value, ("columns", "rows"), f"parameter {name!r}")
            out[name] = _table(name, parameter.dims, value["columns"], value["rows"])
        elif parameter is not None:
            out[name] = np.asarray(value, dtype=np.float64)
        elif isinstance(value, dict):
            out[name] = _members(name, value)
        else:
            out[name] = np.asarray(value)
    return out


def write_npz(arrays: Mapping[str, Any], path: Any) -> None:
    """Write the arrays `_arrays` gathered as an `.npz` at `path`."""
    np.savez_compressed(path, **arrays)


def read_npz(path: Any, definition: Definition) -> dict[str, Any]:
    """Return the mapping `build` takes, from an `.npz` beside a file."""
    out = {}
    with np.load(path, allow_pickle=False) as held:
        for name in held.files:
            array = held[name]
            if array.dtype.names is None:
                out[name] = array
                continue
            parameter = definition.parameters.get(name)
            dims = () if parameter is None else parameter.dims
            _columns(name, dims, list(array.dtype.names))
            labels = {dim: array[dim] for dim in dims}
            values = np.asarray(array["value"], dtype=np.float64)
            out[name] = (labels, values)
    return out


def _loads(text: str, data: Any, directory: Any) -> Any:
    spec = _spec(text)
    definition = _definition(spec)
    inside = spec.get("data")
    if inside is not None and data is not None:
        raise ValueError(
            "the file contains data and data= is given; pass one of the two sources"
        )
    if isinstance(inside, str):
        if directory is None:
            raise ValueError(
                f"the text refers to {inside!r} as its data, and text has no "
                f"directory to find it in; read the file with load(path)"
            )
        if Path(inside).name != inside:
            raise ValueError(
                f"a data file is written beside the model file with no "
                f"directory; got {inside!r}"
            )
        data = directory / inside
    elif isinstance(inside, dict):
        return definition.build(from_inline(inside, definition))
    elif inside is not None:
        raise ValueError(
            f"data is an inline mapping or the name of an .npz beside the "
            f"file; got {type(inside).__name__}"
        )
    if data is None:
        return definition
    if isinstance(data, (str, Path)):
        return definition.build(read_npz(Path(data), definition))
    return definition.build(data)


def loads(text: str, data: Any = None) -> Any:
    """Return the definition `text` declares, or the model it builds.

    `data` is a mapping `build` takes or the path of an `.npz`. Text that
    refers to a sidecar raises. Text has no directory to read the sidecar
    from.
    """
    return _loads(text, data, None)


def load(path: Any, data: Any = None) -> Any:
    """Return the definition the file at `path` declares, or its model."""
    path = Path(path)
    return _loads(path.read_text(), data, path.parent)


def dumps(
    what: Any, inline: bool = False, instructions: bool = False, version: int = VERSION
) -> str:
    """Return the text of a definition's file, or of a model's file.

    A model's data is included with `inline=True` alone. The text contains no
    sidecar line. `instructions=True` prefixes the comment block that
    explains the format. `version` is 4 or 3. Version 3 writes the
    declarations a piecewise declaration generated in its place, and raises
    ValueError for a definition that contains one. Raises ValueError for a
    definition with `inline=True`. Raises TypeError for a value that is not a
    definition or a model.
    """
    if isinstance(what, Definition):
        if inline:
            raise ValueError(
                "a definition contains no data to inline; pass a model, or "
                "pass inline=False"
            )
        return yaml_text(structure(what, version), instructions)
    if not isinstance(what, Model):
        raise TypeError(
            f"a file contains a definition or a model; got {type(what).__name__}"
        )
    mapping = structure(what, version)
    if inline:
        mapping["data"] = to_inline(what, version)
    return yaml_text(mapping, instructions)


def save(
    what: Any,
    path: Any,
    inline: bool = False,
    instructions: bool = False,
    version: int = VERSION,
) -> None:
    """Write `what` to `path`: a definition's file, or a model's with its data.

    A model's data goes to an `.npz` beside the file under the file's stem, or
    into the file itself with `inline=True`. `instructions=True` writes the
    comment block that explains the format at the top of the file. `version`
    is 4 or 3. Version 3 writes the declarations a piecewise declaration
    generated in its place, and raises ValueError for a definition that
    contains one.
    """
    path = Path(path)
    if isinstance(what, Model) and not inline:
        mapping = structure(what, version)
        arrays = _arrays(what, version)
        sidecar = path.with_suffix(".npz")
        mapping["data"] = sidecar.name
        write_npz(arrays, sidecar)
        path.write_text(yaml_text(mapping, instructions))
        return
    path.write_text(dumps(what, inline, instructions, version))
