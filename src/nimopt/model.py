"""A model: its variables, its constraints and the matrix they assemble into."""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from nimblend import EntryBuffer, ProductCoord, SparseArray

from nimopt.coefficient import Coefficient
from nimopt.constraint import Constraint
from nimopt.explanation import (
    ConstraintShape,
    Explanation,
    objective_constant,
    objective_text,
    param_shape,
    set_shape,
    variable_shape,
)
from nimopt.names import COLUMN, ROW
from nimopt.param import Param
from nimopt.progress import reporter
from nimopt.sets import Alias
from nimopt.symbol import read_at_its_sets
from nimopt.syntax import render
from nimopt.term import Expression
from nimopt.variable import Variable

if TYPE_CHECKING:
    from nimopt.absence import Absence
    from nimopt.row import Row
    from nimopt.session import Session
    from nimopt.solution import Solution


class _Written:
    """The nonzeros a pass has written, reported to a progress reporter.

    A term's own count is known only once its constraint finishes. A term
    advances the report by nothing, and a constraint advances it by its own
    coefficients. Reporting the term keeps the line moving while a large
    constraint is built.
    """

    def __init__(self, held: Any, total: int, what: str) -> None:
        self.held = held
        self.done = 0
        self.what = ""
        if held is not None:
            held.start(total, what)

    def term(self, term: Any) -> None:
        """Report that a term of the constraint being written is built."""
        if self.held is not None:
            self.what = term.variable.name
            self.held.step(self.done, self.what)

    def constraint(self, name: str, nnz: int) -> None:
        """Report that a constraint of `nnz` coefficients is written."""
        self.done += nnz
        if self.held is not None:
            self.held.step(self.done, name)

    def close(self) -> None:
        """Report that the pass is finished."""
        if self.held is not None:
            self.held.done()


class Model:
    """Variables numbered into one column space, constraints into one matrix.

    A variable's columns are a contiguous range of that space, computed from
    its multi-index. Declaring a variable allocates its bounds and nothing per
    column.
    """

    def __init__(self, name: str = "model", sense: str = "min") -> None:
        if sense not in ("min", "max"):
            raise ValueError(f"sense is 'min' or 'max'; got {sense!r}")
        self.name = str(name)
        self._sense = sense
        self.variables = {}
        self.constraints = {}
        self._objective = None
        self._n_columns = 0
        self._n_rows = 0

    @property
    def sense(self) -> str:
        """Return the direction this model's objective is optimized in."""
        return self._sense

    @property
    def objective_constant(self) -> float:
        """Return the objective's fixed cost, zero where the objective has none."""
        return 0.0 if self._objective is None else self._objective.constant

    @property
    def objective(self) -> Any:
        """Return the expression the model's sense optimizes, or None."""
        return self._objective

    def __repr__(self) -> str:
        return (
            f"Model({self.name!r}, {len(self.variables)} variables, "
            f"{self.n_columns} columns, {self.n_rows} rows)"
        )

    @property
    def n_columns(self) -> int:
        """Return the number of columns declared."""
        return self._n_columns

    @property
    def n_rows(self) -> int:
        """Return the number of rows declared."""
        return self._n_rows

    @property
    def nnz(self) -> int:
        """Return the number of coefficients the assembled matrix contains."""
        return sum(c.nnz for c in self.constraints.values())

    def var(
        self,
        name: str,
        sets: Any,
        subset: Any = None,
        lower: float | Param = 0.0,
        upper: float | Param = np.inf,
        integer: bool = False,
    ) -> Variable:
        """Declare a variable over `sets`, or over `subset` of them."""
        if name in self.variables:
            raise ValueError(
                f"variable {name!r} is already declared; declare another name"
            )
        variable = Variable(
            name,
            sets,
            start=self._n_columns,
            total_columns=self._n_columns,
            subset=subset,
            lower=lower,
            upper=upper,
            integer=integer,
        )
        self.variables[name] = variable
        self._n_columns += variable.n_columns
        self._renumber()
        return variable

    def _adopt(self, variable: Variable) -> None:
        if variable.name in self.variables:
            raise ValueError(
                f"variable {variable.name!r} is already declared; declare another name"
            )
        variable._bind(self._n_columns, self._n_columns)
        self.variables[variable.name] = variable
        self._n_columns += variable.n_columns
        self._renumber()

    def _renumber(self) -> None:
        """Set the column space's current size on every variable."""
        for variable in self.variables.values():
            variable.total_columns = self._n_columns

    def constraint(
        self, name: str, relation: Any, where: Any = None, over: Any = None
    ) -> Constraint:
        """Declare a constraint from a comparison of an expression.

        The relation is symbolic. The constraint stores the term list, not a
        block. `where=` takes a domain over the constraint's free dimensions
        and narrows its rows. `over=` takes one and declares the rows.
        """
        if name in self.constraints:
            raise ValueError(
                f"constraint {name!r} is already declared; declare another name"
            )
        constraint = Constraint(name, relation, where, over)
        self.constraints[name] = constraint
        self._n_rows += constraint.n_rows
        return constraint

    def set_objective(self, expression: Any) -> None:
        """Set the objective the model's sense optimizes.

        A constant in the expression is a fixed cost. It adds no column, and
        the solved objective reports it beside the solver's value.
        """
        expression = read_at_its_sets(expression)
        if not isinstance(expression, Expression):
            raise TypeError(
                f"an objective is an expression over the model's columns; got "
                f"{type(expression).__name__}"
            )
        if expression.frame:
            raise ValueError(
                f"objective expression has free dimensions {expression.frame}; "
                f"sum the expression over them"
            )
        self._objective = expression

    def objective_coefficients(self) -> npt.NDArray[np.float64]:
        """Return one cost per column, zero where the objective has no term.

        The objective materialises here, not where it is set. Its column
        dimension spans every column the model declares, including those
        declared after it.
        """
        if self._objective is None:
            return np.zeros(self._n_columns, dtype=np.float64)
        self._renumber()
        block, _ = self._objective.materialise()
        return block.to_dense()

    def column_bounds(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the `(lower, upper)` bound of every column."""
        lower = np.empty(self._n_columns, dtype=np.float64)
        upper = np.empty(self._n_columns, dtype=np.float64)
        for variable in self.variables.values():
            variable.write_bounds(lower, upper)
        return lower, upper

    def integrality(self) -> npt.NDArray[np.int32]:
        """Return one flag per column, 1 where its variable is integer."""
        flags = np.zeros(self._n_columns, dtype=np.int32)
        for variable in self.variables.values():
            if variable.integer:
                at = slice(variable.start, variable.start + variable.n_columns)
                flags[at] = 1
        return flags

    def explain(self) -> Explanation:
        """Return what this model built, its declarations each with its count.

        A model contains variables and constraints. Its sets and parameters
        are read out of them and reported in order of first appearance.
        """
        parameters = self._parameters()
        dimensions, aliases = self._dimensions(parameters)
        return Explanation(
            name=self.name,
            sense=self.sense,
            built=True,
            sets=tuple(set_shape(s, len(s)) for s in dimensions),
            aliases=tuple((a.name, a.base.name) for a in aliases),
            parameters=tuple(param_shape(p, p.nnz) for p in parameters),
            variables=tuple(
                variable_shape(name, v, v.n_columns)
                for name, v in self.variables.items()
            ),
            constraints=tuple(
                ConstraintShape(
                    name, c.frame, c.sense, c.n_rows, c.nnz, render(c.relation)
                )
                for name, c in self.constraints.items()
            ),
            objective=objective_text(self._objective),
            objective_constant=objective_constant(self._objective),
            columns=self._n_columns,
            rows=self._n_rows,
            nonzeros=self.nnz,
        )

    def to_yaml(self, inline: bool = False, instructions: bool = False) -> str:
        """Return this model as file text, with its data inline where asked.

        The text contains the structure alone, or the structure and an inline
        data block. `save` writes a sidecar and the line that refers to it.
        This method writes no file. `instructions=True` prefixes the comment
        block that explains the format.
        """
        from nimopt.files import dumps, structure, to_inline

        mapping = structure(self)
        if inline:
            mapping["data"] = to_inline(self)
        return dumps(mapping, instructions)

    def _variable(self, name: str, other: str | None = None) -> Variable:
        """Return the named variable.

        Raises KeyError for a name that is not a declared variable. The message
        lists the declared variables. `other` is the action the message gives
        where `name` is a constraint.
        """
        if name in self.variables:
            return self.variables[name]
        valid = f"use one of {tuple(self.variables)}"
        if name in self.constraints:
            raise KeyError(
                f"{name!r} is a constraint, not a variable; {other or valid}"
            )
        raise KeyError(f"model {self.name!r} has no variable {name!r}; {valid}")

    def _constraint(self, name: str, other: str | None = None) -> Constraint:
        """Return the named constraint.

        Raises KeyError for a name that is not a declared constraint. The
        message lists the declared constraints. `other` is the action the
        message gives where `name` is a variable.
        """
        if name in self.constraints:
            return self.constraints[name]
        valid = f"use one of {tuple(self.constraints)}"
        if name in self.variables:
            raise KeyError(
                f"{name!r} is a variable, not a constraint; {other or valid}"
            )
        raise KeyError(f"model {self.name!r} has no constraint {name!r}; {valid}")

    def row(self, name: str, **coords: Any) -> "Row":
        """Return one row of this model's matrix, at the coordinate given.

        The row is read from the assembled matrix, and is the row the solver
        is given. Raises KeyError for a name that is not a declared
        constraint. A coordinate the constraint has no row at raises, and the
        message refers to `absent`.
        """
        from nimopt.row import position_of, read

        constraint = self._constraint(name)
        assembled = self.assemble()
        at = assembled.row_of(name).start + position_of(constraint, coords)
        return read(self, assembled, at)

    def absent(self, name: str) -> "Absence":
        """Return which coordinates fell out of the named constraint, and why.

        A dropped row leaves no trace in the matrix. This runs that one
        constraint's shape pass again with a recorder attached. The model and
        the rows already built are unchanged. Raises KeyError for a name that
        is not a declared constraint.
        """
        from nimopt.absence import Recorder
        from nimopt.constraint import narrow

        constraint = self._constraint(name)
        record = Recorder()
        narrow(constraint, record)
        return record.absence(name)

    def _parameters(self) -> tuple[Param, ...]:
        """Return every parameter this model reads, in order of first appearance.

        A derived coefficient is walked into. An objective over an arithmetic
        of two parameters reports both, with the sets they introduce.
        """
        found = {}

        def take(held: Any) -> None:
            if isinstance(held, Coefficient):
                leaves = held.parameters()
            else:
                leaves = (held,) if isinstance(held, Param) else ()
            for parameter in leaves:
                if found.setdefault(parameter.name, parameter) is not parameter:
                    raise ValueError(
                        f"model {self.name!r} reads two parameters named "
                        f"{parameter.name!r}; declare one parameter per name"
                    )

        for variable in self.variables.values():
            for held in (variable.lower, variable.upper, variable.subset):
                take(held)
        for constraint in self.constraints.values():
            for term in constraint.expression.terms:
                if term.coefficient is not None:
                    take(term.coefficient)
            take(constraint.rhs)
            take(constraint.where)
            take(constraint.over)
        if self._objective is not None:
            for term in self._objective.terms:
                if term.coefficient is not None:
                    take(term.coefficient)
        return tuple(found.values())

    def _dimensions(
        self, parameters: Iterable[Param]
    ) -> tuple[tuple[Any, ...], tuple[Alias, ...]]:
        """Return the sets and the aliases this model is declared over, in order.

        A dimension a coefficient introduces belongs to no variable, and the
        parameters are walked beside the variables. An alias reads a set's
        members and has none of its own. The two are reported apart, and a
        file contains data for a set alone. The base set of an alias is taken
        with the alias.
        """
        found = {}

        def take(dimension: Any) -> None:
            if isinstance(dimension, Alias):
                take(dimension.base)
            if found.setdefault(dimension.name, dimension) is not dimension:
                raise ValueError(
                    f"model {self.name!r} is declared over two sets named "
                    f"{dimension.name!r}; declare one set per name"
                )

        for variable in self.variables.values():
            for dimension in variable.sets:
                take(dimension)
        for parameter in parameters:
            for dimension in parameter.sets:
                take(dimension)
        held = tuple(found.values())
        return (
            tuple(d for d in held if not isinstance(d, Alias)),
            tuple(d for d in held if isinstance(d, Alias)),
        )

    def assemble(self, progress: Any = False) -> "Assembled":
        """Return the model's matrix and its row and column data, in CSR form.

        Each constraint rebuilds its expression, writes it into its slice of
        one buffer and releases it. One expression is live at a time. The row
        bounds are written into the model's own vectors after the matrix is
        built, and no row is copied twice.
        """
        report = _Written(reporter(progress), self.nnz, f"assembling {self.name}")
        buffer = EntryBuffer(2, self.nnz)
        rows_of = {}
        row_start = 0
        for name, constraint in self.constraints.items():
            constraint.write_into(buffer, row_start, report)
            report.constraint(name, constraint.nnz)
            rows_of[name] = slice(row_start, row_start + constraint.n_rows)
            row_start += constraint.n_rows
        report.close()

        matrix = buffer.array(
            {
                ROW: ProductCoord((row_start,)),
                COLUMN: ProductCoord((self._n_columns,)),
            },
            (ROW, COLUMN),
        )
        indices, values, indptr = matrix.to_csr()
        row_lower = np.empty(row_start, dtype=np.float64)
        row_upper = np.empty(row_start, dtype=np.float64)
        for name, constraint in self.constraints.items():
            constraint.write_bounds(row_lower[rows_of[name]], row_upper[rows_of[name]])
        col_lower, col_upper = self.column_bounds()
        return Assembled(
            matrix,
            indices,
            values,
            indptr,
            row_lower,
            row_upper,
            self.objective_coefficients(),
            col_lower,
            col_upper,
            self.integrality(),
            rows_of,
        )

    def session(
        self,
        solver: str = "highs",
        options: Mapping[str, Any] | None = None,
        progress: Any = False,
    ) -> "Session":
        """Return an open session on this model, for a solve and what follows.

        The matrix is assembled when the session opens. The solver's model is
        kept afterwards, and a conflict is read from the solved instance.
        """
        from nimopt.session import Session

        return Session(self, solver, options, progress)

    def solve(
        self,
        solver: str = "highs",
        options: Mapping[str, Any] | None = None,
        progress: Any = False,
    ) -> "Solution":
        """Assemble the model, solve it, and read the values back onto its sets.

        `options` gives the solver's settings under the names `options()`
        reports. `log=True` among them has the solver write its own iteration
        log. `progress=` reports the assembly. The assembly finishes before
        the solver starts.
        """
        with self.session(solver, options, progress) as session:
            return session.solve()


class Assembled:
    """A model's matrix over `(ROW, COLUMN)`, with its row and column data.

    `matrix` is the labeled array the model built. `indices` and `values` are
    views of the one buffer behind it, and only `indptr` is built. The extents
    are read from `matrix`.
    """

    def __init__(
        self,
        matrix: SparseArray,
        indices: npt.NDArray[np.int32],
        values: npt.NDArray[np.float64],
        indptr: npt.NDArray[np.int32],
        row_lower: npt.NDArray[np.float64],
        row_upper: npt.NDArray[np.float64],
        col_cost: npt.NDArray[np.float64],
        col_lower: npt.NDArray[np.float64],
        col_upper: npt.NDArray[np.float64],
        integrality: npt.NDArray[np.int32],
        rows_of: Mapping[str, slice],
    ) -> None:
        self.matrix = matrix
        self.indices = indices
        self.values = values
        self.indptr = indptr
        self.row_lower = row_lower
        self.row_upper = row_upper
        self.col_cost = col_cost
        self.col_lower = col_lower
        self.col_upper = col_upper
        self.integrality = integrality
        self._rows_of = rows_of

    def __repr__(self) -> str:
        return (
            f"Assembled({self.n_rows} rows, {self.n_cols} columns, "
            f"{self.values.size} nonzeros)"
        )

    @property
    def n_rows(self) -> int:
        """Return the number of rows of the matrix."""
        return self.matrix.shape[0]

    @property
    def n_cols(self) -> int:
        """Return the number of columns of the matrix."""
        return self.matrix.shape[1]

    def row_of(self, name: str) -> slice:
        """Return the range of rows the named constraint occupies.

        Raises KeyError for a name that is not a constraint of the matrix.
        """
        if name not in self._rows_of:
            raise KeyError(
                f"assembled matrix has no constraint {name!r}; use one of "
                f"{tuple(self._rows_of)}"
            )
        return self._rows_of[name]

    def to_dense(self) -> npt.NDArray[np.float64]:
        """Return the matrix as a dense ndarray."""
        return self.matrix.to_dense()
