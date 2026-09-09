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
from nimopt.spelling import spell
from nimopt.symbol import read_at_its_sets
from nimopt.term import Expression
from nimopt.variable import Variable

if TYPE_CHECKING:
    from nimopt.absence import Absence
    from nimopt.row import Row
    from nimopt.session import Session
    from nimopt.solution import Solution


class _Written:
    """The nonzeros a pass has written, told to a reporter as they land.

    A term's own count is known only once its constraint finishes, so a term
    moves the report by nothing and a constraint moves it by its own
    coefficients. Naming the term keeps the line moving while a large
    constraint is built.
    """

    def __init__(self, held: Any, total: int, what: str) -> None:
        self.held = held
        self.done = 0
        self.what = ""
        if held is not None:
            held.start(total, what)

    def term(self, term: Any) -> None:
        """A term of the constraint being written is built."""
        if self.held is not None:
            self.what = term.variable.name
            self.held.step(self.done, self.what)

    def constraint(self, name: str, nnz: int) -> None:
        """A constraint is written, carrying `nnz` coefficients."""
        self.done += nnz
        if self.held is not None:
            self.held.step(self.done, name)

    def close(self) -> None:
        """The pass is finished."""
        if self.held is not None:
            self.held.done()


class Model:
    """Variables numbered into one column space, constraints into one matrix.

    A variable's columns are a contiguous range of that space and are computed
    from its multi-index, so declaring a variable costs its bounds and nothing
    per column.
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
        """The direction this model's objective is optimised in."""
        return self._sense

    @property
    def objective_constant(self) -> float:
        """The fixed cost the objective states, zero where it states none."""
        return 0.0 if self._objective is None else self._objective.constant

    @property
    def objective(self) -> Any:
        """The expression the model's sense optimises, or None where none is set."""
        return self._objective

    def __repr__(self) -> str:
        return (
            f"Model({self.name!r}, {len(self.variables)} variables, "
            f"{self.n_columns} columns, {self.n_rows} rows)"
        )

    @property
    def n_columns(self) -> int:
        """Number of columns declared."""
        return self._n_columns

    @property
    def n_rows(self) -> int:
        """Number of rows declared."""
        return self._n_rows

    @property
    def nnz(self) -> int:
        """Number of coefficients the assembled matrix carries."""
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
            raise ValueError(f"variable {name!r} is already declared")
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
            raise ValueError(f"variable {variable.name!r} is already declared")
        variable._bind(self._n_columns, self._n_columns)
        self.variables[variable.name] = variable
        self._n_columns += variable.n_columns
        self._renumber()

    def _renumber(self) -> None:
        """Tell every variable the column space's current size."""
        for variable in self.variables.values():
            variable.total_columns = self._n_columns

    def eq(
        self, name: str, relation: Any, where: Any = None, over: Any = None
    ) -> Constraint:
        """Declare an equation from a comparison of an expression.

        The relation is symbolic, so the constraint holds the recipe rather
        than a block: declaring a hundred constraints costs a hundred shapes,
        not a hundred blocks. `where=` takes a domain over the constraint's
        free dimensions and narrows the rows it carries. `over=` takes one
        and states them outright, for a constraint whose terms each reach
        some of them.
        """
        if name in self.constraints:
            raise ValueError(f"constraint {name!r} is already declared")
        constraint = Constraint(name, relation, where, over)
        self.constraints[name] = constraint
        self._n_rows += constraint.n_rows
        return constraint

    def set_objective(self, expression: Any) -> None:
        """Set the objective the model's sense optimises.

        A constant the expression carries is a fixed cost: it states no
        column, and the solved objective reports it beside the solver's
        value.
        """
        expression = read_at_its_sets(expression)
        if not isinstance(expression, Expression):
            raise TypeError(
                f"an objective is an expression over the model's columns; a "
                f"constant alone states none, so {type(expression).__name__} "
                f"is not one"
            )
        if expression.frame:
            raise ValueError(
                f"an objective is over the column space alone; this "
                f"expression still carries free dimensions {expression.frame}"
            )
        self._objective = expression

    def objective_coefficients(self) -> npt.NDArray[np.float64]:
        """One cost per column, zero where the objective carries no term.

        The objective materialises here rather than where it is set, so its
        column dimension spans every column the model declares, including
        those declared after it.
        """
        if self._objective is None:
            return np.zeros(self._n_columns, dtype=np.float64)
        self._renumber()
        block, _ = self._objective.materialise()
        return block.to_dense()

    def column_bounds(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The `(lower, upper)` bound of every column."""
        lower = np.empty(self._n_columns, dtype=np.float64)
        upper = np.empty(self._n_columns, dtype=np.float64)
        for variable in self.variables.values():
            variable.write_bounds(lower, upper)
        return lower, upper

    def integrality(self) -> npt.NDArray[np.int32]:
        """One flag per column: 1 where the column's variable is integer."""
        flags = np.zeros(self._n_columns, dtype=np.int32)
        for variable in self.variables.values():
            if variable.integer:
                at = slice(variable.start, variable.start + variable.n_columns)
                flags[at] = 1
        return flags

    def explain(self) -> Explanation:
        """What this model built: its declarations, each with its count.

        A model holds variables and constraints, so its sets and parameters
        are walked out of them and reported in order of first appearance: a
        model does not carry the order they were declared in.
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
                    name, c.frame, c.sense, c.n_rows, c.nnz, spell(c.relation)
                )
                for name, c in self.constraints.items()
            ),
            objective=objective_text(self._objective),
            objective_constant=objective_constant(self._objective),
            columns=self._n_columns,
            rows=self._n_rows,
            nonzeros=self.nnz,
        )

    def to_yaml(self, inline: bool = False) -> str:
        """This model as the text of its file, with its data inline where asked.

        The text states the structure alone, or the structure and an inline
        block. Only `save` writes a sidecar and the line that names it, so
        this never states a file.
        """
        from nimopt.files import dumps, structure, to_inline

        mapping = structure(self)
        if inline:
            mapping["data"] = to_inline(self)
        return dumps(mapping)

    def row(self, name: str, **coords: Any) -> "Row":
        """One row of this model's matrix, at the coordinate named.

        The row is read from the assembled matrix, so what it shows is what
        the solver is given. A coordinate the constraint states no row at is
        refused, naming `absent` as the verb that says why.
        """
        from nimopt.row import position_of, read

        constraint = self.constraints[name]
        assembled = self.assemble()
        at = assembled.row_of(name).start + position_of(constraint, coords)
        return read(self, assembled, at)

    def absent(self, name: str) -> "Absence":
        """Which coordinates fell out of the named constraint, and by which rule.

        A dropped row leaves no trace in the matrix, so this re-runs that one
        constraint's shape pass with a recorder attached. Nothing the model
        carries moves: the re-run answers about the rows already built.
        """
        from nimopt.absence import Recorder
        from nimopt.constraint import narrow

        constraint = self.constraints[name]
        record = Recorder()
        narrow(constraint, record)
        return record.absence(name)

    def _parameters(self) -> tuple[Param, ...]:
        """Every parameter this model reads, in order of first appearance.

        A derived coefficient is walked into, so a model whose objective
        reads an arithmetic of two parameters reports both of them and the
        sets they introduce.
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
                        f"{parameter.name!r}; a name means one parameter"
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
        """The sets and the aliases this model is declared over, each in order.

        A dimension a coefficient introduces is carried by no variable, so the
        parameters are walked beside them. An alias reads a set's members and
        carries none of its own, so the two are reported apart: only a set
        names data a file states. The set an alias names is taken with it,
        because a model may carry the alias and never the set itself.
        """
        found = {}

        def take(dimension: Any) -> None:
            if isinstance(dimension, Alias):
                take(dimension.base)
            if found.setdefault(dimension.name, dimension) is not dimension:
                raise ValueError(
                    f"model {self.name!r} is declared over two sets named "
                    f"{dimension.name!r}; a name means one set"
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
        """The model's matrix, built into one buffer and handed over as CSR.

        Each constraint rebuilds its expression, writes it into its slice of a
        single buffer and releases it, so the matrix exists once and one
        expression stands beside it. A constraint's row bounds are written
        into the model's own vectors by the same rule, as a variable writes
        its column bounds, so no row is copied twice. They are written once
        the matrix is built, because the largest expression is alive while it
        is, and a vector held across that costs its whole length at the peak.

        Building each expression twice — once to measure its shape at
        declaration, once to write it — is what buys that: the merge
        dominates the cost either way, and build time is the cheaper of the
        two things being spent.
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
        """A live backend holding this model, across a solve and what follows.

        The matrix is assembled when the session opens, and the solver's model
        is held afterwards, so a conflict is a question about the instance
        that was solved.
        """
        from nimopt.session import Session

        return Session(self, solver, options, progress)

    def solve(
        self,
        solver: str = "highs",
        options: Mapping[str, Any] | None = None,
        progress: Any = False,
    ) -> "Solution":
        """Assemble the model, solve it, and read the answer back onto its sets.

        `options` states the solver's settings in the vocabulary `options()`
        names, and `log=True` among them has the solver write its own
        iteration log. `progress=` reports the assembly, which finishes
        before a solver starts, so the two never interleave.
        """
        with self.session(solver, options, progress) as session:
            return session.solve()


class Assembled:
    """A model's matrix over `(ROW, COLUMN)`, with its row and column data.

    `matrix` is the labeled array the model built; `indices` and `values` are
    views of the one buffer behind it and only `indptr` is built, so holding
    the array beside them costs a reference and no bytes. The extents come
    from it rather than being carried a second time.
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
        """Number of rows the matrix carries."""
        return self.matrix.shape[0]

    @property
    def n_cols(self) -> int:
        """Number of columns the matrix carries."""
        return self.matrix.shape[1]

    def row_of(self, name: str) -> slice:
        """The range of rows the named constraint occupies."""
        return self._rows_of[name]

    def to_dense(self) -> npt.NDArray[np.float64]:
        """The matrix as a dense ndarray."""
        return self.matrix.to_dense()
