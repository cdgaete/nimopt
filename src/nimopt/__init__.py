"""An LP/MILP builder whose constraint blocks are labeled sparse arrays."""

from nimopt.absence import Absence
from nimopt.coefficient import Coefficient
from nimopt.constraint import Constraint
from nimopt.definition import Definition
from nimopt.explanation import Explanation
from nimopt.files import load, loads, save
from nimopt.model import Assembled, Model
from nimopt.names import COLUMN, ROW
from nimopt.param import Param
from nimopt.row import Row
from nimopt.session import Diagnosis, Session
from nimopt.sets import Alias, Set, product, subset, subset_of
from nimopt.solution import Solution
from nimopt.solvers import Option, available, capabilities, options
from nimopt.term import Expression, Relation, Sum, Term
from nimopt.variable import Variable

__version__ = "0.2.1"

__all__ = [
    "COLUMN",
    "ROW",
    "Absence",
    "Alias",
    "Assembled",
    "Coefficient",
    "Constraint",
    "Definition",
    "Diagnosis",
    "Explanation",
    "Expression",
    "Model",
    "Option",
    "Param",
    "Relation",
    "Row",
    "Session",
    "Set",
    "Solution",
    "Sum",
    "Term",
    "Variable",
    "__version__",
    "available",
    "capabilities",
    "load",
    "loads",
    "options",
    "product",
    "save",
    "subset",
    "subset_of",
]
