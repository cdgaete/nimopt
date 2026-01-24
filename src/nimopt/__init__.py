"""
nimopt - Algebraic optimization modeling with GAMS-like syntax.

Powered by nimblend for efficient labeled array operations.

Example
-------
>>> import nimopt as no
>>>
>>> # Sets
>>> i = no.Set('i', ['seattle', 'sandiego'])
>>> j = no.Set('j', ['newyork', 'chicago', 'topeka'])
>>>
>>> # Parameters (backed by nimblend arrays)
>>> a = no.Param('a', [i], [350, 600])
>>> d = no.Param('d', [i, j], [[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]])
>>>
>>> # Model
>>> m = no.Model(sense='minimize')
>>> x = m.var('x', [i, j], lb=0)
>>>
>>> # Equations
>>> m.eq('supply', no.Sum(j)(x[i,j]) <= a[i])
>>> m.to_lp('model.lp')
"""

from .expression import Constraint, LinearExpr
from .functions import Sum, abs_, exp, log, power, sqrt
from .model import Model
from .param import Param, ParamRef
from .sets import LaggedSet, Set
from .solution_api import extract_solution, load_solution, to_csv
from .variable import Variable, VarRef

__version__ = "0.2.0"

__all__ = [
    "Set",
    "LaggedSet",
    "Param",
    "ParamRef",
    "Variable",
    "VarRef",
    "LinearExpr",
    "Constraint",
    "Sum",
    "sqrt",
    "exp",
    "log",
    "abs_",
    "power",
    "Model",
    "extract_solution",
    "load_solution",
    "to_csv",
]
