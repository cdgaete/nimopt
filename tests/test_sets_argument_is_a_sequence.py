import re

import numpy as np
import pytest

import nimopt as no
from nimopt.param import Param
from nimopt.variable import Variable

G = no.Set("G", np.array(["a", "b"]))

ENTRIES = (
    (
        "variable 'x' is given a single Set; pass a list of sets, such as [G]",
        lambda: Variable("x", G),
    ),
    (
        "parameter 'p' is given a single Set; pass a list of sets, such as [G]",
        lambda: Param("p", G),
    ),
    (
        "parameter 'p' is given a single Set; pass a list of sets, such as [G]",
        lambda: Param.from_dense("p", G, np.array([1.0, 2.0])),
    ),
    (
        "parameter 'p' is given a single Set; pass a list of sets, such as [G]",
        lambda: Param.from_long("p", G, {"G": np.array(["a"])}, np.array([1.0])),
    ),
    (
        "product is given a single Set; pass a list of sets, such as [G]",
        lambda: no.product(G),
    ),
    (
        "subset is given a single Set; pass a list of sets, such as [G]",
        lambda: no.subset(G, {"G": np.array(["a"])}),
    ),
    (
        "subset_of is given a single Set; pass a list of sets, such as [G]",
        lambda: no.subset_of(G, np.array([[0]])),
    ),
)


@pytest.mark.parametrize("message, call", ENTRIES)
def test_a_single_set_in_place_of_a_list_of_sets_raises_with_a_hint(message, call):
    with pytest.raises(TypeError, match=re.escape(message)):
        call()
