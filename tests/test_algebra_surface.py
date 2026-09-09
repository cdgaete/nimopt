import numpy as np
import pytest

from nimopt import Model, Param, Set, Sum


def parts():
    P = Set("P", np.array(["a", "b"]))
    m = Model("m")
    x = m.var("x", (P,))
    c = Param.from_dense("c", (P,), np.array([2.0, 3.0]))
    return P, m, x, c


def test_a_lag_that_is_not_a_whole_number_is_refused_rather_than_truncated():
    # a lag of 1.7 stating a lag of 1 builds a different model and reports
    # success, which is the failure this refusal exists to prevent
    P, *_ = parts()
    with pytest.raises(ValueError, match="a lag is a whole number"):
        P - 1.7
    with pytest.raises(ValueError, match="a lag is a whole number"):
        P + 0.5


def test_a_lag_written_as_a_whole_float_is_the_lag_it_states():
    P, _, x, _ = parts()
    assert x[P - 1.0].terms[0].shifts == x[P - 1].terms[0].shifts


def test_a_lag_of_a_lag_is_refused():
    P, *_ = parts()
    with pytest.raises(TypeError, match="one lag"):
        (P - 1) - 1


def test_a_numpy_array_is_not_a_coefficient():
    # numpy claims the operator and broadcasts, returning an array of
    # expressions that no later error explains
    P, _, x, _ = parts()
    with pytest.raises(TypeError, match="Param.from_dense"):
        np.array([1.0, 2.0]) * x[P]


def test_a_numpy_array_is_not_a_coefficient_for_a_parameter_either():
    P, _, _, c = parts()
    with pytest.raises(TypeError, match="Param.from_dense"):
        np.array([1.0, 2.0]) * c[P]


def test_numpy_is_not_asked_to_reduce_an_expression():
    # np.sum returns the expression unchanged, having reduced nothing
    P, _, x, _ = parts()
    with pytest.raises(TypeError, match="Sum"):
        np.sum(x[P])


# each row is a spelling and the fragment its refusal names. A form that
# raises a bare Python TypeError fails: "unsupported operand type(s)" tells a
# caller nothing about what nimopt expresses.
REFUSED = [
    ("x[P] * y[P]", "linear"),
    ("x[P] ** 2", "linear"),
    ("2.0 / x[P]", "linear"),
    ("x[P] / y[P]", "linear"),
    ("abs(x[P])", "Sum"),
    ("min(x[P], y[P])", "Sum"),
    ("x[P] < 1.0", "<="),
    ("x[P] > 1.0", ">="),
    ("x[P] != 1.0", "one bound"),
    ("(x[P] <= 1.0) <= 2.0", "one bound"),
    ("0.0 <= x[P] <= 1.0", "each bound"),
    ("bool(x[P])", "truth value"),
    ("Sum(P, Sum(P, x[P]))", "reduced once"),
    ("Sum(P, c[P])", "a variable"),
    ("c[P] <= d[P]", "a variable"),
    ("x[P] + c[P]", "a variable"),
    ("c * 2.0", "read it at its sets"),
    ("c[P] / zero[P]", "zero"),
    ("x[P] / 0.0", "zero"),
    ("c[P] ** d[P]", "a power takes a number"),
    ("abs(c[P])", "before"),
    ("Sum(P - 1, x[P])", "state the lag"),
    ("c[P - 1]", "state the lag"),
    ("P - 1.7", "whole number"),
    ("(P - 1) - 1", "one lag"),
    ("m.eq('r', x[P] <= 'a')", "right-hand side"),
    ("m.eq('r', x[P] <= None)", "right-hand side"),
    ("np.sum(x[P])", "Sum"),
    ("np.array([1.0, 2.0]) * x[P]", "Param.from_dense"),
    ("len(x[P])", "no length"),
]


def _surface():
    """The names every row of the table is evaluated against."""
    P = Set("P", np.array(["a", "b"]))
    m = Model("m")
    x = m.var("x", (P,))
    y = m.var("y", (P,))
    c = Param.from_dense("c", (P,), np.array([2.0, 3.0]))
    d = Param.from_dense("d", (P,), np.array([5.0, 7.0]))
    zero = Param.from_dense("zero", (P,), np.array([0.0, 4.0]))
    return {
        "P": P,
        "m": m,
        "x": x,
        "y": y,
        "c": c,
        "d": d,
        "zero": zero,
        "Sum": Sum,
        "Param": Param,
        "np": np,
    }


@pytest.mark.parametrize("spelling,names", REFUSED)
def test_a_form_nimopt_does_not_express_says_so(spelling, names):
    with pytest.raises((TypeError, ValueError, ZeroDivisionError)) as raised:
        eval(spelling, _surface())  # noqa: S307
    message = str(raised.value)
    assert names in message, (spelling, message)
    assert "unsupported operand type" not in message, (spelling, message)
    assert "not supported between instances" not in message, (spelling, message)


REWRITES = [
    ("0.0 <= x[P] <= 1.0", ["m.eq('lo', x[P] >= 0.0)", "m.eq('hi', x[P] <= 1.0)"]),
    ("Sum(P - 1, x[P])", ["Sum(P, x[P - 1])"]),
    ("x[P] < 1.0", ["x[P] <= 1.0"]),
    (
        "np.array([1.0, 2.0]) * x[P]",
        ["Param.from_dense('k', (P,), np.array([1.0, 2.0]))[P] * x[P]"],
    ),
    ("c * 2.0", ["c[P] * 2.0"]),
    ("x[P] ** 2", ["(c[P] ** 2) * x[P]"]),
    ("c[P] / zero[P]", ["c[P] / d[P]"]),
]


@pytest.mark.parametrize("refused,rewrites", REWRITES)
def test_the_rewrite_a_message_names_builds_the_model_it_states(refused, rewrites):
    # a message suggesting code that does not work is worse than one
    # suggesting nothing, and error messages rot faster than anything else
    for rewrite in rewrites:
        assert eval(rewrite, _surface()) is not None  # noqa: S307


def test_the_builtin_sum_of_a_list_of_expressions_adds_them():
    # an expression carries a constant, so the zero the built-in sum starts
    # from adds to one: the reduction it states is the arithmetic it does
    P, _, x, c = parts()
    added = sum([c[P] * x[P], c[P] * x[P]])
    assert len(added.terms) == 2
    assert added.constant == 0.0


def test_the_table_covers_every_form_the_guide_states():
    # the documentation page and this table read the same list, so a refusal
    # cannot be documented without also being executed
    from docs_blocks import DOCS

    page = (DOCS / "guides" / "coefficient-arithmetic.md").read_text()
    missing = [s for s, _ in REFUSED if f"`{s}`" not in page and s.startswith("x[")]
    assert missing == [], missing
