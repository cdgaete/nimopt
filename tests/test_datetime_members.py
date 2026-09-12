import datetime
import textwrap
import warnings

import numpy as np
import pytest

from nimopt import Definition, Model, Param, Set, Sum, load, loads, save

UNITS = ("ns", "us", "s", "D")
DELTAS = ("h", "ns")


def stamps(unit):
    """Three members of a datetime64 set, one day apart.

    A day is the coarsest unit under test, so the three members are distinct
    in every unit.
    """
    return np.array(
        ["2030-01-01", "2030-01-02", "2030-01-03"], dtype=f"datetime64[{unit}]"
    )


def spans(unit):
    """Three members of a timedelta64 set."""
    return np.array([1, 2, 3], dtype=f"timedelta64[{unit}]")


def model(members, fixed):
    """A model over `members`, fixing the first member where `fixed` is True."""
    m = Model("rt", sense="min")
    T = Set("T", members)
    cost = Param.from_dense("cost", (T,), np.array([1.0, 2.0, 3.0]))
    avail = Param.from_long("avail", (T,), {"T": members[:2]}, np.array([1.0, 1.0]))
    x = m.var("x", (T,), upper=10.0)
    m.constraint("total", Sum(T, x[T]) >= 3.0)
    m.constraint("cap", avail[T] * x[T] <= 2.0)
    if fixed:
        m.constraint("start", x[members[0]] == 1.0)
    m.set_objective(Sum(T, cost[T] * x[T]))
    return m


# minimize x0 + 2 x1 + 3 x2 with x0 + x1 + x2 >= 3 and x0, x1 at most 2:
# x0 = 2 and x1 = 1 costs 4; fixing x0 at 1 forces x1 = 2 and costs 5
OPTIMUM = {False: 4.0, True: 5.0}


def check(path, members, fixed):
    """Load the file at `path` and check its members, labels and objective."""
    back = load(path)
    got = back.solve()
    labels = got.primal("x").domain().labels()["T"]
    assert labels.dtype == members.dtype
    assert np.array_equal(labels, members)
    assert got.objective == pytest.approx(OPTIMUM[fixed])


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("unit", UNITS)
def test_a_datetime_set_round_trips_through_both_formats(tmp_path, unit, fixed, inline):
    members = stamps(unit)
    path = tmp_path / "m.yaml"
    save(model(members, fixed), path, inline=inline)
    check(path, members, fixed)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("unit", DELTAS)
def test_a_timedelta_set_round_trips_through_both_formats(
    tmp_path, unit, fixed, inline
):
    members = spans(unit)
    path = tmp_path / "m.yaml"
    save(model(members, fixed), path, inline=inline)
    check(path, members, fixed)


@pytest.mark.parametrize(
    ("unit", "written"),
    [
        ("ns", "2030-01-01T00:00:00.000000000"),
        ("us", "2030-01-01T00:00:00.000000"),
        ("s", "2030-01-01T00:00:00"),
        ("D", "2030-01-01"),
    ],
)
def test_a_fixed_datetime_member_is_written_as_a_quoted_iso_string(unit, written):
    text = model(stamps(unit), True).to_yaml()
    assert f"x['{written}'] == 1" in text


def test_a_fixed_timedelta_member_is_written_as_a_count_and_a_unit():
    text = model(spans("h"), True).to_yaml()
    assert "x['1 h'] == 1" in text


def test_an_inline_datetime_set_is_written_as_its_dtype_and_its_members():
    block = model(stamps("s"), False).to_yaml(inline=True)
    assert "dtype: datetime64[s]" in block
    assert "members: ['2030-01-01T00:00:00', '2030-01-02T00:00:00'" in block


def test_an_inline_timedelta_set_is_written_as_its_dtype_and_its_counts():
    block = model(spans("h"), False).to_yaml(inline=True)
    assert "dtype: timedelta64[h]" in block
    assert "members: [1, 2, 3]" in block


def test_a_saved_file_declares_version_three():
    assert "version: 3\n" in model(stamps("ns"), False).to_yaml()


VERSION_TWO = textwrap.dedent(
    """\
    version: 2
    name: d
    sense: min
    sets: [S]
    parameters:
      c: [S]
    variables:
      x: {sets: [S]}
    constraints:
      cap:
        relation: c[S] * x[S] <= 1
    data:
      S: [a, b]
      c: [2.0, 4.0]
    """
)


def test_a_version_two_file_with_no_datetime_data_loads():
    m = loads(VERSION_TWO)
    assert m.n_rows == 2
    assert m.n_columns == 2


def test_model_row_takes_an_iso_string_for_a_datetime_dimension():
    m = model(stamps("ns"), False)
    row = m.row("cap", T="2030-01-02T00:00:00.000000000")
    assert row.coordinate["T"] == np.datetime64("2030-01-02", "ns")
    assert row.index == 2


def test_model_row_takes_a_count_and_a_unit_for_a_timedelta_dimension():
    m = model(spans("h"), False)
    row = m.row("cap", T="2 h")
    assert row.coordinate["T"] == np.timedelta64(2, "h")


def test_a_row_displays_a_datetime_coordinate_as_its_iso_text():
    m = model(stamps("ns"), False)
    text = repr(m.row("cap", T="2030-01-02T00:00:00.000000000"))
    assert "cap[T='2030-01-02T00:00:00.000000000']" in text
    assert "x[2030-01-02T00:00:00.000000000]" in text


def test_a_row_displays_a_timedelta_coordinate_as_a_count_and_a_unit():
    m = model(spans("h"), False)
    text = repr(m.row("cap", T="2 h"))
    assert "cap[T='2 h']" in text
    assert "x[2 h]" in text


def test_an_absence_displays_a_datetime_coordinate_as_its_iso_text():
    text = repr(model(stamps("ns"), False).absent("cap"))
    assert "T='2030-01-03T00:00:00.000000000'" in text


def test_a_parameter_fixed_at_a_datetime_member_round_trips(tmp_path):
    members = stamps("ns")
    m = Model("rt", sense="min")
    T = Set("T", members)
    cost = Param.from_dense("cost", (T,), np.array([1.0, 2.0, 3.0]))
    x = m.var("x", (T,), upper=10.0)
    m.constraint("pin", cost[members[1]] * x[members[0]] == 4.0)
    m.set_objective(Sum(T, x[T]))
    path = tmp_path / "m.yaml"
    save(m, path)
    assert "cost['2030-01-02T00:00:00.000000000']" in path.read_text()
    # the coefficient is 2.0, so the pinned column is 2.0 and nothing else pays
    assert load(path).solve().objective == pytest.approx(2.0)


def test_model_row_takes_the_member_a_row_displays():
    m = model(stamps("ns"), False)
    row = m.row("cap", T=np.datetime64("2030-01-02", "ns"))
    assert m.row("cap", T=row.coordinate["T"]).index == row.index


def test_a_declared_set_converts_its_fixed_member_at_build():
    d = Definition("d", sense="min")
    T = d.set("T")
    x = d.var("x", (T,), upper=10.0)
    d.constraint("start", x["2030-01-01T00:00:00"] == 4.0)
    d.set_objective(Sum(T, x[T]))
    m = d.build({"T": stamps("ns")})
    assert m.solve().objective == pytest.approx(4.0)


def test_a_fixed_member_the_set_does_not_contain_raises():
    m = Model("rt")
    T = Set("T", stamps("ns"))
    x = m.var("x", (T,))
    with pytest.raises(ValueError, match="read it at a member that set contains"):
        x["2031-01-01T00:00:00"]


def test_a_fixed_member_finer_than_the_set_raises():
    m = Model("rt")
    T = Set("T", np.array(["2030-01-01T00", "2030-01-01T01"], dtype="datetime64[h]"))
    x = m.var("x", (T,))
    with pytest.raises(ValueError, match="does not convert exactly to datetime64"):
        x["2030-01-01T00:30"]


def test_a_malformed_iso_string_raises():
    m = Model("rt")
    T = Set("T", stamps("ns"))
    x = m.var("x", (T,))
    with pytest.raises(ValueError, match="is not an ISO 8601 datetime"):
        x["not-a-date"]


@pytest.mark.parametrize(
    "label",
    [
        "2030-01-01T00:00:00+02:00",
        "2030-01-01T00:00:00-05:00",
        "2030-01-01T00:00:00Z",
        datetime.datetime(2030, 1, 1, tzinfo=datetime.UTC),
        datetime.datetime(
            2030, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
        ),
    ],
)
def test_a_member_that_specifies_a_time_zone_raises(label):
    m = Model("rt")
    T = Set("T", stamps("ns"))
    x = m.var("x", (T,))
    with pytest.raises(ValueError, match="specifies a time zone"):
        x[label]


def test_a_naive_datetime_object_is_a_member():
    m = Model("rt")
    T = Set("T", stamps("ns"))
    x = m.var("x", (T,))
    m.constraint("start", x[datetime.datetime(2030, 1, 1)] == 1.0)
    assert "x['2030-01-01T00:00:00.000000000'] == 1" in m.to_yaml()


def test_a_member_outside_the_range_of_its_dtype_raises():
    m = Model("rt")
    T = Set("T", stamps("ns"))
    x = m.var("x", (T,))
    with pytest.raises(ValueError, match="is outside the range of datetime64"):
        x["9999-01-01"]


@pytest.mark.parametrize("text", ["NaT", "nat"])
def test_a_not_a_time_member_raises_and_numpy_emits_no_warning(text):
    m = Model("rt")
    T = Set("T", stamps("ns"))
    x = m.var("x", (T,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="is not a time at dimension"):
            x[text]


def test_an_inline_set_entry_missing_members_names_the_key():
    text = (INLINE_SET % "timedelta64[h]").replace("    members: [1, 2]\n", "")
    with pytest.raises(ValueError, match="declares no 'members'; write"):
        loads(text)


def test_a_right_hand_side_missing_a_row_shows_the_datetime_coordinate():
    members = stamps("ns")
    m = Model("rt")
    T = Set("T", members)
    rhs = Param.from_long("r", (T,), {"T": members[:1]}, np.array([1.0]))
    x = m.var("x", (T,))
    with pytest.raises(ValueError, match="'T': '2030-01-02T00:00:00.000000000'"):
        m.constraint("c", x[T] <= rhs[T], over=(T,))


def test_an_integer_against_a_datetime_set_raises():
    m = Model("rt")
    T = Set("T", stamps("ns"))
    x = m.var("x", (T,))
    with pytest.raises(ValueError, match="is not a datetime"):
        x[0]


def test_a_malformed_timedelta_text_raises():
    m = Model("rt")
    T = Set("T", spans("h"))
    x = m.var("x", (T,))
    with pytest.raises(ValueError, match="is not a timedelta"):
        x["1 hours"]


def test_an_integer_against_a_timedelta_set_counts_its_own_unit():
    m = Model("rt")
    T = Set("T", spans("h"))
    x = m.var("x", (T,))
    m.constraint("start", x[2] == 1.0)
    assert "x['2 h'] == 1" in m.to_yaml()


INLINE_SET = textwrap.dedent(
    """\
    version: 3
    name: d
    sense: min
    sets: [S]
    variables:
      x: {sets: [S]}
    constraints:
      cap:
        relation: x[S] <= 1
    data:
      S:
        dtype: %s
        members: [1, 2]
    """
)


def test_an_inline_set_entry_with_an_unknown_dtype_raises():
    with pytest.raises(ValueError, match="datetime64 or a timedelta64 dtype"):
        loads(INLINE_SET % "float64")


def test_an_inline_set_entry_with_an_unreadable_dtype_raises():
    with pytest.raises(ValueError, match="datetime64 or a timedelta64 dtype"):
        loads(INLINE_SET % "not-a-dtype")


def test_an_inline_member_finer_than_its_dtype_raises():
    text = (INLINE_SET % "datetime64[h]").replace(
        "members: [1, 2]", "members: ['2030-01-01T00:30']"
    )
    with pytest.raises(ValueError, match="does not convert exactly to datetime64"):
        loads(text)


def test_an_inline_integer_against_a_datetime_dtype_raises():
    with pytest.raises(ValueError, match="is not a datetime"):
        loads(INLINE_SET % "datetime64[h]")


def test_an_inline_set_entry_with_an_unknown_key_raises():
    text = INLINE_SET % "timedelta64[h]"
    message = "contains the unknown key 'labels'; write only 'dtype', 'members'"
    with pytest.raises(ValueError, match=message):
        loads(text.replace("members:", "labels:"))
