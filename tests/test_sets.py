import numpy as np
import pytest

from nimopt.sets import Set, subset, subset_of


def sets():
    return Set("P", np.array(["p1", "p2"])), Set("W", np.array(["w1", "w2", "w3"]))


def test_a_set_names_its_members_and_answers_their_positions():
    P, _ = sets()
    assert len(P) == 2
    assert list(P.position_of(np.array(["p2", "p1"]))) == [1, 0]


def test_a_subset_is_a_domain_over_the_sets_it_spans():
    P, W = sets()
    got = subset((P, W), {"P": np.array(["p1", "p2"]), "W": np.array(["w2", "w1"])})
    assert got.dims == ("P", "W")
    assert got.shape == (2, 3)
    assert got.size == 2
    # the members come back in code order: ("p1","w2") then ("p2","w1")
    assert got.coordinates().tolist() == [[0, 1], [1, 0]]


def test_a_subsets_members_read_back_in_labels():
    P, W = sets()
    got = subset((P, W), {"P": np.array(["p2", "p1"]), "W": np.array(["w1", "w3"])})
    assert list(got.labels()["P"]) == ["p1", "p2"]
    assert list(got.labels()["W"]) == ["w3", "w1"]


def test_a_member_named_twice_raises():
    P, W = sets()
    with pytest.raises(ValueError, match="named twice"):
        subset((P, W), {"P": np.array(["p1", "p1"]), "W": np.array(["w1", "w1"])})


def test_a_label_column_naming_a_member_no_set_carries_raises():
    P, W = sets()
    with pytest.raises(KeyError):
        subset((P, W), {"P": np.array(["p9"]), "W": np.array(["w1"])})


def test_a_set_with_no_label_column_raises():
    P, W = sets()
    with pytest.raises(ValueError, match="no label column"):
        subset((P, W), {"P": np.array(["p1"])})


def test_members_are_also_named_by_position():
    P, W = sets()
    got = subset_of((P, W), np.array([[0, 1], [1, 0]], dtype=np.int32))
    # the columns name ("p1","w2") and ("p2","w1")
    assert got.coordinates().tolist() == [[0, 1], [1, 0]]


def test_the_two_constructors_name_the_same_members():
    P, W = sets()
    by_label = subset(
        (P, W), {"P": np.array(["p1", "p2"]), "W": np.array(["w2", "w1"])}
    )
    by_position = subset_of((P, W), np.array([[0, 1], [1, 0]], dtype=np.int32))
    assert by_label.coordinates().tolist() == by_position.coordinates().tolist()


def test_a_set_declares_without_members():
    S = Set("S")
    assert S.declared
    assert S.name == "S"


def test_a_declared_set_refuses_to_report_a_size():
    # a size of zero is a fact a caller acts on; an unbound set has none
    with pytest.raises(ValueError, match="set 'S' is declared and carries no members"):
        len(Set("S"))


def test_binding_gives_a_declared_set_its_members():
    S = Set("S")
    S._bind(np.array(["a", "b"]))
    assert not S.declared
    assert len(S) == 2
    assert list(S.position_of(np.array(["b"]))) == [1]


def test_a_set_given_members_is_not_declared():
    assert not Set("S", np.array(["a"])).declared
