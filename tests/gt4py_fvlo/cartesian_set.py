import pytest

from gt4py_fvlo.utils.index_space import UnitRange, union

def test_product_set():
    ref_set = UnitRange(1, 11)*UnitRange(1, 11)*UnitRange(1, 11)

    assert ref_set == UnitRange(1, 11)*UnitRange(1, 11)*UnitRange(1, 11)
    assert ref_set != UnitRange(2, 11)*UnitRange(1, 11)*UnitRange(1, 11)

    assert ref_set.size == 10*10*10

    set1 = ref_set[:, :, :]
    assert set1.size == 10 * 10 * 10
    for arg in set1.args:
        assert arg[0] == 1
        assert arg[-1] == 10

    set2 = ref_set[1:-1, 1:-1, 1:-1]
    assert set2.size == 8 * 8 * 8
    for arg in set2.args:
        assert arg[0] == 2
        assert arg[-1] == 9

    # ensure ref_set has not changed
    assert ref_set.size == 10 * 10 * 10

def test_product_set_2():
    ref_set = UnitRange(1, 7)*UnitRange(11, 16)*UnitRange(17, 29)

    set1 = ref_set[0:-1, 2:, -6:-5]

    for i, (low, high) in enumerate(((1, 5), (13, 15), (23, 23))):
        assert set1.args[i][0] == low
        assert set1.args[i][-1] == high

def test_product_set_3():
    ref_set = UnitRange(1, 7) * UnitRange(11, 16) * UnitRange(17, 29)

    assert (1, 11, 17) in ref_set
    assert (3, 13, 21) in ref_set
    assert (6, 15, 28) in ref_set

    assert (0, 11, 17) not in ref_set
    assert (1, 10, 17) not in ref_set
    assert (1, 11, 16) not in ref_set

    assert (7, 15, 28) not in ref_set

    assert ref_set.issubset(ref_set)

    subset1 = (UnitRange(5, 6) * UnitRange(11, 16) * UnitRange(17, 29))
    assert subset1.issubset(ref_set)

    subset2 = (UnitRange(0, 6) * UnitRange(11, 16) * UnitRange(17, 29))
    assert not subset2.issubset(ref_set)


def test_product_set_4():
    ref_set = UnitRange(1, 7) * UnitRange(11, 16) * UnitRange(17, 29)

    trans_set = ref_set.translate(1, 2, 3)

    assert trans_set.args[0][0] == 1 + 1
    assert trans_set.args[1][0] == 11 + 2
    assert trans_set.args[2][0] == 17 + 3


def test_product_set_extend():
    ref_set = UnitRange(1, 11) * UnitRange(1, 11) * UnitRange(1, 11)

    exset = ref_set.extend(1, 2, 3)

    assert exset.args[0][0] == 0
    assert exset.args[0][-1] == 11

    assert exset.args[1][0] == -1
    assert exset.args[1][-1] == 12

    assert exset.args[2][0] == -2
    assert exset.args[2][-1] == 13


def test_product_set_extend_2():
    ref_set = UnitRange(1, 11) * UnitRange(1, 11) * UnitRange(1, 11)

    exset = ref_set.extend((0, 0), (1, 2), 0)

    assert exset.args[0][0] == 1
    assert exset.args[0][-1] == 10

    assert exset.args[1][0] == 0
    assert exset.args[1][-1] == 12

    assert exset.args[2][0] == 1
    assert exset.args[2][-1] == 10


def test_product_set_iter():
    pass


def test_cartesian_set_hash():

    # product set
    assert hash(UnitRange(0, 10) * UnitRange(0, 10)) == hash(UnitRange(0, 10) * UnitRange(0, 10))
    assert hash(UnitRange(0, 10) * UnitRange(0, 10)) != hash(UnitRange(1, 10) * UnitRange(0, 10))

    # union set
    # todo: add tests for union, complement with different ordering

    # complement
    assert hash((UnitRange(0, 10) * UnitRange(0, 10)).without(UnitRange(5, 10) * UnitRange(5, 10))) != hash(
        (UnitRange(0, 10) * UnitRange(0, 10)).without(UnitRange(0, 10) * UnitRange(5, 10)))

def test_cartesian_set_bounds():
    samples = {
        (UnitRange(0, 10) * UnitRange(20, 30)): UnitRange(0, 10)*UnitRange(20, 30),
        union(UnitRange(0, 10) * UnitRange(0, 10), UnitRange(0, 20) * UnitRange(10, 20)): UnitRange(0, 20) * UnitRange(0, 20),
        (UnitRange(0, 10) * UnitRange(0, 10)).without(UnitRange(5, 10) * UnitRange(5, 10)): UnitRange(0, 10) * UnitRange(0, 10),
        (UnitRange(0, 10) * UnitRange(0, 10)).without(UnitRange(0, 10) * UnitRange(5, 10)): UnitRange(0, 10) * UnitRange(0, 5)
    }

    for arg, expected in samples.items():
        assert arg.bounds == expected

import math

def test_union_cartesian_set1():
    a = UnitRange(0, 3) * UnitRange(3, math.inf)
    b = UnitRange(3, 4) * UnitRange(1, 4)
    c = UnitRange(1, 3) * UnitRange(3, 4)

    bla = a.without(b, c)

    blub=1+1

def test_union_cartesian_set():
    inputs = [
        # 2D
        (UnitRange(0, 2) * UnitRange(0, 2), UnitRange(2, 4) * UnitRange(2, 4)),  # disjoint
        (UnitRange(0, 3) * UnitRange(0, 3), UnitRange(1, 4) * UnitRange(1, 4)),   # overlapping
        # 3D
        (UnitRange(0, 2) * UnitRange(0, 2) * UnitRange(0, 2),
         UnitRange(2, 4) * UnitRange(2, 4) * UnitRange(2, 4)),  # disjoint
        (UnitRange(0, 3) * UnitRange(0, 3) * UnitRange(0, 3),
         UnitRange(1, 4) * UnitRange(1, 4) * UnitRange(1, 4))  # overlapping
    ]
    exptected_sizes = [2*2*2, 3*3+5, 2*2*2*2, 3*3*3+5*3+4]
    expected_bounds = [
        UnitRange(0, 4) * UnitRange(0, 4),
        UnitRange(0, 4) * UnitRange(0, 4),
        UnitRange(0, 4)*UnitRange(0, 4)*UnitRange(0, 4),
        UnitRange(0, 4)*UnitRange(0, 4)*UnitRange(0, 4)
    ]
    for (a, b), size, bounds in zip(inputs, exptected_sizes, expected_bounds):
        ab = union(a, b)

        assert ab.size == size

        assert a.issubset(ab)
        assert b.issubset(ab)

        assert union(a, b) == ab

        assert ab.bounds == bounds

        assert not ab.empty


def test_union_cartesian_set_empty():
    a = union(UnitRange(1, 1) * UnitRange(0, 4), UnitRange(0, 4) * UnitRange(1, 1))

    assert a.empty


def test_union_cartesian_without():
    a = UnitRange(0, 2) * UnitRange(0, 2)
    b = UnitRange(3, 5) * UnitRange(0, 2)
    c = UnitRange(1, 4) * UnitRange(1, 3)

    ab_m_c = union(a, b).without(c)

    assert ab_m_c.intersect(c).empty
    assert set([(0, 0), (1, 0), (0, 1), (3, 0), (4, 0), (4, 1)]) == set(ab_m_c)


def test_union_cartesian_intersect():
    inputs = [
        (UnitRange(0, 2) * UnitRange(0, 2), UnitRange(3, 5) * UnitRange(0, 2)),
        (UnitRange(0, 2) * UnitRange(0, 2), UnitRange(1, 3) * UnitRange(1, 3))
    ]
    expected_intersection = [
        UnitRange(0, 0) * UnitRange(0, 0),
        UnitRange(1, 2) * UnitRange(1, 2)
    ]
    for (a, b), intersection in zip(inputs, expected_intersection):
        assert a.intersect(b) == intersection

def test_product_set_complement():
    a = UnitRange(1, 11) * UnitRange(1, 11) * UnitRange(1, 11)
    b = UnitRange(2, 10) * UnitRange(2, 10) * UnitRange(2, 10)

    assert a.without(b) == (UnitRange(1, 11) * UnitRange(1, 11) * UnitRange(1, 11)).without(UnitRange(2, 10) * UnitRange(2, 10) * UnitRange(2, 10))

    (UnitRange(1, 4) * UnitRange(1, 4))[1:2, 2:3] == UnitRange(2, 3) * UnitRange(3, 4)

    # test canonicalization
    assert (UnitRange(0, 4) * UnitRange(0, 4)).without(UnitRange(1, 2) * UnitRange(1, 5)) == (UnitRange(0, 4) * UnitRange(0, 4)).without(UnitRange(1, 2) * UnitRange(1, 6))

    # test elements
    a = UnitRange(0, 4) * UnitRange(0, 4)
    for p in [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]:
        b = UnitRange(p[0], p[0]+1) * UnitRange(p[1], p[1]+1)
        amb = a.without(b)

        # test point-wise
        for p in amb:
            assert p in a and p not in b

        for s in [a.extend(1, 1).without(a), b]:  # todo: use union
            for p in s:
                assert p not in amb

        # test "set-wise"
        assert a.complement().intersect(amb).empty
        assert b.intersect(amb).empty

        assert not a.intersect(b).issubset(amb)

        # assert a.intersect(b) + b == a # todo: UnionSet

def test_product_set_emptiness():
    r = UnitRange(0, 3)
    a = r*r

    ranges = [
        # inside
        UnitRange(0, 1), UnitRange(1, 2), UnitRange(2, 3),
        UnitRange(0, 2), UnitRange(1, 3),
        UnitRange(0, 3),
        # crossing
        UnitRange(-2, -1), UnitRange(-1, 0), UnitRange(2, 4),
        # outside
        UnitRange(3, 4), UnitRange(4, 5)
    ]

    # intersection
    #  if neither r1 nor r2 overlaps with r the intersection of a with r1*r2 should be empty
    for r1 in ranges:
        for r2 in ranges:
            expect_empty = r1.intersect(r).empty or r2.intersect(r).empty
            assert expect_empty == a.intersect(r1*r2).empty
