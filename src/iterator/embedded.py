from dataclasses import dataclass
import itertools

import iterator
from iterator.builtins import (
    builtin_dispatch,
    is_none,
    lift,
    reduce,
    shift,
    deref,
    scan,
    domain,
    named_range,
    if_,
    minus,
    plus,
    mul,
    div,
    greater,
    nth,
    make_tuple,
)
from iterator.runtime import CartesianAxis, Offset
from iterator.utils import tupelize
import numpy as np
import numbers

EMBEDDED = "embedded"


class NeighborTableOffsetProvider:
    def __init__(self, tbl, origin_axis, neighbor_axis, max_neighbors) -> None:
        self.tbl = tbl
        self.origin_axis = origin_axis
        self.neighbor_axis = neighbor_axis
        self.max_neighbors = max_neighbors


@deref.register(EMBEDDED)
def deref(iter):
    return iter.deref()


@if_.register(EMBEDDED)
def if_(cond, t, f):
    return t if cond else f


@nth.register(EMBEDDED)
def nth(i, tup):
    return tup[i]


@make_tuple.register(EMBEDDED)
def make_tuple(*args):
    return (*args,)


@lift.register(EMBEDDED)
def lift(stencil):
    def impl(*args):
        class wrap_iterator:
            def __init__(self, *, offsets=[], elem=None) -> None:
                self.offsets = offsets
                self.elem = elem

            # TODO needs to be supported by all iterators that represent tuples
            def __getitem__(self, index):
                return wrap_iterator(offsets=self.offsets, elem=index)

            def shift(self, *offsets):
                return wrap_iterator(offsets=[*offsets, *self.offsets], elem=self.elem)

            def max_neighbors(self):
                # TODO cleanup, test edge cases
                open_offsets = get_open_offsets(*self.offsets)
                assert open_offsets
                assert isinstance(
                    args[0].offset_provider[open_offsets[0].value],
                    NeighborTableOffsetProvider,
                )
                return args[0].offset_provider[open_offsets[0].value].max_neighbors

            def deref(self):
                class DelayedIterator:
                    def __init__(self, wrapped_iterator, lifted_offsets, *, offsets=[]) -> None:
                        self.wrapped_iterator = wrapped_iterator
                        self.lifted_offsets = lifted_offsets
                        self.offsets = offsets

                    def is_none(self):
                        shifted = self.wrapped_iterator.shift(*self.lifted_offsets, *self.offsets)
                        return shifted.is_none()

                    def max_neighbors(self):
                        shifted = self.wrapped_iterator.shift(*self.lifted_offsets, *self.offsets)
                        return shifted.max_neighbors()

                    def shift(self, *offsets):
                        return DelayedIterator(
                            self.wrapped_iterator,
                            self.lifted_offsets,
                            offsets=[*offsets, *self.offsets],
                        )

                    def deref(self):
                        shifted = self.wrapped_iterator.shift(*self.lifted_offsets, *self.offsets)
                        return shifted.deref()

                shifted_args = tuple(map(lambda arg: DelayedIterator(arg, self.offsets), args))

                if any(shifted_arg.is_none() for shifted_arg in shifted_args):
                    return None

                if self.elem is None:
                    return stencil(*shifted_args)
                else:
                    return stencil(*shifted_args)[self.elem]

        return wrap_iterator()

    return impl


@reduce.register(EMBEDDED)
def reduce(fun, init):
    def sten(*iters):
        # assert check_that_all_iterators_are_compatible(*iters)
        first_it = iters[0]
        n = first_it.max_neighbors()
        res = init
        for i in range(n):
            # we can check a single argument
            # because all arguments share the same pattern
            if iterator.builtins.deref(iterator.builtins.shift(i)(first_it)) is None:
                break
            res = fun(
                res,
                *(iterator.builtins.deref(iterator.builtins.shift(i)(it)) for it in iters),
            )
        return res

    return sten


class _None:
    """Dummy object to allow execution of expression containing Nones in non-active path

    E.g.
    `if_(is_none(state), 42, 42+state)`
    here 42+state needs to be evaluatable even if is_none(state)

    TODO: all possible arithmetic operations
    """

    def __add__(self, other):
        return _None()

    def __radd__(self, other):
        return _None()

    def __sub__(self, other):
        return _None()

    def __rsub__(self, other):
        return _None()

    def __mul__(self, other):
        return _None()

    def __rmul__(self, other):
        return _None()

    def __truediv__(self, other):
        return _None()

    def __rtruediv__(self, other):
        return _None()

    def __getitem__(self, i):
        return _None()


@is_none.register(EMBEDDED)
def is_none(arg):
    return isinstance(arg, _None)


@domain.register(EMBEDDED)
def domain(*args):
    domain = {}
    for arg in args:
        domain.update(arg)
    return domain


@named_range.register(EMBEDDED)
def named_range(tag, start, end):
    return {tag: range(start, end)}


@minus.register(EMBEDDED)
def minus(first, second):
    return first - second


@plus.register(EMBEDDED)
def plus(first, second):
    return first + second


@mul.register(EMBEDDED)
def mul(first, second):
    return first * second


@div.register(EMBEDDED)
def div(first, second):
    return first / second


@greater.register(EMBEDDED)
def greater(first, second):
    return first > second


def named_range(axis, range):
    return ((axis, i) for i in range)


def domain_iterator(domain):
    return (
        dict(elem)
        for elem in itertools.product(*map(lambda tup: named_range(tup[0], tup[1]), domain.items()))
    )


def execute_shift(pos, tag, index, *, offset_provider):
    if tag in pos and pos[tag] is None:  # sparse field with offset as neighbor dimension
        new_pos = pos.copy()
        new_pos[tag] = index
        return new_pos
    assert tag.value in offset_provider
    offset_implementation = offset_provider[tag.value]
    if isinstance(offset_implementation, CartesianAxis):
        assert offset_implementation in pos
        new_pos = pos.copy()
        new_pos[offset_implementation] += index
        return new_pos
    elif isinstance(offset_implementation, NeighborTableOffsetProvider):
        assert offset_implementation.origin_axis in pos
        new_pos = pos.copy()
        del new_pos[offset_implementation.origin_axis]
        if offset_implementation.tbl[pos[offset_implementation.origin_axis], index] is None:
            return None
        else:
            new_pos[offset_implementation.neighbor_axis] = offset_implementation.tbl[
                pos[offset_implementation.origin_axis], index
            ]
        return new_pos

    assert False


# The following holds for shifts:
# shift(tag, index)(inp) -> full shift
# shift(tag)(inp) -> incomplete shift
# shift(index)(shift(tag)(inp)) -> full shift
# Therefore the following transformation holds
# shift(e2c,0)(shift(v2c,2)(cell_field))
# = shift(0)(shift(e2c)(shift(2)(shift(v2c)(cell_field))))
# = shift(v2c, 2, e2c, 0)(cell_field)
# = shift(v2c,e2c,2,0)(cell_field) <-- v2c,e2c twice incomplete shift
# = shift(2,0)(shift(v2c,e2c)(cell_field))
# for implementations it means everytime we have an index, we can "execute" a concrete shift
def group_offsets(*offsets):
    tag_stack = []
    index_stack = []
    complete_offsets = []
    for offset in offsets:
        if not isinstance(offset, int):
            if index_stack:
                index = index_stack.pop(0)
                complete_offsets.append((offset, index))
            else:
                tag_stack.append(offset)
        else:
            assert not tag_stack
            index_stack.append(offset)
    return complete_offsets, tag_stack


def shift_position(pos, *offsets, offset_provider):
    complete_offsets, open_offsets = group_offsets(*offsets)
    # assert not open_offsets # TODO enable this, check failing test and make everything saver

    new_pos = pos.copy()
    for tag, index in complete_offsets:
        new_pos = execute_shift(new_pos, tag, index, offset_provider=offset_provider)
        if new_pos is None:
            return None
    return new_pos


def get_open_offsets(*offsets):
    return group_offsets(*offsets)[1]


class MDIterator:
    def __init__(self, field, pos, *, offsets=[], offset_provider, column_axis=None) -> None:
        self.field = field
        self.pos = pos
        self.offsets = offsets
        self.offset_provider = offset_provider
        self.column_axis = column_axis

    def shift(self, *offsets):
        return MDIterator(
            self.field,
            self.pos,
            offsets=[*offsets, *self.offsets],
            offset_provider=self.offset_provider,
            column_axis=self.column_axis,
        )

    def max_neighbors(self):
        open_offsets = get_open_offsets(*self.offsets)
        assert open_offsets
        assert isinstance(self.offset_provider[open_offsets[0].value], NeighborTableOffsetProvider)
        return self.offset_provider[open_offsets[0].value].max_neighbors

    def is_none(self):
        return shift_position(self.pos, *self.offsets, offset_provider=self.offset_provider) is None

    def deref(self):
        shifted_pos = shift_position(self.pos, *self.offsets, offset_provider=self.offset_provider)

        if not all(axis in shifted_pos.keys() for axis in self.field.axises):
            raise IndexError("Iterator position doesn't point to valid location for its field.")
        slice_column = {}
        if self.column_axis is not None:
            slice_column[self.column_axis] = slice(shifted_pos[self.column_axis], None)
            del shifted_pos[self.column_axis]
        ordered_indices = get_ordered_indices(
            self.field.axises,
            shifted_pos,
            slice_axises=slice_column,
        )
        return self.field[ordered_indices]


def make_in_iterator(inp, pos, offset_provider, *, column_axis):
    sparse_dimensions = [axis for axis in inp.axises if isinstance(axis, Offset)]
    assert len(sparse_dimensions) <= 1  # TODO multiple is not a current use case
    new_pos = pos.copy()
    for axis in sparse_dimensions:
        new_pos[axis] = None
    if column_axis is not None:
        # if we deal with column stencil the column position is just an offset by which the whole column needs to be shifted
        new_pos[column_axis] = 0
    return MDIterator(
        inp,
        new_pos,
        offsets=[*sparse_dimensions],
        offset_provider=offset_provider,
        column_axis=column_axis,
    )


builtin_dispatch.push_key(EMBEDDED)  # makes embedded the default


class LocatedField:
    """A Field with named dimensions/axises.

    Axis keys can be any objects that are hashable.
    """

    def __init__(self, getter, axises, *, setter=None, array=None):
        self.getter = getter
        self.axises = axises
        self.setter = setter
        self.array = array

    def __getitem__(self, indices):
        indices = tupelize(indices)
        return self.getter(indices)

    def __setitem__(self, indices, value):
        if self.setter is None:
            raise TypeError("__setitem__ not supported for this field")
        self.setter(indices, value)

    def __array__(self):
        if self.array is None:
            raise TypeError("__array__ not supported for this field")
        return self.array()

    @property
    def shape(self):
        if self.array is None:
            raise TypeError("`shape` not supported for this field")
        return self.array().shape


def get_ordered_indices(axises, pos, *, slice_axises={}):
    """pos is a dictionary from axis to offset"""
    assert all(axis in [*pos.keys(), *slice_axises] for axis in axises)
    return tuple(pos[axis] if axis in pos else slice_axises[axis] for axis in axises)


def _tupsum(a, b):
    def combine_slice(s, t):
        is_slice = False
        if isinstance(s, slice):
            is_slice = True
            first = s.start
            assert s.step is None
            assert s.stop is None
        else:
            assert isinstance(s, numbers.Integral)
            first = s
        if isinstance(t, slice):
            is_slice = True
            second = t.start
            assert t.step is None
            assert t.stop is None
        else:
            assert isinstance(t, numbers.Integral)
            second = t
        start = first + second
        return slice(start, None) if is_slice else start

    return tuple(combine_slice(*i) for i in zip(a, b))


def np_as_located_field(*axises, origin=None):
    def _maker(a: np.ndarray):
        if a.ndim != len(axises):
            raise TypeError("ndarray.ndim incompatible with number of given axises")

        if origin is not None:
            offsets = get_ordered_indices(axises, origin)
        else:
            offsets = tuple(0 for _ in axises)

        def setter(indices, value):
            a[_tupsum(indices, offsets)] = value

        def getter(indices):
            return a[_tupsum(indices, offsets)]

        return LocatedField(getter, axises, setter=setter, array=a.__array__)

    return _maker


def index_field(axis):
    return LocatedField(lambda index: index[0], (axis,))


@iterator.builtins.shift.register(EMBEDDED)
def shift(*offsets):
    def impl(iter):
        return iter.shift(*reversed(offsets))

    return impl


@dataclass
class Column:
    axis: CartesianAxis
    range: range


class ScanArgIterator:
    def __init__(self, wrapped_iter, k_pos, *, offsets=[]) -> None:
        self.wrapped_iter = wrapped_iter
        self.offsets = offsets
        self.k_pos = k_pos

    def deref(self):
        return self.wrapped_iter.deref()[self.k_pos]

    def shift(self, *offsets):
        return ScanArgIterator(self.wrapped_iter, offsets=[*offsets, *self.offsets])


def shifted_scan_arg(k_pos):
    def impl(iter):
        return ScanArgIterator(iter, k_pos=k_pos)

    return impl


def fendef_embedded(fun, *args, **kwargs):
    assert "offset_provider" in kwargs

    @iterator.runtime.closure.register(EMBEDDED)
    def closure(domain, sten, outs, ins):  # domain is Dict[axis, range]

        column = None
        if "column_axis" in kwargs:
            _column_axis = kwargs["column_axis"]
            column = Column(_column_axis, domain[_column_axis])
            del domain[_column_axis]

        @iterator.builtins.scan.register(
            EMBEDDED
        )  # TODO this is a bit ugly, alternative: pass scan range via iterator
        def scan(scan_pass, is_forward, init):
            def impl(*iters):
                if column is None:
                    raise RuntimeError("Column axis is not defined, cannot scan.")

                _range = column.range
                if not is_forward:
                    _range = reversed(_range)

                state = init
                if state is None:
                    state = _None()
                cols = []
                for i in _range:
                    state = scan_pass(
                        state, *map(shifted_scan_arg(i), iters)
                    )  # more generic scan returns state and result as 2 different things
                    cols.append([*tupelize(state)])

                cols = tuple(map(np.asarray, (map(list, zip(*cols)))))
                # transpose to get tuple of columns as np array

                if not is_forward:
                    cols = tuple(map(np.flip, cols))
                return cols

            return impl

        for pos in domain_iterator(domain):
            ins_iters = list(
                make_in_iterator(
                    inp,
                    pos,
                    kwargs["offset_provider"],
                    column_axis=column.axis if column is not None else None,
                )
                for inp in ins
            )
            res = sten(*ins_iters)
            if not isinstance(res, tuple):
                res = (res,)
            if not len(res) == len(outs):
                IndexError("Number of return values doesn't match number of output fields.")

            for r, out in zip(res, outs):
                if column is None:
                    ordered_indices = get_ordered_indices(out.axises, pos)
                    out[ordered_indices] = r
                else:
                    colpos = pos.copy()
                    for k in column.range:
                        colpos[column.axis] = k
                        ordered_indices = get_ordered_indices(out.axises, colpos)
                        out[ordered_indices] = r[k]

    fun(*args)


iterator.runtime.fendef_registry[None] = fendef_embedded
