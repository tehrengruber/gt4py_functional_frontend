import math
import operator
from typing import List, Callable, Tuple, Any, Union, Literal, Callable
import numpy as np
from eve.datamodels import DataModel, field
from .cartesian_sets import Dimension, Position, Index, UnitRange, ProductSet, CartesianSet, intersect, union, MaskedCartesianSet

connectivities = {}

image_type = np.ndarray
data_types = (int, float)

class Shift(DataModel):
    dim: Dimension
    offset: int

def _intersect_domains(a, *args):
    if any(arg == ... for arg in (a, *args)):
        # we don't know anything about the domains yet, so return an ellipsis again
        return ...
    return intersect(a, *args)

def _field_elwise_op_method_generator(op):
    def method(self, other):
        if isinstance(other, data_types):
            result = LazyField(self.domain, lambda pos: op(self[pos], other))
        else:
            result = LazyField(_intersect_domains(self.domain, other.domain), lambda pos: op(self[pos], other[pos]))
        if isinstance(result.domain, ProductSet):
            return result.materialize()
        return result
    return method

class AbstractField:
    def __getitem__(self, offset: Union[Position, CartesianSet, Shift]):
        if not isinstance(offset, Shift):
            # other types are handles by child classes
            raise TypeError()

        return LazyField(..., lambda pos: self[shift_position(pos, offset)])

    def __call__(self, offset):
        return LazyField(..., lambda pos: self[shift_position(pos, offset)])

    __add__ = _field_elwise_op_method_generator(operator.add)
    __sub__ = _field_elwise_op_method_generator(operator.sub)
    __mul__ = _field_elwise_op_method_generator(operator.mul)
    __rmul__ = _field_elwise_op_method_generator(operator.mul)

class Field(DataModel, AbstractField):
    domain: ProductSet
    image: image_type

    def _translate_to_memory_indices(self, pos):
        # make sure we have a tuple even for 1d fields
        image_idx = (pos,) if isinstance(pos, Index) else pos

        if not image_idx in self.domain:
            raise IndexError()

        return tuple(idx - o for idx, o in zip(image_idx, self.domain.bounds.origin))

    def __getitem__(self, arg):
        if isinstance(arg, (tuple, Position)):
            pos = arg
            assert pos in self.domain

            return self.image[self._translate_to_memory_indices(pos)]

        return super(Field, self).__getitem__(arg)

    def __setitem__(self, pos, val):
        self.image[self._translate_to_memory_indices(pos)] = val

class LazyField(DataModel, AbstractField):
    domain: Union[type(Ellipsis), CartesianSet] # ellipsis signifies that the domain is not known
    func: Callable[[Tuple[Index, ...]], Any]

    def __getitem__(self, arg):
        if isinstance(arg, (tuple, Position)):
            pos = arg
            assert self.domain == ... or arg in self.domain
            return self.func(pos)
        elif isinstance(arg, CartesianSet):
            domain = arg
            assert self.domain == ... or domain.issubset(self.domain)
            return LazyField(domain, self.func).materialize()

        return super(Field, self).__getitem__(arg)

    def materialize(self, out=None):
        inplace = out is not None
        if not inplace:  # allocate output
            out = Field(self.domain, np.zeros(self.domain.shape))

        # todo: very slow right now as we need to map to memory indices all the time
        for pos in self.domain:
            out[pos] = self.func(pos)

        if inplace:
            return
        return out

class LocalOperator(DataModel):
    func: Callable

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getitem__(self, domain):
        return FieldOperator(domain, self)

class FieldOperator(DataModel):
    domain: Union[type(Ellipsis), CartesianSet]
    local_op: LocalOperator

    def __call__(self, *fields, out=None) -> LazyField:
        valid_domain = _intersect_domains(*(field.domain for field in fields))
        assert self.domain == ... or valid_domain == ... or self.domain.issubset(valid_domain)
        domain = self.domain if self.domain != ... else valid_domain
        result = LazyField(domain, lambda pos: self.local_op(*(Iterator(pos, field) for field in fields)))
        if isinstance(domain, ProductSet):
            return result.materialize(out=out)
        return result

def beautified_iterator(func):
    return LocalOperator(func)

def lifted_iterator(func):
    return func

def shift_position(position, offset):
    return connectivities[type(offset.dim)](position, offset.offset)

class Iterator(DataModel):
    position: Position
    field: Union[LazyField, Field]

    def __call__(self, offset=None):
        if offset is None:
            position = self.position
        else:
            position = shift_position(self.position, offset)
        return self.field[position]