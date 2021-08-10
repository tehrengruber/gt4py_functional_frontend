import ast
import operator
from copy import copy
import types
from types import LambdaType
from typing import List, Callable, Tuple, Any, Union, Literal
import inspect
import textwrap
import tempfile
import eve.datamodels as datamodels
from functools import reduce
import typing_inspect
from eve.datamodels import DataModel, field
from gt4py_fvlo.utils.index_space import UnitRange, ProductSet, CartesianSet, intersect, union
import gt4py_fvlo
import numpy as np

from gt4py_fvlo.tracing.tracing import tracable, is_tracable, isinstance_, if_, zip_, tuple_


class AbstractField:
    pass


class LazyMap(DataModel):
    func: Callable
    domain: ProductSet

    def __getitem__(self, memory_idx):
        image_idx = tuple(image_idx + o for image_idx, o in zip(memory_idx, self.domain.origin))
        assert image_idx in domain
        return self.func(image_idx)

    def materialize(self):
        return np.array([self.func(idx) for idx in self.domain]).reshape(self.domain.shape)

#image_type = LazyMap
#def _map(stencil, domain):
#    return LazyMap(stencil, domain)

image_type = np.ndarray
def map_(stencil, domain):
    return np.array([stencil(*idx) for idx in domain]).reshape(domain.shape)

#getitem(map(lambda i, j: getitem(field, (i, j)), domain), (i, j)) = field(i, j)

class Field(DataModel, AbstractField):
    domain: ProductSet
    image: image_type

    #@tracable
    def __getitem__(self, image_idx):
        assert image_idx in self.domain
        memory_idx = tuple(idx-o for idx, o in zip_(image_idx, self.domain.origin))
        return self.image[memory_idx]

    def view(self, domain):
        assert domain.issubset(self.domain)
        return apply_stencil(lambda *idx: self[idx], domain)

    def transparent_view(self, domain):
        if domain == self.domain:
            return self
        return TransparentView(domain, self)


class TransparentView(DataModel, AbstractField):
    domain: ProductSet
    field: Field

    def __getitem__(self, position):
        assert position in self.field.domain
        return self.field[position]


@tracable
def new_accessor(field, position):
    def accessor(*shift):
        gen = (idx+s for idx, s in zip_(position, shift))
        idx = tuple_(gen)
        return field[idx]
    return accessor

@tracable
def apply_stencil(stencil: "Callable", domain, *fields):
    if len(fields) > 0:
        assert is_tracable(stencil)
        assert all(isinstance_(field, AbstractField) for field in fields)
        @tracable
        def wrapped_stencil(*pos):
            return stencil(*(new_accessor(field, pos) for field in fields))

        from .extent_analysis import extent_analysis

        per_field_accesses = extent_analysis(wrapped_stencil, fields)

        per_field_valid_domains = []
        for field, accesses in zip(fields, per_field_accesses):
            accessable_field_domain = field.field.domain if isinstance(field, TransparentView) else field.domain
            per_field_valid_domains.append(intersect(accessable_field_domain, *(accessable_field_domain.translate(*map(operator.neg, access)) for access in accesses)))

        valid_domain = intersect(*per_field_valid_domains)
        domain = intersect(*(field.domain for field in fields))
        if not domain.issubset(valid_domain):
            raise ValueError("Not enough halo lines.")

        return apply_stencil(wrapped_stencil, valid_domain).transparent_view(domain)
    return Field(domain, map_(stencil, domain))

@tracable
def fmap(stencil: "Callable", field: "Field"):
    return apply_stencil(stencil, field.domain, field)