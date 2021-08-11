import operator
from typing import List, Callable, Tuple, Any, Union, Literal
import numpy as np
from eve.datamodels import DataModel, field
from .utils.index_space import UnitRange, ProductSet, CartesianSet, intersect, union
from .tracing.tracing import tracable, is_tracable, isinstance_, if_, zip_, tuple_


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

def lazy_map(stencil, domain):
    return LazyMap(stencil, domain)

#image_type = LazyMap
#map_ = lazy_map

image_type = np.ndarray
def map_(stencil, domain):
    return np.array([stencil(*idx) for idx in domain]).reshape(domain.shape)


class Field(DataModel, AbstractField):
    domain: ProductSet
    image: image_type

    #@tracable
    def __getitem__(self, image_idx):
        # make sure we have a tuple even for 1d fields
        image_idx = (image_idx,) if isinstance(image_idx, int) else image_idx

        if any(isinstance(idx, slice) for idx in image_idx):
            assert all(not isinstance(idx, slice) or idx == slice(None) for idx in image_idx)
            relative_idx = tuple(slice(None) if idx == slice(None) else idx - o for idx, o in zip_(image_idx, self.domain.origin))
            return Field(self.domain[relative_idx], self.image[relative_idx])

        if len(image_idx) != self.domain.dim:
            raise TypeError()
        if not image_idx in self.domain:
            raise IndexError()
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

        per_field_accesses = extent_analysis(domain.dim, wrapped_stencil, fields)

        per_field_valid_domains = []
        for field, accesses in zip(fields, per_field_accesses):
            accessable_field_domain = field.field.domain if isinstance(field, TransparentView) else field.domain
            per_field_valid_domains.append(intersect(accessable_field_domain, *(accessable_field_domain.translate(*map(operator.neg, access)) for access in accesses)))

        valid_domain = intersect(*per_field_valid_domains)
        domain = intersect(*(field.domain for field in fields))
        if not domain.issubset(valid_domain):
            raise ValueError("Not enough halo lines.")

        return apply_stencil(wrapped_stencil, valid_domain).transparent_view(domain)

    if any(r.start == -math.inf or r.stop == math.inf for r in domain.args):
        return Field(domain, lazy_map(stencil, domain))

    return Field(domain, map_(stencil, domain))

@tracable
def fmap(stencil: "Callable", field: "Field"):
    return apply_stencil(stencil, field.domain, field)

# adapter to iterator view fields
def located_field_as_fvlo_field(located_field, origin=None):
    image = located_field.array()
    domain = ProductSet.from_shape(image.shape)
    if origin:
        domain = domain.translate(*(-o for o in origin))
    return Field(domain, image)