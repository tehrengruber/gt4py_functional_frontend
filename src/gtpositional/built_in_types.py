import types
from typing import TypeVar, Tuple, Type, Generic, Union, Literal, Any
from gtpositional.utils.built_in_type import built_in_type
from eve.datamodels import DataModel
import typing_inspect
import numpy as np

class Dimension:
    pass

DomainArgsT = TypeVar("DomainT", bound=Tuple[Type[Dimension], ...])
DT = TypeVar("DT")

@built_in_type
class Domain(Generic[DomainArgsT, DT]):
    pass

# structured dims
I = types.new_class("I", (Dimension,))
J = types.new_class("J", (Dimension,))
K = types.new_class("K", (Dimension,))

FromT = TypeVar("Dimension", bound=Dimension)
ToT = TypeVar("ToT", bound=Dimension)
MaxNeighborT = TypeVar("MaxNeighborT")
HasSkipValuesT = TypeVar("HasSkipValuesT", bound=Literal[True, False])

@built_in_type
class Connectivity(Dimension, Generic[FromT, ToT, MaxNeighborT, HasSkipValuesT]):
    pass

# unstructured dims
Vertex = types.new_class("Vertex", (Dimension,))
Edge = types.new_class("Edge", (Dimension,))
Cell = types.new_class("Cell", (Dimension,))

@built_in_type
class Field(DataModel, Generic[DomainArgsT, DT]):
    domain: Any #ProductSet
    image: np.ndarray

    def __getitem__(self, image_idx):
        # make sure we have a tuple even for 1d fields
        image_idx = (image_idx,) if isinstance(image_idx, int) else image_idx

        if any(isinstance(idx, slice) for idx in image_idx):
            assert all(not isinstance(idx, slice) or idx == slice(None) for idx in image_idx)
            relative_idx = tuple(slice(None) if idx == slice(None) else idx - o for idx, o in zip_(image_idx, self.domain.bounds.origin))
            return Field(self.domain[relative_idx], self.image[relative_idx])

        if len(image_idx) != self.domain.dim:
            raise TypeError()
        if not image_idx in self.domain:
            raise IndexError()
        #assert image_idx in self.domain
        memory_idx = tuple(idx-o for idx, o in zip_(image_idx, self.domain.bounds.origin))
        return self.image[memory_idx]

field = Field[(Vertex, Edge), np.float32]

Field(axis=123, dtype=123)

bla = Domain[(Vertex, Edge), (1, 2, 3)]
blub = 1+1

DimT = TypeVar("DimT", bound=Dimension)


#bla = Index[Vertex](1)


class Index(Generic[DimT]):
    axis: DimT
    value: Union[int, float]

    def __init__(self, value = None, axis: DimT = None):
        bla=1+1
        ...

    def get_axis(self) -> DimT:
        return self.axis


#index = Index(value=123, axis=I)

index = Index[I](value=123)


#d=bla/1

blub = 1+1