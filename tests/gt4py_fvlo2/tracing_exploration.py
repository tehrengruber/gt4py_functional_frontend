import inspect
import types
from types import SimpleNamespace
from typing import Callable, Optional, Tuple, Generic, TypeVar
import typing_inspect

from eve.datamodels import datamodel, DataModel

from gtc_unstructured.frontend.built_in_types import BuiltInType

from gt4py_fvlo.tracing.tracing import trace, tracable
from gt4py_fvlo.tracing.tracerir import Symbol

# structured dims
K = types.new_class("K", ())
# unstructured dims
V, E, C = (types.new_class(type_name, ()) for type_name in ["V", "E", "C"])
# "sparse" dims / local connectivities
V2V, V2E, V2C = (types.new_class(type_name, ()) for type_name in ["V2V", "V2E", "V2C"])
E2V, E2E, E2C = (types.new_class(type_name, ()) for type_name in ["E2V", "E2E", "E2C"])
C2V, C2E, C2C = (types.new_class(type_name, ()) for type_name in ["C2V", "C2E", "C2C"])

class Field(BuiltInType):
    pass

class Connectivity(BuiltInType):
    pass

T = TypeVar("T")

class Position(DataModel, Generic[T]):
    position: Tuple[int, ...]

    def __getitem__(self, item):
        return self.position[item]

    @classmethod
    def __len__(cls):
        T = cls.__args__[0]
        assert typing_inspect.is_tuple_type(T)
        dims = typing_inspect.get_args(T)
        return len(dims)

class LocalOperator(DataModel):
    func: Callable
    name_: Optional[str]

@tracable
def local_operator(func):
    return LocalOperator(func, None)

gt = SimpleNamespace(local_operator=local_operator)

import numpy as np
DT = np.double

@gt.local_operator
def stencil(
    inp: Field[[V], DT], v2v: Connectivity[V, V2V], *, position: Position[Tuple[V]]
):
    (v,) = position
    return inp[v2v[v, 0]] + inp[v2v[v, 1]] + inp[v2v[v, 2]] + inp[v2v[v, 3]]

# todo: validate
arg_spec = inspect.getfullargspec(stencil.func)

res = trace(stencil.func,
            symbolic_args=tuple(Symbol(arg_name, arg_spec.annotations[arg_name]) for arg_name in arg_spec.args),
            symbolic_kwargs={arg_name: Symbol(arg_name, arg_spec.annotations[arg_name]) for arg_name in arg_spec.kwonlyargs})

bla = 1+1