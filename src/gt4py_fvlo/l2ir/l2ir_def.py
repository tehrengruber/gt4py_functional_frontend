from typing import Any, List, Tuple, Union, Literal

import typing_inspect
from eve import SymbolName
from eve.datamodels import DataModel, field
from ..utils.uid import uid

BuiltInTypeMeta = Any

class L2IRNode(DataModel):
    pass

class Expr(L2IRNode):
    pass

class Constant(Expr):
    val: Union[int, bool, float, str]

class SymbolRef(Expr):
    name: str

class SymbolName(L2IRNode):
    name: str
    type_: Union[BuiltInTypeMeta, SymbolRef]

class Var(L2IRNode):
    name: SymbolName
    value: Expr

class Let(Expr):
    vars: Tuple[Var, ...]
    expr: Expr

class Function(Expr):
    pass

class Lambda(Function):
    args: Tuple[SymbolName, ...]
    expr: Expr

class BuiltInFunction(Function):
    name: str
    args: Tuple[SymbolName, ...]
    return_type: BuiltInTypeMeta

    def is_applicable(self, arg_types):
        if len(arg_types) != len(self.args):
            return False
        matches = True
        for arg, candidate_type in zip(self.args, arg_types):
            arg_type = arg.type_
            if typing_inspect.is_tuple_type(arg_type):
                arg_el_types = typing_inspect.get_args(arg_type)
                candidate_el_types = typing_inspect.get_args(candidate_type)
                if arg_el_types[-1] == Ellipsis:
                    arg_el_types = (*arg_el_types[:-1], *(arg_el_types[-2] for _ in range(len(candidate_el_types)-(len(arg_el_types)-1))))
                return arg_el_types == candidate_el_types
            elif arg_type in [int, str, bool, Array, Struct, ProductSet]:
                matches &= arg_type == candidate_type
                break
            else:
                raise NotImplementedError()

        return matches

class PolymorphicFunction(Function):
    overloads: Tuple[SymbolRef, ...]

class Call(Expr):
    func: Union[Function, SymbolRef]
    args: Tuple[Expr, ...]

class StructDecl(Expr):
    attr_names: Tuple[str, ...]
    attr_types: Tuple[BuiltInTypeMeta, ...]

    def get_attr_type(self, attr_name):
        for cand_attr_name, attr_type in zip(self.attr_names, self.attr_types):
            if attr_name == cand_attr_name:
                return attr_type
        raise TypeError()

class Construct(Function):
    pass

class GetStructAttr(Function):
    pass

class ConstructTuple(Function):
    pass

from collections import OrderedDict
declarations = OrderedDict()

def declare_function(name: str, arg_types: List[BuiltInTypeMeta], return_type: BuiltInTypeMeta):
    uid_ = uid('arg')
    args = tuple(SymbolName(name=f"arg{uid_}_{i}", type_=type_) for i, type_ in enumerate(arg_types))
    func = BuiltInFunction(name=name, args=args, return_type=return_type)
    if not name in declarations:
        declarations[name] = func
    else:
        if not isinstance(declarations[name], PolymorphicFunction):
            prev_declaration_name = f"_{name}_{uid(name)}"
            declarations[prev_declaration_name] = declarations[name]
            declarations[name] = PolymorphicFunction(overloads=(SymbolRef(prev_declaration_name),))

        declaration_name = f"_{name}_{uid(name)}"
        declarations[declaration_name] = func

        declarations[name] = PolymorphicFunction(overloads=(*declarations[name].overloads, SymbolRef(declaration_name)))

# built-in-types
Int = int
Index = int

import numpy as np
number_types = [int, np.int32, np.int64, np.float32, np.float64]

class Array:
    pass

class Struct:
    type_: StructDecl

from gt4py_fvlo.utils.index_space import UnitRange, ProductSet
import itertools

# index arithmetic
declare_function("add", (Index, Int), Index)
declare_function("sub", (Index, Index), int)
declare_function("mul", (Int, Index), Index)

# regular arithmetic
for number_type in number_types:
    for arity in range(1, 10):
        declare_function("add", tuple(itertools.repeat(number_type, arity)), number_type)
        declare_function("mul", tuple(itertools.repeat(number_type, arity)), number_type)

# tuple
for number_type in number_types:
    declare_function("getitem", (Tuple[number_type, ...], Int), number_type)

# ProductSet
for dim in range(1, 4):
    declare_function("getitem", (ProductSet, Tuple[tuple(itertools.repeat(Int, dim))]), Tuple[tuple(itertools.repeat(Int, dim))])

# field operations
declare_function("map", (Lambda, Array), ProductSet)
for number_type in number_types:
    declare_function("getitem", (Array, Tuple[Index, ...]), number_type)
    declare_function("reduce", (Lambda, UnitRange, Tuple[Union[Index, Literal[-1]], ...], Int, number_type), number_type)

for number_type in number_types:
    declare_function("scan", (Lambda, UnitRange, Tuple[number_type, ...]), Array)

#declare_function("scan", Lambda, ProductSet, Tuple[Number, ...])
#map(lambda i, j: scan(func, domain[0, 0, :], initial_state), domain[:, :, 0])

# view
#map(lambda pos: arr[pos], UnitRange(0, 5)*UnitRange(0, 5))
