import types
import typing
from typing import Callable, Dict, Union, Any, Tuple

import eve

from ..generic import *

def _py_func_scope_factory():
    py_func_expr_symbol = Symbol("py_func_" + str(uid("py_func_")), type_=typing.Any)
    return PyFuncScope(Lambda(args=(py_func_expr_symbol,), expr=py_func_expr_symbol))

# wrap scope inside an object opaque to visitors
class PyFuncScope:
    def __init__(self, expr):
        self.expr = expr

class PyFunc(Function):
    func: Callable

    # use a lambda function to allow capturing conversions until the function is evaluated eventually
    scope: PyFuncScope = eve.datamodels.field(default_factory=_py_func_scope_factory, kw_only=True)

from eve.datamodels import root_validator

class PyClosureVar(GenericIRNode):
    val: Any

    @root_validator
    def _root_validator(cls, instance):
        if instance.val is None:
            raise ValueError("'name' value cannot appear in 'friends' list.")

class PyExternalFunction(Function):
    func: Callable

class PyGeneratorExpr(GenericIRNode):
    generator: types.GeneratorType

class PyZip(GenericIRNode):
    iterables: Tuple[GenericIRNode, ...]

class PyZipIteratorEl(GenericIRNode):
    expr: GenericIRNode

class OpaqueCall(GenericIRNode):
    func: GenericIRNode # todo: revisit, formalize
    args: Tuple[GenericIRNode, ...]
    kwargs: Dict[str, GenericIRNode]

class DataModelConstruction(GenericIRNode):
    type_: type
    attrs: Dict[str, GenericIRNode]