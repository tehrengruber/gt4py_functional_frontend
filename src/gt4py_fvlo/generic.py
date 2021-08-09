from typing import Tuple, Dict, TypeVar, Generic, Optional, Union, Any

from eve.datamodels import DataModel, field
from eve.utils import FrozenNamespace

from .utils.uid import uid


class GenericIRNode(DataModel):
    pass

class Constant(GenericIRNode):
    val: Union[int, str, bool, float]

    def __repr__(self):
        return str(self.val)


class Symbol(GenericIRNode):
    name: str
    type_: Optional[Any]

    def __hash__(self):
        return hash(self.name)


class Function(GenericIRNode):
    pass


class BuiltInFunction(Function):
    name: str

    def __repr__(self):
        return self.name

# todo: make built-in
class Tuple_(GenericIRNode):
    elts: Tuple[GenericIRNode, ...]


class Lambda(Function):
    args: Tuple[Symbol, ...]
    expr: GenericIRNode

    kwargs: Dict[str, Symbol] = field(default_factory=lambda: dict(), kw_only=True)

    def __repr__(self):
        return f"{' '.join(map(str, self.args))} -> {self.expr}"

class Call(GenericIRNode):
    func: Function
    args: Tuple[GenericIRNode, ...]
    kwargs: Dict[str, GenericIRNode]

    def __repr__(self):
        return f"{self.func}({', '.join(map(str, self.args))})"

class Var(DataModel):
    name: Symbol
    value: GenericIRNode

class Let(GenericIRNode):
    vars: Tuple[Var, ...]
    expr: GenericIRNode

#import inspect
#x = None
#result = {}
#for x in locals():
#    if x != "base" and inspect.isclass(locals()[x]) and issubclass(locals()[x], base):
#        result[x] = locals()[x]
#fs = FrozenNamespace(**result)