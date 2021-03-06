from typing import List, Union
from eve import Node
from eve.traits import SymbolName, SymbolTableTrait
from eve.type_definitions import SymbolRef
from iterator.util.sym_validation import validate_symbol_refs


class Sym(Node):  # helper
    id: SymbolName


class Expr(Node):
    ...


class BoolLiteral(Expr):
    value: bool


class IntLiteral(Expr):
    value: int


class FloatLiteral(Expr):
    value: float  # TODO other float types


class StringLiteral(Expr):
    value: str


class NoneLiteral(Expr):
    _none_literal: int = 0


class OffsetLiteral(Expr):
    value: Union[int, str]


class AxisLiteral(Expr):
    value: str


class SymRef(Expr):
    id: SymbolRef


class Lambda(Expr, SymbolTableTrait):
    params: List[Sym]
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: List[Expr]


class FunctionDefinition(Node, SymbolTableTrait):
    id: SymbolName
    params: List[Sym]
    expr: Expr

    def __eq__(self, other):
        return isinstance(other, FunctionDefinition) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Setq(Node):
    id: SymbolName
    expr: Expr


class StencilClosure(Node):
    domain: Expr
    stencil: Expr
    outputs: List[SymRef]
    inputs: List[SymRef]


class FencilDefinition(Node, SymbolTableTrait):
    id: SymbolName
    params: List[Sym]
    closures: List[StencilClosure]


class Program(Node, SymbolTableTrait):
    function_definitions: List[FunctionDefinition]
    fencil_definitions: List[FencilDefinition]
    setqs: List[Setq]

    builtin_functions = list(
        Sym(id=name)
        for name in [
            "domain",
            "named_range",
            "compose",
            "lift",
            "is_none",
            "make_tuple",
            "nth",
            "reduce",
            "deref",
            "shift",
            "scan",
            "plus",
            "minus",
            "mul",
            "div",
            "greater",
            "less",
            "if_",
        ]
    )
    _validate_symbol_refs = validate_symbol_refs()
