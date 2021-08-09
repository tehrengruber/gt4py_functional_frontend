from typing import List

from eve import NodeTranslator

from ..tracerir import *
from ..utils import merge_symtable

class PassManager(NodeTranslator):
    def __init__(self, passes: List[NodeTranslator]):
        self.passes = passes

    def visit_Lambda(self, node: Lambda, *, symtable):
        expr = self.visit(node.expr, symtable=merge_symtable(symtable, {arg: arg for arg in [*node.args, *node.kwargs.values()]}))
        return self._transform(Lambda(args=node.args, expr=expr), symtable=symtable)

    def visit_Let(self, node: Let, *, symtable):
        vars_ = self.visit(node.vars, symtable=symtable)
        expr = self.visit(node.expr, symtable=merge_symtable(symtable, {var.name: var.value for var in vars_}))
        return self._transform(Let(vars=vars_, expr=expr), symtable=symtable)

    def visit_Var(self, node: Var, *, symtable):
        return Var(name=node.name, value=self.visit(node.value, symtable=symtable))

    def visit_Constant(self, node: Constant, **_):
        return node

    def _transform(self, node, *, symtable):
        for pass_ in self.passes:
            if pass_.is_applicable(node, symtable=symtable):
                transformed_node = pass_.transform(node, symtable=symtable)
                return self.visit(transformed_node, symtable=symtable)
        return node

    def generic_visit(self, node, *, symtable):
        node = super().generic_visit(node, symtable=symtable)
        return self._transform(node, symtable=symtable)

    @classmethod
    def apply(cls, passes: List[NodeTranslator], node: GenericIRNode):
        return cls(passes).visit(node, symtable={})
