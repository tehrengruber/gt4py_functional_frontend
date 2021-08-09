from eve import NodeTranslator, NodeVisitor

from ..tracerir import *
from ..utils import merge_symtable

class ScopeTranslator(NodeTranslator):
    def visit_Lambda(self, node: Lambda, *, symtable, depth, **kwargs):
        expr_symtable = merge_symtable(symtable, {arg: arg for arg in node.args}, {arg: arg for arg in node.kwargs.values()})
        return Lambda(args=node.args, expr=self.visit(node.expr, symtable=expr_symtable, depth=depth+1, **kwargs))

    def visit_Let(self, node: Let, *, symtable, **kwargs):
        vars_ = tuple([Var(var.name, self.visit(var.value, symtable=symtable, **kwargs)) for var in node.vars])
        expr = self.visit(node.expr, symtable=merge_symtable(symtable, {var.name: var.value for var in vars_}), **kwargs)

        return Let(vars=vars_, expr=expr)

    @classmethod
    def apply(cls, node: GenericIRNode, **kwargs):
        kwargs = {"symtable": {}, "depth": 0, **kwargs}
        return cls().visit(node, **kwargs)

class ScopeVisitor(NodeVisitor):
    def visit_Lambda(self, node: Lambda, *, symtable, depth, **kwargs):
        expr_symtable = merge_symtable(symtable, {arg: arg for arg in node.args}, {arg: arg for arg in node.kwargs.values()})
        self.visit(node.expr, symtable=expr_symtable, depth=depth+1, **kwargs)

    def visit_Let(self, node: Let, *, symtable, **kwargs):
        # do not visit symbol declarations
        for var in node.vars:
            self.visit(var.value, symtable=symtable, **kwargs)
        self.visit(node.expr, symtable=merge_symtable(symtable, {var.name: var.value for var in node.vars}), **kwargs)

    @classmethod
    def apply(cls, node, **kwargs):
        kwargs = {"symtable": {}, "depth": 0, **kwargs}
        cls().visit(node, **kwargs)