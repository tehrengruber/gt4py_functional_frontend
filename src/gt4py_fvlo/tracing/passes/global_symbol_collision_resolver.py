from ..tracerir import *
from ...utils.uid import uid
from ..pass_helper.scope_visitor import ScopeVisitor, ScopeTranslator
from ..pass_helper.conversion import alpha_conversion

class GlobalSymbolCollisionCollector(ScopeVisitor):
    def visit_Lambda(self, node: Lambda, *, collisions, decls, **kwargs):
        for arg in node.args:
            if arg in decls:
                collisions.append(arg)
            decls[arg] = node
        super().visit_Lambda(node, collisions=collisions, decls=decls, **kwargs)

    def visit_Let(self, node, *, collisions, decls, **kwargs):
        for var in node.vars:
            if var.name in decls:
                collisions.append(var.name)
            decls[var.name] = var.value
        super().visit_Let(node, collisions=collisions, decls=decls, **kwargs)

    @classmethod
    def apply(cls, node, decls=None): # if you want the decl to a collision just pass a dict yourself
        if not decls:
            decls = {}
        collisions = []
        super(GlobalSymbolCollisionCollector, cls).apply(node, decls=decls, collisions=collisions)
        return collisions

class GlobalSymbolCollisionResolver(ScopeTranslator):
    def visit_Lambda(self, node: Lambda, *, symtable, collisions, **kwargs):
        new_symtable = {arg: (arg if arg not in collisions else Symbol(name=arg.name+f"_c{uid('collision')}", type_=arg.type_)) for arg in node.args}
        args = tuple(new_symtable[arg] for arg in node.args)
        expr = self.visit(alpha_conversion(node.expr, new_symtable, closure_symbols=symtable), symtable=symtable, collisions=collisions, **kwargs)
        return Lambda(args=args, expr=expr)

    def visit_Let(self, node: Let, *, symtable, collisions, **kwargs):
        if len(collisions) == 0:
            return node
        new_symtable = {
            var.name: (var.name if var.name not in collisions else Symbol(name=var.name.name + f"_c{uid('collision')}", type_=var.name.type_)) for
            var in node.vars}
        new_collisions = [collision for collision in collisions if collision not in new_symtable]
        vars_ = tuple(Var(name=new_symtable[var.name], value=self.visit(var.value, symtable=symtable, collisions=new_collisions, **kwargs)) for var in node.vars)
        # todo: pass correct symboltable
        expr = self.visit(alpha_conversion(node.expr, new_symtable, closure_symbols=symtable, allow_unresolved=True),
                          symtable=symtable, collisions=new_collisions, **kwargs)
        return Let(vars=vars_, expr=expr)

    @classmethod
    def apply(cls, node, **kwargs):
        collisions = GlobalSymbolCollisionCollector.apply(node)
        new_node = super(GlobalSymbolCollisionResolver, cls).apply(node, collisions=collisions, **kwargs)
        assert not GlobalSymbolCollisionCollector.apply(new_node)
        return new_node