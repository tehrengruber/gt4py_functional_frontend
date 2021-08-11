from ..tracerir import *
from ..pass_helper.scope_visitor import ScopeTranslator, ScopeVisitor
from ..pass_helper.collect_symbol_refs import CollectSymbolRefs
from .global_symbol_collision_resolver import GlobalSymbolCollisionCollector

class SymbolRefDeclDepth(ScopeVisitor):
    def visit_Lambda(self, node: Lambda, *, decl_depths, depth, **kwargs):
        for arg in node.args:
            assert arg not in decl_depths
            decl_depths[arg] = depth+1
        super().visit_Lambda(node, depth=depth, decl_depths=decl_depths, **kwargs)

    def visit_Let(self, node: Let, *, decl_depths, depth, **kwargs):
        for var in node.vars:
            assert var.name not in decl_depths
            decl_depths[var.name] = depth
        super().visit_Let(node, depth=depth, decl_depths=decl_depths, **kwargs)

    @classmethod
    def apply(cls, node):
        decl_depths = {}
        cls().visit(node, decl_depths=decl_depths, depth=0, symtable={})
        return decl_depths


class SingleUseInliner(ScopeTranslator):
    def visit_Symbol(self, node: Symbol, *, ref_counts, decl_depths, symtable, depth):
        if ref_counts[node] == 1 and depth == decl_depths[node]:
            return symtable[node]
        return node

    @classmethod
    def apply(cls, node, symtable=None):
        if symtable == None:
            symtable = {}

        # ensure we have no symbol collisions
        collisions = GlobalSymbolCollisionCollector.apply(node)
        assert not collisions

        # the actual inlining
        ref_counts = CollectSymbolRefs.apply(node, collect_inner=True)
        decl_depths = {**{sym: 0 for sym in symtable.keys()}, **SymbolRefDeclDepth.apply(node)}
        transformed_node = super(SingleUseInliner, cls).apply(node, ref_counts=ref_counts, decl_depths=decl_depths, symtable=symtable)

        return transformed_node