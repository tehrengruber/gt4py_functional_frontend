from ..tracerir_utils import peel_let
from ..tracerir import *
from ..pass_helper.collect_symbol_refs import CollectSymbolRefs

class RemoveUnusedSymbols:
    def is_applicable(self, node, *, symtable):
        if not isinstance(node, Let):
            return False
        used_symbols = CollectSymbolRefs.apply(node.expr)
        for var in node.vars:
            if var.name not in used_symbols:
                return True
        return False

    def transform(self, node: Let, *, symtable):
        used_symbols = CollectSymbolRefs.apply(node.expr)
        remove_vars = [var.name for var in node.vars if var.name not in used_symbols]
        return peel_let(node, remove_vars)