from ..l2ir import *
from eve import NodeTranslator

class AlphaConversion(NodeTranslator):
    def visit_SymbolRef(self, node: SymbolRef, *, symtable):
        if node.name in symtable:
            return symtable[node.name]
        return node

def alpha_conversion(node, symtable):
    return AlphaConversion().visit(node, symtable=symtable)
