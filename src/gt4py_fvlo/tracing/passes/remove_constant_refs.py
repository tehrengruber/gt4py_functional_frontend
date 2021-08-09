from ..tracerir import *

class RemoveConstantRefs:
    def is_applicable(self, node, *, symtable):
        return isinstance(node, Symbol) and isinstance(symtable[node], Constant)

    def transform(self, node: Symbol, *, symtable):
        return symtable[node]