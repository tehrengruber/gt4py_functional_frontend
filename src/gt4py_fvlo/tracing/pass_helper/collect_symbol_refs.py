from eve import NodeVisitor

from ..tracerir import Symbol, Lambda, Let, PyFunc
from .scope_visitor import ScopeVisitor

class CollectSymbolRefs(ScopeVisitor):
    def visit_Symbol(self, node: Symbol, *, symtable, sym_refs, collect_inner, **kwargs):
        if collect_inner or node not in symtable:
            if node not in sym_refs:
                sym_refs[node]=0
            sym_refs[node]+=1

    def visit_PyFunc(self, node: PyFunc, *, symtable, sym_refs, collect_inner, **kwargs):
        # todo: collisions?
        pyfunc_all_sym_refs = CollectSymbolRefs.apply(node.scope.expr).items()
        pyfunc_sym_refs = {s: count for s, count in pyfunc_all_sym_refs if s not in symtable}
        sym_refs.update(pyfunc_sym_refs)

    @classmethod
    def apply(cls, node, collect_inner=False):
        """
        :param node:
        :param collect_inner: also collect refs to symbols declared inside the node. invariant: no colliections
        :return:
        """
        sym_refs = {}
        super(CollectSymbolRefs, cls).apply(node, sym_refs=sym_refs, collect_inner=collect_inner)
        return sym_refs