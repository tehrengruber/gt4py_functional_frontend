from typing import List, Any

from eve import NodeTranslator, NodeVisitor

from ...utils.uid import uid
from ..utils import merge_symtable, getclosurevars
from ..tracerir import *
from .collect_symbol_refs import CollectSymbolRefs
from .scope_visitor import ScopeTranslator

class CollectPyFunc(eve.NodeVisitor):
    def visit_PyFunc(self, node: PyFunc, *, collected_pyfuncs: List[PyFunc]):
        collected_pyfunc.append(node)

    @classmethod
    def apply(cls, node):
        collected_pyfuncs = []
        cls().visit(node, collected_pyfuncs=collected_pyfuncs)
        return collected_pyfuncs

def collect_symbol_refs_from_py_func(func: Callable):
    all_symbol_refs = {}

    closure_vars = [*getclosurevars(func).values()]

    for closure_var in closure_vars:
        from ..tracing import Tracer  # todo: remove circular import
        assert isinstance(closure_var, Tracer)
        closure_expr = closure_var.expr
        symbol_refs = CollectSymbolRefs.apply(closure_expr)
        pyfuncs = CollectPyFunc.apply(closure_expr)

        all_symbol_refs = [*all_symbol_refs, *symbol_refs, *[collect_symbol_refs_from_py_func(pyfunc.func) for pyfunc in pyfuncs]]
    return all_symbol_refs


class Conversion(ScopeTranslator):
    def visit_Symbol(self, node: Symbol, *, new_symtable, symtable, allow_unresolved, **kwargs):
        if node in new_symtable:
            return new_symtable[node]
        if not allow_unresolved:
            assert node in symtable
        return node

    def visit_PyFunc(self, node: PyFunc, *, new_symtable, **kwargs):
        symbol_refs = [*collect_symbol_refs_from_py_func(node.func),
                       *CollectSymbolRefs.apply(node.scope.expr).keys()]

        scope_expr = node.scope.expr.expr
        scope_vars = tuple(Var(old_symbol, new_symbol) for old_symbol, new_symbol in new_symtable.items() if old_symbol in symbol_refs)
        if len(scope_vars) > 0:
            scope_expr = Let(vars=scope_vars, expr=scope_expr)
        new_scope = Lambda(args=node.scope.expr.args, expr=scope_expr)

        return PyFunc(func=node.func, scope=PyFuncScope(new_scope))


def beta_reduction(node: Lambda, new_args, new_kwargs, closure_symbols={}, allow_unresolved=False):
    symtable = merge_symtable(
        {arg: new_arg for arg, new_arg in zip(node.args, new_args)},
        {s: s for s in new_kwargs.items()})
    return Conversion.apply(node.expr, symtable=closure_symbols, new_symtable=symtable, allow_unresolved=allow_unresolved)

def alpha_conversion(node, new_symtable, closure_symbols={}, allow_unresolved=False):
    return Conversion.apply(node, symtable=closure_symbols, new_symtable=new_symtable, allow_unresolved=allow_unresolved)