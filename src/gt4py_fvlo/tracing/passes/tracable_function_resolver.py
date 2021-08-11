import inspect
from eve import NodeTranslator

from ..tracing import is_tracable
from ..tracerir_utils import symbolic_args_from_args, evaluate_py_func, let_reduction
from .constant_folding import ConstantFold
from ..pass_helper.collect_symbol_refs import CollectSymbolRefs
from ..tracerir import *

class TracableFunctionResolver():
    def __init__(self, *, closure_wrapper=None, custom_tracable_externals=[]):
        if not closure_wrapper:
            from ..tracing import wrap_closure_var
            closure_wrapper = wrap_closure_var
        self.closure_wrapper = closure_wrapper
        self.custom_tracable_externals = custom_tracable_externals

    def _is_tracable_external(self, func):
        return isinstance(func, PyExternalFunction) and (is_tracable(func.func) or func.func in self.custom_tracable_externals)

    def is_applicable(self, node, *, symtable):
        return isinstance(node, Call) and (isinstance(node.func, PyFunc) or self._is_tracable_external(node.func))

    def transform(self, node: Call, *, symtable):
        argspec = inspect.getfullargspec(node.func.func)
        arg_prefixes = [arg for arg in argspec.args]
        if argspec.varargs:
            for i, arg in enumerate(node.args[len(argspec.args):]):
                arg_prefixes.append(f"{argspec.varargs}_{i}_{uid(argspec.varargs)}")
        symbolic_args, symbolic_kwargs = symbolic_args_from_args(node.args, node.kwargs,
                                                                 arg_prefixes=arg_prefixes)
        if self._is_tracable_external(node.func):
            from ..tracing import trace # todo: remove ciruclar import
            traced_func_expr = trace(node.func.func, symbolic_args, symbolic_kwargs, closure_wrapper=self.closure_wrapper).expr
        elif isinstance(node.func, PyFunc):
            symbol_refs = set(CollectSymbolRefs.apply(node.func.scope.expr).keys())
            assert symbol_refs.issubset(symtable.keys())
            traced_func_expr = evaluate_py_func(node.func, symbolic_args, symbolic_kwargs, symtable)

        if len(node.args) == 0 and not node.kwargs:
            return traced_func_expr.expr

        new_expr = let_reduction(traced_func_expr, node.args, node.kwargs)
        #new_expr = ConstantFold().visit(new_expr, symtable=symtable)

        return new_expr