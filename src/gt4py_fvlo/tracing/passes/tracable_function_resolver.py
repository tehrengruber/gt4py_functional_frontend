import inspect
from eve import NodeTranslator

from ..tracerir_utils import symbolic_args_from_args, evaluate_py_func, let_reduction
from .constant_folding import ConstantFold
from ..pass_helper.collect_symbol_refs import CollectSymbolRefs
from ..tracerir import *

def is_tracable_external(expr):
    from ..tracing import _tracable
    return isinstance(expr, PyExternalFunction) and expr.func in _tracable

class TracableFunctionResolver():
    def __init__(self, *, closure_wrapper=None):
        if not closure_wrapper:
            from ..tracing import wrap_closure_var
            closure_wrapper = wrap_closure_var
        self.closure_wrapper = closure_wrapper

    def is_applicable(self, node, *, symtable):
        return isinstance(node, Call) and (is_tracable_external(node.func) or isinstance(node.func, PyFunc))

    def transform(self, node: Call, *, symtable):
        argspec = inspect.getfullargspec(node.func.func)
        arg_prefixes = [arg for arg in argspec.args]
        if argspec.varargs:
            for i, arg in enumerate(node.args[len(argspec.args):]):
                arg_prefixes.append(f"{argspec.varargs}_{i}_{uid(argspec.varargs)}")
        symbolic_args, symbolic_kwargs = symbolic_args_from_args(node.args, node.kwargs,
                                                                 arg_prefixes=arg_prefixes)
        if is_tracable_external(node.func):
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