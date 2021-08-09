from typing import List

from ..utils.uid import uid

from .tracerir import *
from .pass_helper.conversion import beta_reduction
from .pass_helper.type_inference import TypeInference

def symbolic_args_from_args(args, kwargs, *, arg_prefixes=None):
    def symbol_for_arg(name_prefix, arg):
        name = f"{name_prefix}_{uid(name_prefix)}"
        type_ = TypeInference().visit(arg)
        return Symbol(name=name, type_=type_)

    if not arg_prefixes:
        arg_prefixes = [f"arg_{i}" for i in range(0, len(args))]

    symbolic_args = tuple((symbol_for_arg(arg_prefix, arg) for i, (arg_prefix, arg) in enumerate(zip(arg_prefixes, args))))
    symbolic_kwargs = {k: symbol_for_arg(k, arg) for k, arg in kwargs.items()}
    return symbolic_args, symbolic_kwargs

def evaluate_py_func(expr: PyFunc, symbolic_args, symbolic_kwargs={}, closure_symbols={}):
    from .tracing import Tracer # todo: remove cirular import
    res = expr.func(*(Tracer(a) for a in symbolic_args),
                     **{k: Tracer(v) for k, v in symbolic_kwargs.items()})
    body = res.expr if isinstance(res, Tracer) else Constant(res)

    from .pass_helper.collect_symbol_refs import CollectSymbolRefs
    refs = CollectSymbolRefs.apply(expr.scope.expr.expr)

    return Lambda(args=symbolic_args, kwargs=symbolic_kwargs, expr=beta_reduction(expr.scope.expr, (body,), {}, closure_symbols))

def let_reduction(node: Lambda, new_args, new_kwargs, closure_symbols={}):
    assert len(node.args)==len(new_args)
    symtable = {
        **closure_symbols,
        **{arg: new_arg for arg, new_arg in zip(node.args, new_args)},
        **{node.kwargs[k]: new_kwargs[k] for k in new_kwargs}
    }
    if isinstance(node.expr, PyFunc):
        scope = node.expr.scope.expr
        new_scope = Lambda(args=scope.args, expr=Let(vars=tuple(Var(old_symbol, new_symbol) for old_symbol, new_symbol in symtable.items()),
                        expr=scope.expr))

        #todo: check for collisions
        #assert len(symtable.keys() & set(node.expr.symtable)) == 0
        return PyFunc(node.expr.func, scope=PyFuncScope(new_scope))

    return Let(vars=tuple(Var(k, v) for k, v in symtable.items()), expr=node.expr)

def peel_let(let: Let, symbols: List[Symbol], *, transformer=None):
    vars_ = tuple(var for var in let.vars if var.name not in symbols)
    partial_symtable = {var.name: var.value for var in vars_}
    expr = let.expr if not transformer else transformer(let.expr, partial_symtable)
    if len(vars_) == 0:
        return expr
    return Let(vars=vars_, expr=expr)

def resolve_symbol(node, symtable, require_decl=False, expected_type=None):
    while isinstance(node, Symbol) and symtable[node] != node:
        node = symtable[node]
    if require_decl:
        assert not isinstance(node, Symbol)
    if expected_type:
        assert isinstance(node, expected_type)
    return node