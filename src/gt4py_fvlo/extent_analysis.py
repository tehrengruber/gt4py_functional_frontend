from typing import Callable, Sequence

from .tracing import tracing
from .tracing.tracerir import Symbol, Constant, Call, PyExternalFunction
from gt4py_fvlo.utils.uid import uid

from .model import AbstractField, Field, TransparentView


class ClosureWrapper(tracing.ClosureWrapper):
    def __init__(self):
        self.closure_fields = {}

        super(ClosureWrapper, self).__init__()

    def wrap(self, arg):
        if isinstance(arg, AbstractField):
            sym = Symbol(f"symbolic_field_{uid('symbolic_field_')}", type_=type(arg))
            self.closure_fields[sym] = arg
            return sym
        return super(ClosureWrapper, self).wrap(arg)

def extent_analysis(num_dims, wrapped_stencil: Callable, fields: Sequence[AbstractField]):
    closure_wrapper = ClosureWrapper()

    symbolic_stencil = tracing.trace(wrapped_stencil, tuple(
        Symbol(name=dim, type_=int) for _, dim in zip(range(0, num_dims), ("I", "J", "K"))),
                                     closure_wrapper=closure_wrapper.wrap).expr

    symtable = {sym: sym for sym in closure_wrapper.closure_fields.keys()}

    from gt4py_fvlo.tracing.pass_helper.pass_manager import PassManager

    from gt4py_fvlo.tracing.passes.constant_folding import ConstantFold
    from gt4py_fvlo.tracing.passes.datamodel import DataModelConstructionResolver, DataModelMethodResolution, \
        DataModelGetAttrResolver, DataModelCallOperatorResolution, DataModelExternalGetAttrInliner
    from gt4py_fvlo.tracing.passes.opaque_call_resolution import OpaqueCallResolution1, OpaqueCallResolution2
    from gt4py_fvlo.tracing.passes.fix_call_type import FixCallType
    from gt4py_fvlo.tracing.passes.if_resolver import IfResolver
    from gt4py_fvlo.tracing.passes.tuple_getitem_resolver import TupleGetItemResolver
    from gt4py_fvlo.tracing.passes.remove_constant_refs import RemoveConstantRefs
    from gt4py_fvlo.tracing.passes.tracable_function_resolver import TracableFunctionResolver
    from gt4py_fvlo.tracing.passes.remove_unused_symbols import RemoveUnusedSymbols
    from gt4py_fvlo.tracing.passes.single_use_inliner import SingleUseInliner

    pass_manager = PassManager([
        DataModelConstructionResolver(),
        DataModelCallOperatorResolution(),
        DataModelGetAttrResolver(),
        DataModelMethodResolution(),
        OpaqueCallResolution1(),
        OpaqueCallResolution2(),
        FixCallType(),
        IfResolver(),
        TupleGetItemResolver(),
        RemoveConstantRefs(),
        TracableFunctionResolver(closure_wrapper=closure_wrapper.wrap)
    ])

    trc2 = pass_manager.visit(symbolic_stencil, symtable=symtable)

    trc3 = ConstantFold().visit(trc2, symtable=symtable)

    trc4 = PassManager([RemoveUnusedSymbols()]).visit(trc3, symtable=symtable)

    trc6 = SingleUseInliner.apply(trc4, symtable=symtable)

    trc7 = PassManager([DataModelExternalGetAttrInliner(), RemoveUnusedSymbols()]).visit(trc6, symtable=symtable)

    from gt4py_fvlo.tracing.pass_helper.scope_visitor import ScopeVisitor
    from gt4py_fvlo.tracing.pass_helper.conversion import beta_reduction, alpha_conversion
    from gt4py_fvlo.tracing.pass_helper.materializer import Materializer
    from gt4py_fvlo.tracing.tracerir_utils import resolve_symbol

    location = trc7.args

    # todo: instead subtract and check constant

    trc7 = alpha_conversion(trc7.expr, {idx: Constant(0) for idx in location}, closure_symbols=symtable)

    class ExtentAnalysis(ScopeVisitor):
        def visit_Call(self, node: Call, *, symtable, extents, **kwargs):
            if isinstance(node.func, PyExternalFunction) and (node.func.func == Field.__getitem__ or node.func.func == TransparentView.__getitem__):
                field_extent = Materializer().visit(node.args[1])
                field_sym = node.args[0]
                assert isinstance(field_sym, Symbol)
                if not field_sym in extents:
                    extents[field_sym] = []
                extents[field_sym].append(field_extent)

            return self.generic_visit(node, symtable=symtable, extents=extents, **kwargs)

        @classmethod
        def apply(cls, node):
            extents = {}
            super(ExtentAnalysis, cls).apply(node, extents=extents)
            return extents

    extents_per_symbol = ExtentAnalysis.apply(trc7)

    from functools import reduce

    extents = []
    for field in fields:
        accesses = []
        for cf, f in closure_wrapper.closure_fields.items():
            if id(f) == id(field) and cf in extents_per_symbol:
                accesses.extend(extents_per_symbol[cf])

        extents.append(set(accesses))

    return extents