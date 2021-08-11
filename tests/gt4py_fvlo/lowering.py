import typing_inspect
import numpy as np

from gt4py_fvlo.utils.index_space import UnitRange, ProductSet, CartesianSet, intersect, union

from gt4py_fvlo.tracing.tracing import tracable, trace
from gt4py_fvlo.tracing.tracerir import *
from gt4py_fvlo.model import Field, fmap, apply_stencil, map_, if_

domain = UnitRange(0, 5)*UnitRange(0, 5)
input = apply_stencil(lambda i, j: i, domain)

field = Field(domain[1:-1, 1:-1], map_(lambda i, j: input[i+1, j+1], domain[1:-1, 1:-1]))
assert field[1, 1] == input[2, 2] # e.g. field[i, j] == input[i+1, j+1]

# Field(domain, map_(lambda i, j: input[i+1, j+1], domain))[i, j] == input[i+1, j+1]

@tracable
def shift(input):
    def stencil(i, j):
        return input[i+1, j+1]
    return apply_stencil(stencil, input.domain) # input.domain[1:-1, 1:-1] not supported during tracing yet

@tracable
def identity(input):
    def stencil(i, j):
        return input[i, j]
    return apply_stencil(stencil, input.domain)

@tracable
def shift_with_identity(input):
    return identity(shift(input))

#trc = trace(laplap, (Symbol(name="INPUT_FIELD", type_=Field),))
trc = trace(shift_with_identity, (Symbol(name="INPUT_FIELD", type_=Field),))

from eve import NodeTranslator
import gt4py_fvlo.tracing.tracing as tracing
from gt4py_fvlo.utils.uid import uid
from gt4py_fvlo.tracing.pass_helper.pass_manager import PassManager


class SymbolicEvalStencil(NodeTranslator):
    def is_applicable(self, node, *, symtable):
        return isinstance(node, Call) and isinstance(node.func, tracing.PyExternalFunction) and node.func.func == map_ \
               and not isinstance(node.args[0], Lambda)

    def transform(self, node: Call, *, symtable):
        args = node.args
        assert not node.kwargs

        idx_symbs = tuple(Symbol(name=f"{dim}_{uid(dim)}", type_=int) for dim in ["I", "J"])
        pos_stencil_expr = Lambda(
            args=idx_symbs,
            expr=OpaqueCall(func=args[0], args=idx_symbs, kwargs={}))

        return tracing.Call(node.func, args=(pos_stencil_expr, *node.args[1:]), kwargs={})


from gt4py_fvlo.tracing.passes.constant_folding import ConstantFold
from gt4py_fvlo.tracing.passes.datamodel import DataModelConstructionResolver, DataModelMethodResolution, \
    DataModelGetAttrResolver, DataModelCallOperatorResolution, DataModelExternalGetAttrInliner
from gt4py_fvlo.tracing.passes.opaque_call_resolution import OpaqueCallResolution1, OpaqueCallResolution2
from gt4py_fvlo.tracing.passes.fix_call_type import FixCallType
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
    SymbolicEvalStencil(),
    OpaqueCallResolution1(),
    OpaqueCallResolution2(),
    FixCallType(),
    TupleGetItemResolver(),
    RemoveConstantRefs(),
    TracableFunctionResolver()
])

trc2 = pass_manager.visit(trc.expr, symtable={})

trc3 = ConstantFold().visit(trc2, symtable={})

trc4 = PassManager([RemoveUnusedSymbols()]).visit(trc3, symtable={})

trc5 = SingleUseInliner.apply(trc4)

trc6 = PassManager([DataModelExternalGetAttrInliner(), RemoveUnusedSymbols()]).visit(trc5, symtable={})

class NAryOpsTransformer(NodeTranslator):
    def visit_Call(self, node: Call):
        if isinstance(node.func, BuiltInFunction) and node.func.name == "__add__":
            first_arg, *rem_args = node.args
            if isinstance(first_arg, Call) and isinstance(first_arg.func, BuiltInFunction) and first_arg.func.name == "__add__":
                # todo: validate domains match
                return self.visit(Call(BuiltInFunction("__add__"), args=(*first_arg.args, *rem_args), kwargs={}))
        return self.generic_visit(node)

trc7 = NAryOpsTransformer().visit(trc6)

from gt4py_fvlo.tracing.pass_helper.scope_visitor import ScopeTranslator
from gt4py_fvlo.tracing.pass_helper.conversion import beta_reduction
from gt4py_fvlo.tracing.tracerir_utils import resolve_symbol

class InlineOnce(ScopeTranslator):
    def visit_Call(self, node: Call, symtable, **kwargs):
        if isinstance(node.func, PyExternalFunction) and node.func.func == Field.__getitem__:
            field = resolve_symbol(node.args[0], symtable)
            if isinstance(field, DataModelConstruction): # todo: no typing for datamodels yet...
                stencil = field.attrs["image"].args[0]
                return beta_reduction(stencil, node.args[1].elts, {}, closure_symbols=symtable)
        return self.generic_visit(node, symtable=symtable, **kwargs)

trc8 = InlineOnce.apply(trc7)
#trc8 = trc7

trc9 = PassManager([RemoveUnusedSymbols()]).visit(trc8, symtable={})

trc10 = PassManager([TracableFunctionResolver(custom_tracable_externals=[Field.__getitem__])]).visit(trc9, symtable={})


import gt4py_fvlo.gtl2ir.gtl2ir as gtl2ir
import re

class TranslateToGTL2IR(ScopeTranslator):
    def _translate_symbolname(self, node: Symbol):
        if node.type_ == Field:
            new_node = gtl2ir.SymbolName(name=node.name, type_=gtl2ir.SymbolRef("Field"))
        elif typing_inspect.is_tuple_type(node.type_) or node.type_ in [int, ProductSet]:
            new_node = gtl2ir.SymbolName(name=node.name, type_=node.type_)
        elif issubclass(node.type_, np.ndarray):
            new_node = gtl2ir.SymbolName(name=node.name, type_=gtl2ir.Array)
        else:
            raise ValueError()
        return new_node

    def visit_GenericIRNode(self, node: tracing.GenericIRNode, **kwargs):
        raise ValueError()

    def visit_DataModelConstruction(self, node: DataModelConstruction, **kwargs):
        assert node.type_ == Field
        return gtl2ir.Call(func=gtl2ir.Construct(), args=(gtl2ir.SymbolRef("Field"), self.visit(node.attrs["domain"], **kwargs), self.visit(node.attrs["image"], **kwargs)))

    def visit_Symbol(self, node: tracing.Symbol, **kwargs):
        return gtl2ir.SymbolRef(node.name)

    def visit_Let(self, node: Let, **kwargs):
        vars_ = tuple(gtl2ir.Var(name=self._translate_symbolname(var.name), value=self.visit(var.value, **kwargs)) for var in node.vars)
        return gtl2ir.Let(vars=vars_, expr=self.visit(node.expr, **kwargs))

    def visit_Lambda(self, node: Lambda, **kwargs):
        return gtl2ir.Lambda(args=tuple(self._translate_symbolname(arg) for arg in node.args), expr=self.visit(node.expr, **kwargs))

    def visit_Constant(self, node: Constant, **kwargs):
        return gtl2ir.Constant(val=node.val)

    def visit_Tuple_(self, node: Tuple_, **kwargs):
        return gtl2ir.Call(func=gtl2ir.ConstructTuple(), args=self.visit(node.elts, **kwargs))

    def visit_Call(self, node: Call, **kwargs):
        if isinstance(node.func, tracing.PyExternalFunction) and node.func.func == map_:
            return gtl2ir.Call(func=gtl2ir.SymbolRef("map"), args=self.visit(node.args, **kwargs))
        elif isinstance(node.func, BuiltInFunction):
            func_name = node.func.name

            magic_method_match = re.match("__(?!(__))(.*)__", node.func.name)
            if magic_method_match:
                func_name = magic_method_match.groups()[1]

            if func_name == "rmul":
                func_name = "mul"

            if func_name == "getattr":
                return gtl2ir.Call(func=gtl2ir.GetStructAttr(), args=self.visit(node.args, **kwargs))
            else:
                return gtl2ir.Call(func=gtl2ir.SymbolRef(func_name), args=self.visit(node.args, **kwargs))
        elif isinstance(node.func, Lambda):
            assert not node.kwargs
            return gtl2ir.Call(func=self.visit(node.func, **kwargs), args=self.visit(node.args, **kwargs))
        raise ValueError()

def translate_gtl2ir(node: Lambda):
    vars = []
    for func_name, declaration in gtl2ir.declarations.items():
        vars.append(gtl2ir.Var(name=gtl2ir.SymbolName(name=func_name, type_=None), value=declaration))
    vars.append(gtl2ir.Var(name=gtl2ir.SymbolName(name="Field", type_=None), value=gtl2ir.StructDecl(attr_names=("domain", "image"), attr_types=(gtl2ir.ProductSet, gtl2ir.Array))))
    return gtl2ir.Let(vars=tuple(vars), expr=TranslateToGTL2IR().apply(node))

gtl2ir_node = translate_gtl2ir(trc10)

bla=1+1