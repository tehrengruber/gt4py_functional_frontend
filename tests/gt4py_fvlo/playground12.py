import ast
import operator
from copy import copy
import types
from types import LambdaType
from typing import List, Callable, Tuple, Any, Union, Literal
import inspect
import textwrap
import tempfile
import eve.datamodels as datamodels
import typing_inspect
from eve.datamodels import DataModel, field
from gt4py_fvlo.utils.index_space import UnitRange, ProductSet, CartesianSet, intersect, union
import gt4py_fvlo
import numpy as np

from gt4py_fvlo.tracing.tracing import trace, tracable, isinstance_, if_, tuple_, zip_, OpaqueCall, PyClosureVar, PyExternalFunction, DataModelConstruction

from gt4py_fvlo.generic import Lambda, BuiltInFunction, Constant, Call, Symbol, Tuple_, GenericIRNode, Function, Let, Var

class Stencil(DataModel):
    impl: Callable
    extent: Tuple[Union[int, Tuple[int, int]], ...]

    @tracable
    def __call__(self, *args):
        return self.impl(*args)


class AbstractField:
    pass


class LazyMap(DataModel):
    func: Callable
    domain: ProductSet

    def __getitem__(self, memory_idx):
        origin = self.domain[0, 0]
        image_idx = tuple(image_idx + o for image_idx, o in zip(memory_idx, origin))
        assert image_idx in domain
        return self.func(image_idx)

    def materialize(self):
        return np.array([self.func(idx) for idx in self.domain]).reshape(self.domain.shape)

#image_type = LazyMap
#def _map(stencil, domain):
#    return LazyMap(stencil, domain)

image_type = np.ndarray
def map_(stencil, domain):
    return np.array([stencil(*idx) for idx in domain]).reshape(domain.shape)

#getitem(map(lambda i, j: getitem(field, (i, j)), domain), (i, j)) = field(i, j)

class Field(DataModel, AbstractField):
    domain: ProductSet
    image: image_type

    #@tracable
    def __getitem__(self, image_idx):
        assert image_idx in self.domain
        origin = self.domain[0, 0]
        memory_idx = tuple(idx-o for idx, o in zip_(image_idx, origin))
        return self.image[memory_idx]

    def view(self, domain):
        assert domain.issubset(self.domain)
        return apply_stencil(lambda *idx: self[idx], domain)

    def transparent_view(self, domain):
        if domain == self.domain:
            return self
        return TransparentView(domain, self)


class TransparentView(DataModel, AbstractField):
    domain: ProductSet
    field: Field

    def __getitem__(self, position):
        assert position in self.field.domain
        return self.field[position]

@tracable
def new_accessor(field, position):
    def accessor(*shift):
        gen = (idx+s for idx, s in zip_(position, shift))
        idx = tuple_(gen)
        return field[idx]
    return accessor

def extent_analysis(wrapped_stencil, fields):
    import gt4py_fvlo.tracing.tracing as tracing
    from gt4py_fvlo.tracing.tracerir_utils import let_reduction
    from gt4py_fvlo.utils.uid import uid

    closure_fields = {}

    class ClosureWrapper(tracing.ClosureWrapper):
        @classmethod
        def wrap(cls, arg):
            if isinstance(arg, AbstractField):
                sym = Symbol(f"symbolic_field_{uid('symbolic_field_')}", type_=type(arg))
                closure_fields[sym] = arg
                return sym
            return super(ClosureWrapper, cls).wrap(arg)

    symbolic_stencil = trace(wrapped_stencil, tuple(Symbol(name=dim, type_=int) for dim in ("I", "J")), closure_wrapper=ClosureWrapper.wrap).expr

    symtable = {sym: sym for sym in closure_fields.keys()}

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
        TracableFunctionResolver(closure_wrapper=ClosureWrapper.wrap)
    ])

    trc2 = pass_manager.visit(symbolic_stencil, symtable=symtable)

    trc3 = ConstantFold().visit(trc2, symtable=symtable)

    trc4 = PassManager([RemoveUnusedSymbols()]).visit(trc3, symtable=symtable)

    from gt4py_fvlo.tracing.passes.global_symbol_collision_resolver import GlobalSymbolCollisionCollector, GlobalSymbolCollisionResolver
    collisions = GlobalSymbolCollisionCollector.apply(trc4)
    assert not collisions
    #trc5 = GlobalSymbolCollisionResolver.apply(trc4, symtable=symtable)

    trc6 = SingleUseInliner.apply(trc4, symtable=symtable)

    trc7 = PassManager([DataModelExternalGetAttrInliner(), RemoveUnusedSymbols()]).visit(trc6, symtable=symtable)

    from gt4py_fvlo.tracing.pass_helper.scope_visitor import ScopeVisitor
    from gt4py_fvlo.tracing.pass_helper.conversion import beta_reduction, alpha_conversion
    from gt4py_fvlo.tracing.pass_helper.materializer import Materializer
    from gt4py_fvlo.tracing.tracerir_utils import resolve_symbol
    import gt4py_fvlo.tracing.tracing as tracing

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
        for cf, f in closure_fields.items():
            if id(f) == id(field) and cf in extents_per_symbol:
                accesses.extend(extents_per_symbol[cf])

        extents.append(set(accesses))

    return extents

@tracable
def apply_stencil(stencil: "Callable", domain, *fields):
    if len(fields) > 0:
        from gt4py_fvlo.tracing.tracing import _tracable
        assert stencil in _tracable
        assert all(isinstance_(field, AbstractField) for field in fields)
        @tracable
        def wrapped_stencil(*pos):
            return stencil(*(new_accessor(field, pos) for field in fields))

        from functools import reduce

        per_field_accesses = extent_analysis(wrapped_stencil, fields)

        per_field_valid_domains = []
        for field, accesses in zip(fields, per_field_accesses):
            accessable_field_domain = field.field.domain if isinstance(field, TransparentView) else field.domain
            per_field_valid_domains.append(intersect(accessable_field_domain, *(accessable_field_domain.translate(*map(operator.neg, access)) for access in accesses)))

        valid_domain = intersect(*per_field_valid_domains)
        domain = intersect(*(field.domain for field in fields))
        if not domain.issubset(valid_domain):
            raise ValueError("Not enough halo lines.")

        return apply_stencil(wrapped_stencil, valid_domain).transparent_view(domain)
    return Field(domain, map_(stencil, domain))

@tracable
def fmap(stencil, field: "Field"):
    return apply_stencil(stencil, field.domain, field)

@tracable
def laplacian(field: "Field"):
    def stencil(f):
        return -4 * f(0, 0) + f(-1, 0) + f(1, 0) + f(0, -1) + f(0, 1)

    return fmap(stencil, field)

@tracable
def laplap(field):
    lap = laplacian(field)
    laplap = laplacian(lap)
    return laplap

#
# Example 1 - zero origin
#
domain = UnitRange(0, 5)*UnitRange(0, 5)

i_image = map_(lambda i, j: i, domain)
j_image = map_(lambda i, j: j, domain)

assert (i_image[0, 0], i_image[0, 0]) == (0, 0)
assert (j_image[4, 4], j_image[4, 4]) == (4, 4)

i_field = Field(domain, i_image)
j_field = Field(domain, j_image)

assert (i_field[0, 0], j_field[0, 0]) == (0, 0)
assert (i_field[domain[-1, -1]], j_field[domain[-1, -1]]) == (4, 4)

#
# Example 2 - non-zero origin
#
domain = UnitRange(1, 6)*UnitRange(1, 6) # equal to domain.translate(1, 1)

i_image = map_(lambda i, j: i, domain)
j_image = map_(lambda i, j: j, domain)

assert (i_image[0, 0], i_image[0, 0]) == (1, 1)
assert (j_image[4, 4], j_image[4, 4]) == (5, 5)

i_field = Field(domain, i_image)
j_field = Field(domain, j_image)

assert (i_field[1, 1], i_field[1, 1]) == (1, 1)
assert (i_field[5, 5], i_field[5, 5]) == (5, 5)

#
# Example 3 - apply_stencil (frontend feature)
#  instead of constructing the field manually from a domain and an array we can automate that part
domain = UnitRange(0, 5)*UnitRange(0, 5)

# manual construction
one_field = Field(domain, map_(lambda i, j: 1, domain))

# automatic one using apply_stencil
one_field = apply_stencil(lambda i, j: 1, domain)

#
# Example - virtual assignment (frontend-feature)
def virtual_assign(field, domain, stencil):
    return apply_stencil(lambda i, j: stencil(i, j) if (i, j) in domain else field[i, j], field.domain)

zero_field = apply_stencil(lambda i, j: 0, domain)
#  stateful pseudo-code: zero_field[domain] = map(lambda i, j: 1, domain)
modified_field = virtual_assign(zero_field, domain[1:-1, 1:-1], lambda i, j: 1)
# conclusion: effort to translate from stateful to functional not higher than from dusk to gt4py

# question: what is the effort to remove the conditional inside the "loop"?
# answer: the same effort you would have in a stateful inline pass, but now it's something seperate that
#  can be debugged in isolation
#  (@tehrengruber see notes for algorithm)

#
# Example 3 - laplacian of f(x, y) = 1/6*x^3
#
f = lambda x, y: 1/6*x**3
lap_f = lambda x, y: x

domain = UnitRange(0, 5)*UnitRange(0, 5)
interior_domain = domain[1:-1, 1:-1]
input = apply_stencil(f, domain)

def lap_sten(i, j):
    return -4 * input[i, j] + input[i-1, j] + input[i+1, j] + input[i, j-1] + input[i, j+1]

# without bcs
#  (wanted to keep the example simple, so the domain get's smaller here)
lap_interior = apply_stencil(lap_sten, interior_domain)

# with bcs (use analytical solution on boundary)
lap = apply_stencil(lambda i, j: lap_sten(i, j) if (i, j) in interior_domain else lap_f(i, j), domain)

#
# Example - laplacian closer to the frontend
#
@tracable
def laplacian(input: "Field"):
    def stencil(i, j):
        return -4 * input[i, j] + input[i-1, j] + input[i+1, j] + input[i, j-1] + input[i, j+1]

    return apply_stencil(stencil, input.domain[1:-1, 1:-1])

@tracable
def laplap(field):
    lap = laplacian(field)
    laplap = laplacian(lap)
    return laplap

domain = UnitRange(0, 5)*UnitRange(0, 5)
input = apply_stencil(lambda i, j: 1/6*i**3, domain)

result = laplap(input)

#
# HDiff
#

shape = (5, 7)

#def hdiff_reference_impl():
rng = np.random.default_rng()
#inp = rng.normal(size=(shape[0] + 4, shape[1] + 4))
#coeff = rng.normal(size=shape)
inp = map_(lambda i, j: i*j*i*j, UnitRange(0, shape[0] + 4)*UnitRange(0, shape[1] + 4))
coeff = map_(lambda i, j: 2*i*j*i*j, UnitRange(0, shape[0])*UnitRange(0, shape[1]))

lap = 4 * inp[1:-1, 1:-1] - (
    inp[2:, 1:-1] + inp[:-2, 1:-1] + inp[1:-1, 2:] + inp[1:-1, :-2]
)
uflx = lap[1:, 1:-1] - lap[:-1, 1:-1]
flx = np.where(uflx * (inp[2:-1, 2:-2] - inp[1:-2, 2:-2]) > 0, 0, uflx)
ufly = lap[1:-1, 1:] - lap[1:-1, :-1]
fly = np.where(ufly * (inp[2:-2, 2:-1] - inp[2:-2, 1:-2]) > 0, 0, ufly)
out = inp[2:-2, 2:-2] - coeff * (
    flx[1:] - flx[:-1, :] + fly[:, 1:] - fly[:, :-1]
)

    #return inp, coeff, out


vertex_domain = UnitRange(0, shape[0]) * UnitRange(0, shape[1])
staggered_domain = {
    "I": UnitRange(0, shape[0] - 1) * UnitRange(0, shape[1]),
    "J": UnitRange(0, shape[0]) * UnitRange(0, shape[1] - 1)}

@tracable
def laplacian(inp):
    return -4 * inp(0, 0) + (inp(1, 0) + inp(-1, 0) + inp(0, 1) + inp(0, -1))

def flux_x(f, inp):
    @tracable
    def stencil(f, inp):
        flux = f(0, 0) - f(1, 0)
        return if_(flux * (inp(1, 0) - inp(0, 0)) > 0.0, lambda: 0.0, lambda: flux)

    return apply_stencil(stencil, staggered_domain["I"], f, inp)

def flux_y(f, inp):
    @tracable
    def stencil(f, inp):
        flux = f(0, 0) - f(0, 1)
        return if_(flux * (inp(0, 1) - inp(0, 0)) > 0.0, lambda: 0.0, lambda: flux)

    return apply_stencil(stencil, staggered_domain["J"], f, inp)


def horizontal_diffusion(inp, coeff):
    lap = fmap(laplacian, inp)

    flux_x_ = flux_x(lap, inp)
    flux_y_ = flux_y(lap, inp)

    @tracable
    def hdiff_stencil(inp, coeff, flx, fly):
        return inp(0, 0) - (coeff(0, 0) * (flx(0, 0) - flx(-1, 0) + fly(0, 0) - fly(0, -1)))

    return apply_stencil(hdiff_stencil, inp.domain, inp, coeff, flux_x_, flux_y_)

#inp, coeff, out = hdiff_reference_impl()

inp_field = Field(vertex_domain.extend(2, 2), inp).transparent_view(vertex_domain)
coeff_field = Field(vertex_domain, coeff)

# functional version
out_s = horizontal_diffusion(inp_field, coeff_field)

# stateful version
map_to(horizontal_diffusion, input=(inp, coeff), output=(out,))

#
# Example in-out fields
#  Proposition: we don't want to waste memory bandwidth on point-wise stencils

# simple example, identity
io_field = apply_stencil(lambda i, j: 1, domain)

io_field = apply_stencil(lambda i, j: io_field[i, j], domain)


# consider something stateful like this (fencil in antons model)
run_program(identity, input=(io_field,), output=(io_field,))
# central question now, how to avoid the unnecessary copy?
# answer: inline everything check if read to io_field with offset, if no avoid copy, if yes copy
#  done.


#
# Example - ease of inlining in a functional model
#
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

from gt4py_fvlo.tracing.passes.global_symbol_collision_resolver import GlobalSymbolCollisionCollector
collisions = GlobalSymbolCollisionCollector.apply(trc4)
assert not collisions

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

import gt4py_fvlo.gtl2ir.gtl2ir as gtl2ir
import re

class TranslateToGTL2IR(NodeTranslator):
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
        return gtl2ir.Call(func=gtl2ir.Construct(), args=(gtl2ir.SymbolRef("Field"), self.visit(node.attrs["domain"]), self.visit(node.attrs["image"])))

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
    return gtl2ir.Let(vars=tuple(vars), expr=TranslateToGTL2IR().visit(node))

gtl2ir_node = translate_gtl2ir(trc9)

bla=1+1