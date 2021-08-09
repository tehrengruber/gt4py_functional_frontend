from ..l2ir import *
from eve import NodeTranslator, NodeVisitor
from .type_inference import TypeInference
from .alpha_conversion import alpha_conversion

class CollectSymbolRefs(NodeVisitor):
    def visit_SymbolRef(self, node: SymbolRef, *, symbol_refs):
        if node.name not in symbol_refs:
            symbol_refs[node.name]=0
        symbol_refs[node.name]+=1

    @classmethod
    def apply(cls, node):
        symbol_refs = {}
        cls().visit(node, symbol_refs=symbol_refs)
        return symbol_refs

def peel_let(let: Let, symbols: List[SymbolName], *, transformer=None):
    vars = tuple(var for var in let.vars if var.name not in symbols)
    expr = let.expr if not transformer else transformer(let.expr)
    if len(vars) == 0:
        return expr
    return Let(vars=vars, expr=expr)

class ConstantFold(NodeTranslator):
    def visit_Lambda(self, node: Lambda, *, symtable):
        return self.generic_visit(node, symtable={**symtable, **{arg.name: arg for arg in node.args}})

    def visit_Call(self, node: Call, *, symtable):
        func = symtable[node.func.name] if isinstance(node.func, SymbolRef) else node.func
        if isinstance(func, Lambda):
            # remove unnecessary function declarations
            return self.visit(Let(vars=tuple(Var(arg, val) for arg, val in zip(func.args, node.args)), expr=func.expr), symtable=symtable)
        elif isinstance(func, PolymorphicFunction):
            # dispatch resolution
            args = self.visit(node.args, symtable=symtable)
            arg_types = TypeInference().visit(args, symtable=symtable)
            for overload_ref in func.overloads:
                overload = symtable[overload_ref.name]
                if overload.is_applicable(arg_types):
                    return self.visit(Call(func=overload_ref, args=args), symtable=symtable)

            raise TypeError()
        elif len(node.args) > 1 and isinstance(node.args[0], Call) and isinstance(node.args[0].func, ConstructTuple) and isinstance(node.args[1], Constant): # todo: create machinary for this
            return node.args[0].args[node.args[1].val]
        return self.generic_visit(node, symtable=symtable)

    def visit_Let(self, node: Let, *, symtable):
        # merge let statements
        if isinstance(node.expr, Let):
            # todo: just a draft untested
            new_vars = node.vars

            inner_let = node.expr
            inner_let_symbol_refs = CollectSymbolRefs().visit(inner_let)
            for inner_symbol, var in node.expr.inner_symbol:
                if inner_symbol not in inner_let_symbol_refs:
                    inner_let = peel_let(inner_let, inner_symbol)
                    new_vars.append(var)

            return self.visit(Let(vars=new_vars, expr=inner_let))

        # remove superfluous symbol name -> symbol ref vars
        rebound_vars = [var for var in node.vars if isinstance(var.value, SymbolRef)]
        if rebound_vars:
            return self.visit(peel_let(node, [var.name for var in rebound_vars],
                                       transformer=lambda expr: alpha_conversion(node.expr,
                                                                                 {var.name.name: var.value for var in
                                                                                  rebound_vars})), symtable=symtable)

        # remove vars that are only used once by inserting them at their place of usage
        #used_symbols = CollectSymbolRefs.apply(node.expr)
        # todo

        # visit vars first so that the symbol table for the expr is already constant folded
        vars_ = self.visit(node.vars, symtable=symtable)
        return self.generic_visit(Let(vars=vars_, expr=node.expr), symtable={**symtable, **{var.name.name: var.value for var in vars_}})