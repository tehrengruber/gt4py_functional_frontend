from typing import List

from eve import NodeTranslator, NodeVisitor

from ..tracerir import *
from ..tracerir_utils import peel_let
from ..pass_helper.conversion import alpha_conversion
from ..pass_helper.collect_symbol_refs import CollectSymbolRefs

class ConstantFold(NodeTranslator):
    def visit_PyFunc(self, node, *, symtable):
        # PyFunc must not be visited as their scope should not be modified
        return self.generic_visit(node, symtable=symtable)

    def visit_Lambda(self, node: Lambda, *, symtable):
        return self.generic_visit(node, symtable={**symtable, **{arg: arg for arg in [*node.args, *node.kwargs.values()]}})

    def visit_Call(self, node: Call, *, symtable):
        func = symtable[node.func.name] if isinstance(node.func, Symbol) else node.func
        if isinstance(func, Lambda):
            # remove unnecessary function declarations
            return self.visit(Let(vars=tuple(Var(arg, val) for arg, val in zip(func.args, node.args)), expr=func.expr), symtable=symtable)
        return self.generic_visit(node, symtable=symtable)

    def visit_Let(self, node: Let, *, symtable):
        # remove superfluous symbol name -> symbol ref vars
        rebound_vars = [var for var in node.vars if isinstance(var.value, Symbol)]
        if rebound_vars:
            return self.visit(peel_let(node, [var.name for var in rebound_vars],
                                       transformer=lambda expr, partial_symtable: alpha_conversion(node.expr,
                                                                                 {var.name: var.value for var in
                                                                                  rebound_vars}, {**symtable, **partial_symtable})), symtable=symtable)
        # merge let inside let statements
        if isinstance(node.expr, Let):
            changes = 0
            outer_symbols = set([var.name for var in node.vars])
            new_vars = [*node.vars]

            inner_let = node.expr
            for inner_var in inner_let.vars:
                inner_symbol_refs = CollectSymbolRefs.apply(inner_var.value)
                collisions = outer_symbols & set(inner_symbol_refs.keys())
                if len(collisions) == 0:
                    new_vars.append(inner_var)
                    changes += 1
            inner_let = peel_let(inner_let, [new_var.name for new_var in new_vars])

            # if there are no changes just proceed with the generic visit
            if changes:
                return self.visit(Let(vars=tuple(new_vars), expr=inner_let), symtable=symtable)


        # remove vars that are only used once by inserting them at their place of usage
        # todo: this is unnecessarily costly. can be done once globally
        # todo: this might inline expressions into the map statement which we don't want
        #let_var_names = set([var.name for var in node.vars])
        #used_symbols = CollectSymbolRefs.apply(node.expr)
        #spliced_var_names = set([used_symbol for used_symbol, num_usages in used_symbols.items() if used_symbol in let_var_names and num_usages == 1])
        #spliced_symtable = {var.name: var.value for var in node.vars if var.name in spliced_var_names}
        #if len(spliced_var_names) > 0:
        #    return self.visit(peel_let(node, spliced_var_names,
        #                               transformer=lambda expr, partial_symtable: alpha_conversion(node.expr, spliced_symtable, {**symtable, **partial_symtable})),
        #                      symtable=symtable)

        # visit vars first so that the symbol table for the expr is already constant folded
        vars_ = self.visit(node.vars, symtable=symtable)
        return self.generic_visit(Let(vars=vars_, expr=node.expr), symtable={**symtable, **{var.name: var.value for var in vars_}})