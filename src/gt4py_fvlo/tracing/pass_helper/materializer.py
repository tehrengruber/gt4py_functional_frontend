from eve import NodeTranslator

from ..tracerir import *

class Materializer(NodeTranslator):
    def visit_GenericIRNode(self, node):
        raise NotMaterializable()

    def visit_PyExternalFunction(self, node):
        return node.func

    def visit_Constant(self, node: Constant):
        return node.val

    def visit_Tuple_(self, node: Tuple_):
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_PyClosureVar(self, node: PyClosureVar):
        return node.val

    def visit_Call(self, node):
        materialized_args = tuple(self.visit(arg) for arg in node.args)
        materialized_kwargs = {k: self.visit(arg) for k, arg in node.kwargs.items()}
        if isinstance(node.func, PyExternalFunction):
            return node.func.func(*materialized_args, **materialized_kwargs)
        elif isinstance(node.func, Lambda):
            wrapped_args = tuple(wrap_closure_var(marg) for marg in materialized_args)
            wrapped_kwargs = {kw: wrap_closure_var(marg) for kw, marg in materialized_kwargs.items()}
            try:
                evaluated = beta_reduction(node.func, wrapped_args, wrapped_kwargs)
            except ValueError: # todo specify
                raise NotMaterializable()
            return self.generic_visit(evaluated)
        elif isinstance(node.func, BuiltInFunction):
            inst, args = materialized_args[0], materialized_args[1:]
            return getattr(inst, node.func.name)(*args, **materialized_kwargs)
        raise ValueError()