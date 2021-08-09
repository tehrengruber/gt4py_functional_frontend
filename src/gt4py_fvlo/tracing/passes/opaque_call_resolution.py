from ..tracerir import *

class OpaqueCallResolution1:
    def is_applicable(self, node, *, symtable):
        return isinstance(node, OpaqueCall) and isinstance(node.func, Symbol)

    def transform(self, node: OpaqueCall, *, symtable):
        # todo: this is essentially an inlining operation that prevents some optimizations, e.g. moving
        #  constant computations outside of the function
        def call_node_type(expr):
            return Call if isinstance(expr, Function) else OpaqueCall

        assert node.func in symtable
        func = symtable[node.func]
        return call_node_type(func)(func=func, args=node.args, kwargs=node.kwargs)

class OpaqueCallResolution2:
    def is_applicable(self, node, *, symtable):
        return isinstance(node, OpaqueCall) and isinstance(node.func, Let)

    def transform(self, node: OpaqueCall, *, symtable):
        def call_node_type(expr):
            return Call if isinstance(expr, Function) else OpaqueCall

        return Let(vars=node.func.vars, expr=call_node_type(node.func.expr)(func=node.func.expr, args=node.args, kwargs=node.kwargs))