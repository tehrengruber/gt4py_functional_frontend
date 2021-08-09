from ..tracerir import *

class FixCallType():
    def is_applicable(self, node, *, symtable):
        return isinstance(node, OpaqueCall) and isinstance(node.func, Function)

    def transform(self, node: OpaqueCall, *, symtable): # todo: ugly
        def call_node_type(expr):
            return Call if isinstance(expr, Function) else OpaqueCall

        func = node.func
        return call_node_type(func)(func=func, args=node.args, kwargs=node.kwargs)