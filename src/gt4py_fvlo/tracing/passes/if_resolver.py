from ..tracing import if_
from ..tracerir import *

class IfResolver:
    def is_applicable(self, node, *, symtable):
        return isinstance(node, Call) and node.func == PyExternalFunction(if_) and any(isinstance(arg, (PyFunc, PyExternalFunction)) for arg in node.args[1:])

    def transform(self, node: Call, *, symtable):
        def call_node_type(expr):
            return Call if isinstance(expr, Function) else OpaqueCall

        return Call(func=node.func, args=(node.args[0], *(call_node_type(branch)(func=branch, args=(), kwargs={}) for branch in node.args[1:])), kwargs={})