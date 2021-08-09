from ..tracerir import *

class TupleGetItemResolver:
    def is_applicable(self, node, *, symtable):
        if not isinstance(node, Call) or node.func != BuiltInFunction("__getitem__"):
            return False
        self_arg = node.args[0]
        if isinstance(self_arg, Symbol):
            self_arg = symtable[self_arg]
        return isinstance(self_arg, Tuple_)

    def transform(self, node: Call, *, symtable):
        assert isinstance(node.args[1], Constant)
        self_arg = node.args[0]
        if isinstance(self_arg, Symbol):
            self_arg = symtable[self_arg]
        idx = node.args[1].val
        return self_arg.elts[idx]