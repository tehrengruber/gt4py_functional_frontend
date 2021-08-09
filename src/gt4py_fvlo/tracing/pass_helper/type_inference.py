import inspect

from eve import NodeTranslator

from ..tracerir import *

class TypeInference(NodeTranslator):
    def visit_GenericIRNode(self, node):
        return typing.Any

    def visit_DataModelConstruction(self, node: DataModelConstruction):
        return node.type_

    def visit_Function(self, node):
        return typing.Callable

    def visit_Let(self, node):
        return self.visit(node.expr)

    def visit_OpaqueCall(self, node):
        raise ValueError()
        if isinstance(node.func, PyClosureVar) and inspect.isclass(node.func.val):
            return node.func.val
        return typing.Any

    def visit_Symbol(self, node: Symbol):
        return node.type_

    def visit_Constant(self, node: Constant):
        return type(node.val)

    def visit_Tuple_(self, node: Tuple_):
        return Tuple[self.visit(node.elts)]

    def visit_Call(self, node: Call):
        if isinstance(node.func, Lambda):
            return self.visit(node.func.expr)
        if isinstance(node.func, BuiltInFunction) and node.func.name == "__getattr__" and issubclass(self.visit(node.args[0]), DataModel):
            dm_type = self.visit(node.args[0])
            attr_name = node.args[1]
            assert isinstance(attr_name, Constant)
            return getattr(dm_type.__datamodel_fields__, attr_name.val).type
        return typing.Any