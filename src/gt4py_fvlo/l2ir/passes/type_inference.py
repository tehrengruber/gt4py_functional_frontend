from ..l2ir_def import *
from eve import NodeTranslator

class TypeInference(NodeTranslator):
    def visit_L2IRNode(self, node, *, symtable):
        raise ValueError()

    def visit_Let(self, node: Let, *, symtable):
        return self.visit(node.expr, symtable={**symtable, **{var.name.name: var.value for var in node.vars}})

    def visit_SymbolRef(self, node: SymbolRef, *, symtable):
        return self.visit(symtable[node.name], symtable=symtable)

    def visit_SymbolName(self, node: SymbolName, *, symtable):
        type_ = node.type_
        if isinstance(type_, SymbolRef):
            type_ = symtable[type_.name]
        return type_

    def visit_Lambda(self, node: Lambda, *, symtable):
        return Lambda

    def visit_Constant(self, node: Constant, *, symtable):
        return type(node.val)

    def visit_Call(self, node: Call, *, symtable):
        func = symtable[node.func.name] if isinstance(node.func, SymbolRef) else node.func
        arg_types = self.visit(node.args, symtable=symtable)
        if isinstance(func, Construct):
            raise NotImplementedError()
        elif isinstance(func, ConstructTuple):
            return Tuple[arg_types]
        elif isinstance(func, GetStructAttr):
            assert isinstance(node.args[1], Constant) and isinstance(node.args[1].val, str)
            return arg_types[0].get_attr_type(node.args[1].val)
        elif isinstance(func, BuiltInFunction):
            if not func.is_applicable(arg_types):
                raise TypeError()
            return func.return_type
        raise TypeError()