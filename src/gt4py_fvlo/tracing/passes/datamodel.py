import inspect

from ..tracerir import *
from ..tracerir_utils import resolve_symbol

class DataModelCallOperatorResolution():
    def is_applicable(self, node, *, symtable):
        return isinstance(node, OpaqueCall) and isinstance(node.func, DataModelConstruction)

    def transform(self, node: OpaqueCall, *, symtable):
        dm_constr: DataModelConstruction = node.func
        call_op = getattr(dm_constr.type_, "__call__")
        return Call(func=PyExternalFunction(func=call_op), args=(dm_constr, *node.args), kwargs={})

class DataModelConstructionResolver:
    def is_applicable(self, node, *, symtable):
        return isinstance(node, OpaqueCall) and isinstance(node.func, PyClosureVar) and issubclass(node.func.val, DataModel)

    def transform(self, node: OpaqueCall, *, symtable):
        dm_type = node.func.val

        # collect expressions for all attributes
        attrs = {}
        for i, arg in enumerate(node.args):
            attr_spec = dm_type.__attrs_attrs__[i]
            attrs[attr_spec.name] = arg
        for attr_name, arg in node.kwargs.items():
            attr_spec = getattr(dm_type.__attrs_attrs__, attr_name)
            assert attr_spec.name not in attrs
            attrs[attr_spec.name] = arg

        # generate let statement with a variable for each attribute
        #  we do this to allow constant folding getattr calls without increasing the algorithmic complexity
        vars_ = []
        var_refs = {}
        dm_uid = uid("datamodel")
        for attr_name, attr_expr in attrs.items():
            attr_symb = Symbol(f"{attr_name}_{dm_uid}_{uid(attr_name)}", type_=getattr(dm_type.__datamodel_fields__, attr_name).type)
            vars_.append(Var(name=attr_symb, value=attr_expr))
            var_refs[attr_name] = attr_symb

        dm_constr = DataModelConstruction(type_=dm_type, attrs=var_refs)

        return Let(vars=tuple(vars_), expr=dm_constr)

class DataModelGetAttrResolver:
    def is_applicable(self, node, *, symtable):
        if not isinstance(node, Call) or node.func != BuiltInFunction("__getattr__"):
            return False
        self_node = resolve_symbol(node.args[0], symtable)
        return isinstance(self_node, DataModelConstruction)

    def transform(self, node: Call, *, symtable):
        # todo: use symbol table to resolve node.args
        dm_constr: DataModelConstruction = resolve_symbol(node.args[0], symtable, expected_type=DataModelConstruction)
        attr_name = node.args[1]
        assert isinstance(attr_name, Constant) and isinstance(attr_name.val, str)
        attr_value = dm_constr.attrs[attr_name.val]
        assert isinstance(attr_value, Symbol) # never inline!
        return attr_value

class DataModelExternalGetAttrInliner:
    # inline getattr calls to external symbols (they are cheap)

    def is_applicable(self, node, *, symtable):
        if not isinstance(node, Symbol):
            return False
        refered_node = resolve_symbol(node, symtable)
        return isinstance(refered_node, Call) and isinstance(refered_node.func, BuiltInFunction) and refered_node.func.name == "__getattr__"

    def transform(self, node: Symbol, *, symtable):
        return resolve_symbol(node, symtable)

class DataModelMethodResolution:
    def is_applicable(self, node, *, symtable):
        if not isinstance(node, Call) or not isinstance(node.func, BuiltInFunction) or node.func.name == "__getattr__":
            return False
        self_arg = node.args[0]
        return (isinstance(self_arg, Symbol) and inspect.isclass(self_arg.type_) and issubclass(self_arg.type_, DataModel)) \
               or isinstance(self_arg, DataModelConstruction)

    def transform(self, node: Call, *, symtable):
        self_arg = node.args[0]
        func = getattr(self_arg.type_, node.func.name)
        return Call(func=PyExternalFunction(func=func), args=node.args, kwargs=node.kwargs)