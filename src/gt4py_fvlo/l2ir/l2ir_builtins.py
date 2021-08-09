@call_translator
def getattr(struct: Call[Construct, Tuple[TypeDecl]], attr_name: Constant[Str]):
    type_decl: TypeDecl = call.args[0]
    if not attr_name in type_decl.attr_names:
        raise TypeError("Type {type_decl.name} has no attribute {attr_name}")
    arg_pos = next(i for i, cand_attr_name in enumerate(type_decl.attr_names) if cand_attr_name==attr_name)
    return call.args[arg_pos]

@call_translater
def getitem(call: Call[Construct, Tuple[Tuple]], item: Constant[Int]):
    return call.args[item]