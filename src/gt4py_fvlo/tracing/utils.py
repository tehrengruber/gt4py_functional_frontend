def merge_symtable(symtable1, symtable2, *tail):
    assert len(symtable1.keys() & symtable2.keys()) == 0

    merged_symtable = {**symtable1, **symtable2}

    if len(tail) == 0:
        return merged_symtable

    return merge_symtable(merged_symtable, *tail)

def getclosurevars(func):
    # inspect.getclosurevars does not capture closure vars in nested functions...
    def _getclosurevars(code):
        closure_vars = {}

        # captured globals
        for name in code.co_names:
            #if name not in func.__globals__:
            #    raise ValueError(f"Closure var `{name}` is unbound")
            if name in func.__globals__:
                closure_vars[name] = func.__globals__[name]
        # capture non-locals
        if func.__closure__ is not None:
            for var, cell in zip(code.co_freevars, func.__closure__):
                closure_vars[var] = cell.cell_contents
        # capture nested closure vars
        for const in code.co_consts:
            if isinstance(const, type((lambda a: a).__code__)):
                closure_vars = {**closure_vars, **_getclosurevars(const)}
        return closure_vars
    return _getclosurevars(func.__code__)