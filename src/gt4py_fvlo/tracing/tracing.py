import hashlib
import inspect
import itertools
import textwrap
import ast
import os
import re
import attr
import copy

import typing_inspect

from .tracerir import *
from .builtins import *
from .utils import getclosurevars
from .pass_helper.materializer import Materializer
from .pass_helper.conversion import beta_reduction, alpha_conversion
from .tracerir_utils import symbolic_args_from_args, evaluate_py_func, let_reduction

class ClosureWrapper:
    def wrap(self, arg):
        if isinstance(arg, (int, str)):
            return Constant(arg)
        elif isinstance(arg, DataModel):
            return OpaqueCall(PyClosureVar(type(arg)), args=(),
                              kwargs={k: cls.wrap(v) for k, v in attr.asdict(arg, recurse=False).items()})
        elif isinstance(arg, tuple):
            return Tuple_(tuple(self.wrap(el) for el in arg))
        elif isinstance(arg, Callable) and not inspect.isclass(arg):
            return PyExternalFunction(arg)

        assert not isinstance(arg, Tracer)
        assert not isinstance(arg, GenericIRNode)

        return PyClosureVar(arg)

def wrap_closure_var(arg):
    return ClosureWrapper().wrap(arg)

def transform_to_dialect(arg):
    if isinstance(arg, PyZipIteratorElTracer):
        return PyZipIteratorEl(arg.expr)
    if isinstance(arg, Tracer):
        return arg.expr
    elif isinstance(arg, (int, str, float, bool)):
        return Constant(arg)
    elif isinstance(arg, Callable):
        return PyFunc(arg)
    elif isinstance(arg, tuple):
        return Tuple_(tuple(transform_to_dialect(el) for el in arg))
    elif isinstance(arg, types.GeneratorType):
        return PyGeneratorExpr(arg)

    raise ValueError()


class Tracer:
    expr: Any

    def __init__(self, expr):
        assert not isinstance(expr, Tracer)
        self.expr = expr

    def __mul__(self, *args):
        return type(self)(BuiltInFunction("__mul__"))(self, *args)

    def __rmul__(self, *args):
        return type(self)(BuiltInFunction("__rmul__"))(self, *args)

    def __add__(self, *args):
        return type(self)(BuiltInFunction("__add__"))(self, *args)

    def __sub__(self, *args):
        return type(self)(BuiltInFunction("__sub__"))(self, *args)

    def __rsub__(self, *args):
        return type(self)(BuiltInFunction("__rsub__"))(self, *args)

    def __neg__(self):
        return type(self)(BuiltInFunction("__neg__"))(self)

    def __lt__(self, other):
        return type(self)(BuiltInFunction("__lt__"))(self, other)

    def __gt__(self, other):
        return type(self)(BuiltInFunction("__gt__"))(self, other)

    def __le__(self, other):
        return type(self)(BuiltInFunction("__le__"))(self, other)

    def __ge__(self, other):
        return type(self)(BuiltInFunction("__ge__"))(self, other)

    def __eq__(self, other):
        return type(self)(BuiltInFunction("__eq__"))(self, other)

    def __ne__(self, other):
        return type(self)(BuiltInFunction("__ne__"))(self, other)

    def __getitem__(self, *args):
        return type(self)(BuiltInFunction("__getitem__"))(self, *args)

    def __iter__(self):
        return type(self)(BuiltInFunction("__iter__"))(self)

    def __next__(self):
        return type(self)(BuiltInFunction("__next__"))(self)

    def __contains__(self, arg):
        return type(self)(BuiltInFunction("__contains__"))(self, arg)

    def __getattr__(self, attr, *args):
        # do not allow magic methods
        assert not re.match("__(?!(__))(.*)__", attr)
        return type(self)(BuiltInFunction("__getattr__"))(self, attr, *args)

    def __call__(self, *args, **kwargs):
        def call_node_type(expr):
            t = Call
            if not isinstance(expr, Function):
                t = OpaqueCall
            return t

        parsed_args = tuple(transform_to_dialect(arg) for arg in args)
        parsed_kwargs = {k: transform_to_dialect(v) for k, v in kwargs.items()}

        # constant fold isinstance_ calls
        if isinstance(self.expr, PyExternalFunction) and self.expr.func == isinstance_:
            assert parsed_args[0].type_
            assert isinstance(parsed_args[1], PyClosureVar)
            return issubclass(parsed_args[0].type_, Materializer().visit(parsed_args[1]))

        # constant fold if_ calls
        if isinstance(self.expr, PyExternalFunction) and self.expr.func == if_:
            if isinstance(parsed_args[0], Constant):
                assert all(isinstance(parsed_arg, PyFunc) for parsed_arg in parsed_args[1:])
                return parsed_args[1].func() if parsed_args[0].val else parsed_args[2].func()
            #raise ValueError()

        if isinstance(self.expr, PyExternalFunction) and self.expr.func == zip_:
            assert not kwargs
            return Tracer(PyZip(parsed_args))

        if isinstance(self.expr, PyExternalFunction) and self.expr.func == tuple_:
            assert len(parsed_args)==1
            arg = parsed_args[0]
            if isinstance(arg, PyGeneratorExpr):
                result = transform_to_dialect(tuple(arg.generator))
                assert isinstance(result, Tuple_) and len(result.elts)==1

                current_idx_symb = Symbol(name=f"zip_idx_{uid('zip_idx')}", type_=int)

                zip_args = []
                body = Rezip().visit(result.elts[0], current_idx_symb=current_idx_symb, zip_args=zip_args)

                tuple_len=None
                for zip_arg in zip_args:
                    if isinstance(zip_arg, Tuple_):
                        tuple_len = len(zip_arg.elts)
                    elif isinstance(zip_arg, Symbol):
                        tt = zip_arg.type_
                        if typing_inspect.is_tuple_type(tt):
                            # todo: check for elipsis
                            tuple_len = len(typing_inspect.get_args(tt))
                assert tuple_len

                tuple_expr = Tuple_(elts=tuple(alpha_conversion(body, {current_idx_symb: Constant(idx)}, allow_unresolved=True) for idx in range(0, tuple_len)))
                return Tracer(tuple_expr)
            raise ValueError()

        if isinstance(self.expr, BuiltInFunction) and self.expr.name=="__iter__":
            if isinstance(parsed_args[0], PyZip):
                return PyZipIter(Tracer(iterable) for iterable in parsed_args[0].iterables)
            elif isinstance(parsed_args[0], Tuple_):
                assert len(parsed_args)==1
                return tuple(Tracer(el_expr) for el_expr in parsed_args[0].elts).__iter__()
            elif isinstance(parsed_args[0], Symbol) and issubclass(typing_inspect.get_origin(parsed_args[0].type_), DataModel):
                # generic datamodel with static size
                assert issubclass(parsed_args[0].type_.__origin__.__len__.__self__, DataModel)
                return tuple(Tracer(Call(func=BuiltInFunction("__getitem__"), args=(parsed_args[0], Constant(0)), kwargs={})) for i in
                                    range(parsed_args[0].type_.__origin__.__len__())).__iter__()
            else:
                raise ValueError

        return Tracer(call_node_type(self.expr)(self.expr, parsed_args, parsed_kwargs))

class Rezip(eve.NodeTranslator):
    def visit_PyZipIteratorEl(self, node: PyZipIteratorEl, *, current_idx_symb, zip_args):
        zip_args.append(node.expr)
        return Call(func=BuiltInFunction("__getitem__"), args=(node.expr, current_idx_symb), kwargs={})

class PyZipIteratorElTracer(Tracer):
    pass

class PyZipIter(tuple):
    def __init__(self, elements):
        self._consumed = False

    def __next__(self):
        if self._consumed:
            raise StopIteration

        self._consumed+=1
        return tuple(PyZipIteratorElTracer(transform_to_dialect(el)) for el in self)

_tracable = set()
def tracable(func):
    _tracable.add(func)
    return func

def is_tracable(func):
    return func in _tracable


def trace(func, symbolic_args=None, symbolic_kwargs=None, closure_wrapper=wrap_closure_var):
    func_uid = f"{func.__name__}_{hashlib.md5(inspect.getsource(func).encode('utf-8')).hexdigest()}"

    # prepare function sources
    func_ast = ast.parse(textwrap.dedent(inspect.getsource(func)))
    func_ast = copy.deepcopy(func_ast)
    func_ast.body[0].decorator_list = []  # remove decorator
    args_ast = func_ast.body[0].args
    for arg in [*args_ast.args, *args_ast.kwonlyargs, *args_ast.posonlyargs]:
        arg.annotation = None
    func_source = ast.unparse(func_ast)

    closure_vars = getclosurevars(func)

    # special built-ins
    closure_vars["tuple"] = tuple_

    #
    # recompile function, with "rebound" closure vars
    #  assemble source
    source = "def trace(tracer_cls, __closure_vars):\n"
    source = source + textwrap.indent("".join(
        [f"""{closure_var}=tracer_cls(__closure_vars["{closure_var}"])\n""" for closure_var in closure_vars.keys()]),
                                      "    ")
    source = source + textwrap.indent(func_source, "    ") + "\n"
    if func.__name__ != "<lambda>":
        source = source + textwrap.indent(f"return {func.__name__}", "    ")
    else:
        source = source + textwrap.indent("raise RuntimeError('this should not happend')",  "    ")

    # write to file
    #fp = tempfile.NamedTemporaryFile(suffix=f"{func_uid}.py", dir=os.path.dirname(__file__), delete=False)
    dir = os.path.dirname(inspect.getfile(func))+f"/tracing_tmp"
    os.makedirs(dir, exist_ok=True)
    fp = open(dir+f"/{func_uid}.py", "wb+")
    fp.write(bytes(source, 'UTF-8'))
    fp.flush()

    import importlib.util
    spec = importlib.util.spec_from_file_location("__func_uid", fp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # evaluate function
    #  validate argspec is to our expectation
    argspec = inspect.getfullargspec(func)
    argspec_expectations = {#"varargs": None,
                            "varkw": None,
                            "defaults": None,
                            #"kwonlyargs": [],
                            "kwonlydefaults": None}
    for attr, val in argspec_expectations.items():
        assert getattr(argspec, attr) == val

    # eval
    def bla(var):
        if closure_wrapper(var) == None:
            bla=1+1
        return Tracer(closure_wrapper(var))
    trace_func = mod.trace(bla, closure_vars)
    if symbolic_args is None:
        assert NotImplementedError
        symbolic_args = tuple((Symbol(arg) for arg in argspec.args))
    if symbolic_kwargs is None:
        assert len(argspec.kwonlyargs) == 0
        symbolic_kwargs = {}
    expr = transform_to_dialect(trace_func(
        # positional args
        *(Tracer(Symbol(arg.name, arg.type_)) for arg in symbolic_args),
        # kwargs
        **{k: Tracer(Symbol(v.name, v.type_)) for k, v in symbolic_kwargs.items()}))

    lambda_expr = Lambda(symbolic_args, expr, kwargs=symbolic_kwargs)

    result = Tracer(lambda_expr)

    return result