import itertools

from iterator.builtins import *
from iterator.embedded import np_as_located_field
from iterator.runtime import *
import numpy as np


@fundef
def ldif(d):
    return lambda inp: deref(shift(d, -1)(inp)) - deref(inp)


@fundef
def rdif(d):
    # return compose(ldif(d), shift(d, 1))
    return lambda inp: ldif(d)(shift(d, 1)(inp))


@fundef
def dif2(d):
    # return compose(ldif(d), lift(rdif(d)))
    return lambda inp: ldif(d)(lift(rdif(d))(inp))


i = offset("i")
j = offset("j")


@fundef
def lap(inp):
    return dif2(i)(inp) + dif2(j)(inp)


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@fendef
def fencil(x, y, z, output, input):
    closure(
        domain(named_range(IDim, 0, x), named_range(JDim, 0, y), named_range(KDim, 0, z)),
        lap,
        [output],
        [input],
    )


fencil(*([None] * 5), backend="lisp")
fencil(*([None] * 5), backend="cpptoy")

from gt4py_fvlo.model import Field, fmap, UnitRange
from gt4py_fvlo.tracing.tracing import tracable
def laplacian(field: "Field"):
    @tracable
    def stencil(f):
        return -4 * f(0, 0, 0) + f(-1, 0, 0) + f(1, 0, 0) + f(0, -1, 0) + f(0, 1, 0)

    return fmap(stencil, field)


def naive_lap(inp):
    shape = [inp.shape[0] - 2, inp.shape[1] - 2, inp.shape[2]]
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(0, shape[2]):
                out[i, j, k] = -4 * inp[i, j, k] + (
                    inp[i + 1, j, k] + inp[i - 1, j, k] + inp[i, j + 1, k] + inp[i, j - 1, k]
                )
    return out


def test_anton_toy():
    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp = np_as_located_field(IDim, JDim, KDim, origin={IDim: 1, JDim: 1, KDim: 0})(
        rng.normal(size=(shape[0] + 2, shape[1] + 2, shape[2])),
    )
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))
    ref = naive_lap(inp)

    fencil(
        shape[0],
        shape[1],
        shape[2],
        out,
        inp,
        backend="double_roundtrip",
        offset_provider={"i": IDim, "j": JDim},
    )

    inp_field = Field(UnitRange(-1, shape[0]+1)*UnitRange(-1, shape[1]+1)*UnitRange(0, shape[2]), inp.array())
    inp_field2 = inp_field.transparent_view(inp_field.domain[1:-1, 1:-1, :])
    out2 = laplacian(inp_field2)
    assert np.allclose(out2.image, ref)

    assert np.allclose(out, ref)