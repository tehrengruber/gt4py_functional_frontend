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

from gt4py_fvlo.model import Field, fmap, located_field_as_fvlo_field, tracable
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

def fixture_anton_toy():
    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp = np_as_located_field(IDim, JDim, KDim, origin={IDim: 1, JDim: 1, KDim: 0})(
        rng.normal(size=(shape[0] + 2, shape[1] + 2, shape[2])),
    )
    ref = naive_lap(inp)

    return shape, inp, ref

def test_anton_toy():
    shape, inp, ref = fixture_anton_toy()

    # semantic-model
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))
    fencil(
        shape[0],
        shape[1],
        shape[2],
        out,
        inp,
        backend="double_roundtrip",
        offset_provider={"i": IDim, "j": JDim},
    )
    assert np.allclose(out, ref)

    # fvlo
    inp_fvlo_with_halo = located_field_as_fvlo_field(inp, origin=(1, 1, 0))
    inp_fvlo = inp_fvlo_with_halo.transparent_view(inp_fvlo_with_halo.domain[1:-1, 1:-1, :])

    out_fvlo = laplacian(inp_fvlo)
    assert np.allclose(out_fvlo.image, ref)
