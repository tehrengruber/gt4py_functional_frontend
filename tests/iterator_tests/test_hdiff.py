from iterator.builtins import *
from iterator.runtime import *
from iterator.embedded import np_as_located_field
import numpy as np
from .hdiff_reference import hdiff_reference

I = offset("I")
J = offset("J")

IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")


@fundef
def laplacian(inp):
    return -4.0 * deref(inp) + (
        deref(shift(I, 1)(inp))
        + deref(shift(I, -1)(inp))
        + deref(shift(J, 1)(inp))
        + deref(shift(J, -1)(inp))
    )


@fundef
def flux(d):
    def flux_impl(inp):
        lap = lift(laplacian)(inp)
        flux = deref(lap) - deref(shift(d, 1)(lap))
        return if_(flux * (deref(shift(d, 1)(inp)) - deref(inp)) > 0.0, 0.0, flux)

    return flux_impl


@fundef
def hdiff_sten(inp, coeff):
    flx = lift(flux(I))(inp)
    fly = lift(flux(J))(inp)
    return deref(inp) - (
        deref(coeff)
        * (deref(flx) - deref(shift(I, -1)(flx)) + deref(fly) - deref(shift(J, -1)(fly)))
    )


@fendef(offset_provider={"I": IDim, "J": JDim})
def hdiff(inp, coeff, out, x, y):
    closure(
        domain(named_range(IDim, 0, x), named_range(JDim, 0, y)),
        hdiff_sten,
        [out],
        [inp, coeff],
    )


hdiff(*([None] * 5), backend="lisp")
hdiff(*([None] * 5), backend="cpptoy")

def test_hdiff(hdiff_reference):
    inp, coeff, out = hdiff_reference
    shape = (out.shape[0], out.shape[1])

    inp_s = np_as_located_field(IDim, JDim, origin={IDim: 2, JDim: 2})(inp[:, :, 0])
    coeff_s = np_as_located_field(IDim, JDim)(coeff[:, :, 0])
    out_s = np_as_located_field(IDim, JDim)(np.zeros_like(coeff[:, :, 0]))

    # hdiff(inp_s, coeff_s, out_s, shape[0], shape[1])
    # hdiff(inp_s, coeff_s, out_s, shape[0], shape[1], backend="embedded")
    hdiff(inp_s, coeff_s, out_s, shape[0], shape[1], backend="double_roundtrip")

    assert np.allclose(out[:, :, 0], out_s)


from gt4py_fvlo.model import Field, fmap, located_field_as_fvlo_field, tracable, UnitRange, apply_stencil, ProductSet, if_

@tracable
def laplacian(inp):
    return -4 * inp(0, 0, 0) + (inp(1, 0, 0) + inp(-1, 0, 0) + inp(0, 1, 0) + inp(0, -1, 0))

def flux_x(grid, f, inp):
    @tracable
    def stencil(f, inp):
        flux = f(0, 0, 0) - f(1, 0, 0)
        return if_(flux * (inp(1, 0, 0) - inp(0, 0, 0)) > 0.0, lambda: 0.0, lambda: flux)

    return apply_stencil(stencil, grid.staggered_domain["I"], f, inp)

def flux_y(grid, f, inp):
    @tracable
    def stencil(f, inp):
        flux = f(0, 0, 0) - f(0, 1, 0)
        return if_(flux * (inp(0, 1, 0) - inp(0, 0, 0)) > 0.0, lambda: 0.0, lambda: flux)

    return apply_stencil(stencil, grid.staggered_domain["J"], f, inp)


def horizontal_diffusion(grid, inp, coeff):
    lap = fmap(laplacian, inp)

    flux_x_ = flux_x(grid, lap, inp)
    flux_y_ = flux_y(grid, lap, inp)

    @tracable
    def hdiff_stencil(inp, coeff, flx, fly):
        return inp(0, 0, 0) - (coeff(0, 0, 0) * (flx(0, 0, 0) - flx(-1, 0, 0) + fly(0, 0, 0) - fly(0, -1, 0)))

    return apply_stencil(hdiff_stencil, inp.domain, inp, coeff, flux_x_, flux_y_)

from types import SimpleNamespace

def test_hdiff_fvlo(hdiff_reference):
    inp, coeff, out = hdiff_reference
    shape = (out.shape[0], out.shape[1], out.shape[2])

    # construct simple grid
    vertex_domain = ProductSet.from_shape(shape)
    staggered_domain = {
        "I": vertex_domain.shrink((0, 1), 0, 0),
        "J": vertex_domain.shrink(0, (0, 1), 0)}

    grid = SimpleNamespace(vertex_domain=vertex_domain, staggered_domain=staggered_domain)

    # construct input fields
    inp_fvlo = Field(vertex_domain.extend(2, 2, 0), inp).transparent_view(vertex_domain)
    coeff_fvlo = Field(vertex_domain, coeff)

    out_fvlo = horizontal_diffusion(grid, inp_fvlo, coeff_fvlo)

    assert np.allclose(out, out_fvlo.image)
