import numpy as np
from eve.datamodels import DataModel
from iterator_frontend.cartesian_sets import Dimension, UnitRange
import iterator_frontend
from iterator_frontend.model import Shift, Field, Index, Position, Iterator, beautified_iterator, lifted_iterator

class CartesianDimMeta(type):
    def __add__(self, offset: int):
        if not isinstance(offset, int):
            raise NotImplementedError
        return Shift(self(), offset)

    def __sub__(self, other):
        return self+(-other)

class CartesianDim(Dimension, metaclass=CartesianDimMeta):
    pass

class I(CartesianDim):
    pass

class J(CartesianDim):
    pass

class K(CartesianDim):
    pass

class CartesianConnectivity(DataModel):
    dim: Dimension

    def __call__(self, position, offset):
        return Position(*(Index(idx.dim, idx.val+offset) if idx.dim == self.dim else idx for idx in position))

iterator_frontend.model.connectivities = {
    I: CartesianConnectivity(I()),
    J: CartesianConnectivity(J()),
    K: CartesianConnectivity(K())
}

domain_2d = UnitRange(I, 0, 10)*UnitRange(J, 0, 10)

field = Field(domain_2d, np.ones(domain_2d.shape))

@beautified_iterator
def laplacian_bi(inp: Iterator):
    return -4*inp()-inp(I+1)-inp(I-1)-inp(J+1)-inp(J-1)

@lifted_iterator
def laplacian_li(inp: Field):
    return -4*inp-inp(I+1)-inp(I-1)-inp(J+1)-inp(J-1)

# functional
out = laplacian_bi[domain_2d.shrink(1, 1)](field)

# inplace
laplacian_bi[domain_2d.shrink(1, 1)](field, out=out)

out2 = laplacian_li(field)[domain_2d.shrink(1, 1)]

i = I(0)

field[I(0), J(0)]

bla=1+1