import numpy as np

from gt4py_fvlo.utils.index_space import UnitRange, ProductSet, CartesianSet, intersect, union
from gt4py_fvlo.model import Field, fmap, apply_stencil, map_, if_

v2v_arr = np.array(
    [
        [1, 3, 2, 6],
        [2, 3, 0, 7],
        [0, 5, 1, 8],
        [4, 6, 5, 0],
        [5, 7, 3, 1],
        [3, 8, 4, 2],
        [7, 0, 8, 3],
        [8, 1, 6, 4],
        [6, 2, 7, 5],
    ]
)

e2v_arr = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 0],
        [3, 4],
        [4, 5],
        [5, 3],
        [6, 7],
        [7, 8],
        [8, 6],
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [6, 0],
        [7, 1],
        [8, 2],
    ]
)

v2e_arr = np.array(
    [
        [0, 15, 2, 9],  # 0
        [1, 16, 0, 10],
        [2, 17, 1, 11],
        [3, 9, 5, 12],  # 3
        [4, 10, 3, 13],
        [5, 11, 4, 14],
        [6, 12, 8, 15],  # 6
        [7, 13, 6, 16],
        [8, 14, 7, 17],
    ]
)

v2v = Field(UnitRange(0, v2v_arr.shape[0])*UnitRange(0, v2v_arr.shape[1]), v2v_arr)
e2v = Field(UnitRange(0, e2v_arr.shape[0])*UnitRange(0, e2v_arr.shape[1]), e2v_arr)
v2e = Field(UnitRange(0, v2e_arr.shape[0])*UnitRange(0, v2e_arr.shape[1]), v2e_arr)

def stencil(v):
    return sum(e for e in v2e[v, :])

blabs = apply_stencil(stencil, ProductSet(UnitRange(0, v2e_arr.shape[0])))

blub=stencil(0)

bla=1+1