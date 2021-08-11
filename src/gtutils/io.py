from __future__ import annotations

import collections.abc
import enum
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, TypeVar, Union

import meshio
from meshio import read, write
import numpy as np
import numpy.typing as npt

try:
    import atlas4py
except ImportError:
    pass

T = TypeVar("T")
OneOrMore = Union[T, Iterable[T]]


# --- Meshes ---
class CellKind(enum.Enum):
    # Common cells
    vertex = 1
    line = 2
    triangle = 3
    quad = 4
    tetra = 4
    pyramid = 5
    wedge = 6
    hexaedron = 8
    # Specializations
    line3 = 3
    triangle6 = 6
    triangle7 = 7
    quad8 = 8
    quad9 = 9
    tetra10 = 10
    hexaedron20 = 20
    hexaedron24 = 24
    hexaedron27 = 27
    wedge12 = 12
    wedge15 = 15
    pyramid13 = 13
    pyramid14 = 14


def make_mesh(
    points: npt.ArrayLike,
    point_data: Optional[Dict[str, npt.ArrayLike]] = None,
    **cells_and_data_kwargs: Union[npt.ArrayLike, Dict[str, npt.ArrayLike]],
) -> meshio.Mesh:
    """Create a meshio.Mesh from collections of points and cells."""
    cells_dict = {}
    cells_data_dict = {}
    point_data = point_data or {}

    for arg, value in cells_and_data_kwargs.items():
        if arg.endswith("_cells"):
            # Cells definition
            kind = arg[:-6]
            cells_dict[kind] = value
        elif arg.endswith("_data"):
            # Cells data
            assert isinstance(value, dict)
            if not cells_data_dict:
                cells_data_dict = {key: {} for key in value}
            else:
                if not value.keys() == cells_data_dict.keys():
                    raise ValueError("All cell kinds should define the same extra data arrays.")

            kind = arg[:-5]
            for name, data in value.items():
                cells_data_dict[name][kind] = data
        else:
            # Data field for all cells
            assert isinstance(value, dict)
            cells_data_dict[arg] = value

    # Transform the cell fields dict to the right format for meshio
    cells_data = {}
    for name, data in cells_data_dict.items():
        if not cells_data_dict[name].keys() >= cells_dict.keys():
            raise ValueError(f"Fields '{name}' should be defined for all mesh cell kinds.")
        cells_data[name] = [cells_data_dict[name][kind] for kind in cells_dict]

    return meshio.Mesh(points, cells_dict, point_data=point_data, cell_data=cells_data)


def make_mesh_from_atlas(
    mesh: atlas4py.Mesh,
    *,
    edge_blocks: Optional[OneOrMore[int]] = None,
    cell_blocks: Optional[OneOrMore[int]] = None,
) -> meshio.Mesh:
    points = np.asarray(mesh.nodes.lonlat)
    assert len(points) == mesh.nodes.size

    if edge_blocks is None:
        edge_blocks = range(mesh.edges.node_connectivity.blocks)
    elif isinstance(edge_blocks, int):
        edge_blocks = [edge_blocks]
    edges = []
    for block_id in edge_blocks:
        block = mesh.edges.node_connectivity.block(block_id)
        edges.extend([[block[i, j] for j in range(block.cols)] for i in range(block.rows)])
    edges = np.asarray(edges, dtype=int)

    if cell_blocks is None:
        cell_blocks = range(mesh.cells.node_connectivity.blocks)
    elif isinstance(cell_blocks, int):
        cell_blocks = [cell_blocks]
    cells = []
    for block_id in cell_blocks:
        block = mesh.cells.node_connectivity.block(block_id)
        cells.extend([[block[i, j] for j in range(block.cols)] for i in range(block.rows)])
    cells = np.asarray(cells, dtype=int)

    return make_mesh(points=points, line_cells=edges, triangle_cells=cells)


def combine_meshes(*meshes: meshio.Mesh) -> meshio.Mesh:
    """Combine data from different meshes with the same geometry in a new mesh."""

    assert all(np.all(meshes[i - 1].points == meshes[i].points) for i in range(1, len(meshes)))

    point_data = {}
    kwargs = {}
    for mesh in meshes:
        point_data.update(mesh.point_data)
        kwargs.update({f"{kind}_cells": data for kind, data in mesh.cells_dict.items()})
        for name, data in mesh.cell_data_dict.items():
            if name in kwargs:
                kwargs[name].update(data)
            else:
                kwargs[name] = data

    return make_mesh(meshes[0].points, point_data=point_data, **kwargs)
