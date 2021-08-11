from __future__ import annotations

import collections.abc
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import numpy as np
import numpy.typing as npt
import pyvista as pv


# --- PyVista Datasets ---
def make_dataset_from_arrays(
    points: npt.ArrayLike,
    edges: npt.ArrayLike,
    cells: npt.ArrayLike,
    *,
    vertex_fields: Dict[str, npt.ArrayLike] = None,
    edge_fields: Dict[str, npt.ArrayLike] = None,
    cell_fields: Dict[str, npt.ArrayLike] = None,
) -> pv.MultiBlock:
    """Create a dataset with one block per primitive."""
    points = np.asarray(points)
    edges = np.asarray(edges)
    cells = np.asarray(cells)

    # Points should live a in 3D space for VTK
    if points.shape[1] < 3:
        prepared_points = np.concatenate(
            (
                points,
                np.zeros((points.shape[0], 3 - points.shape[1]), dtype=points.dtype),
            ),
            axis=1,
        )
    else:
        prepared_points = points

    # For pyvista, the format of each primitive in the list should be:
    #     (n_items, item_0, item_1, ... item_n)
    prepared_cells = np.concatenate(
        (np.asarray([cells.shape[-1]] * len(cells))[:, None], cells), axis=1
    )
    prepared_edges = np.concatenate(
        (np.asarray([edges.shape[-1]] * len(edges))[:, None], edges), axis=1
    )

    # Create a dataset with a different block per primitive
    blocks = pv.MultiBlock()
    blocks["cells"] = pv.PolyData(prepared_points, prepared_cells)
    cell_fields = cell_fields or {}
    for field_name in cell_fields:
        blocks["cells"].cell_arrays[field_name] = cell_fields[field_name]

    blocks["edges"] = pv.PolyData(prepared_points, lines=prepared_edges)
    edge_fields = edge_fields or {}
    for field_name in edge_fields:
        blocks["edges"].cell_arrays[field_name] = edge_fields[field_name]

    blocks["vertices"] = pv.PolyData(prepared_points)
    vertex_fields = vertex_fields or {}
    for field_name in vertex_fields:
        blocks["vertices"].point_arrays[field_name] = vertex_fields[field_name]

    return blocks


def make_dataset(
    mesh: meshio.Mesh,
    cell_type: Optional[str] = None,
    *,
    edge_fields: Optional[Union[List[str], Dict[str, npt.ArrayLike]]] = None,
    cell_fields: Optional[Union[List[str], Dict[str, npt.ArrayLike]]] = None,
) -> pv.MultiBlock:
    """Create a dataset with one block per primitive."""

    assert len(mesh.cells_dict) >= 1

    if not cell_type:
        for c in mesh.cells_dict:
            if c != "line":
                cell_type = c

    if isinstance(edge_fields, collections.abc.Sequence) and not isinstance(
        edge_fields, collections.abc.Mapping
    ):
        edge_fields = {name: mesh.cell_data_dict[name]["line"] for name in edge_fields}

    if isinstance(cell_fields, collections.abc.Sequence) and not isinstance(
        cell_fields, collections.abc.Mapping
    ):
        cell_fields = {name: mesh.cell_data_dict[name][cell_type] for name in cell_fields}

    return make_dataset_from_arrays(
        points=mesh.points,
        edges=mesh.cells_dict.get("line", []),
        cells=mesh.cells_dict.get(cell_type, []),
        vertex_fields=mesh.point_data,
        edge_fields=edge_fields,
        cell_fields=cell_fields,
    )


def merge_datasets(*, with_vertices: bool = False, **data_kwargs: pv.DataSet) -> pv.MultiBlock:
    """Create a dataset with a different block per primitive."""
    blocks = pv.MultiBlock()
    for key, dataset in data_kwargs.items():
        blocks[key] = dataset

    if with_vertices:
        vertices = {}
        vertex_fields = {}
        for key, dataset in data_kwargs.items():
            vertices[key] = dataset.points
            vertex_fields.update(dataset.point_arrays)

        all_vertices = list(vertices.values())
        assert all(
            np.all(all_vertices[i - 1] == all_vertices[i]) for i in range(1, len(all_vertices))
        )

        blocks["vertices"] = pv.PolyData(all_vertices[0])
        blocks["vertices"].point_arrays.update(vertex_fields)

    return blocks


# --- Plots ---
## Matplotlib
def mpl_tri_plot(mesh: meshio.Mesh, ax: Optional[plt.Axes] = None, **kwargs):
    """Plot a triangular mesh using 'matplotlib.tri'."""
    assert "triangle" in mesh.cells_dict
    triangles = mesh.cells_dict["triangle"]
    xy = mesh.points
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=triangles)
    kwargs.setdefault("marker", "o")
    p = ax or plt
    return p.triplot(triangulation, **kwargs)


## PyVista
def start_pyvista(
    theme: Literal["dark", "default", "document", "paraview"] = "paraview",
    jupyter: bool = False,
) -> None:
    if jupyter:
        pv.start_xvfb()
    pv.set_plot_theme(theme)


def plot_mesh(
    dataset: pv.DataSet,
    title="",
    *,
    plot: Optional[pv.Plotter] = None,
    font_size=10,
    **add_mesh_kwargs,
) -> pv.Plotter:
    if plot is None:
        plot = pv.Plotter()
    if title:
        plot.add_text(title, font_size=font_size)
    plot.add_mesh(dataset, **add_mesh_kwargs)
    return plot


def make_vertex_plot(
    dataset: pv.DataSet,
    title="Vertices",
    *,
    point_size: int = 15,
    plot: Optional[pv.Plotter] = None,
    font_size=10,
    **add_mesh_kwargs,
):
    return plot_mesh(
        dataset,
        title=title,
        plot=plot,
        point_size=point_size,
        render_points_as_spheres=True,
        **add_mesh_kwargs,
    )


def make_edge_plot(
    dataset: pv.DataSet,
    title="Edges",
    *,
    line_width: int = 10,
    plot: Optional[pv.Plotter] = None,
    font_size=10,
    **add_mesh_kwargs,
) -> pv.Plotter:
    return plot_mesh(dataset, title=title, plot=plot, line_width=line_width, **add_mesh_kwargs)


def make_cell_plot(
    dataset: pv.DataSet,
    title="Cells",
    *,
    plot: Optional[pv.Plotter] = None,
    font_size=10,
    **add_mesh_kwargs,
) -> pv.Plotter:
    return plot_mesh(dataset, title=title, plot=plot, **add_mesh_kwargs)


def make_grid_plot(
    dataset: pv.DataSet,
    line_width=5,
    point_size=10,
    font_size=10,
    **add_mesh_kwargs,
) -> pv.Plotter:
    pl = pv.Plotter(shape=(2, 2))

    pl.subplot(0, 0)
    make_vertex_plot(
        dataset["vertices"],
        plot=pl,
        font_size=font_size,
        point_size=point_size,
        **add_mesh_kwargs,
    )
    pl.subplot(1, 1)
    make_vertex_plot(
        dataset["vertices"], title="", plot=pl, point_size=point_size, **add_mesh_kwargs
    )

    pl.subplot(0, 1)
    make_edge_plot(
        dataset["edges"],
        plot=pl,
        font_size=font_size,
        line_width=line_width,
        **add_mesh_kwargs,
    )
    pl.subplot(1, 1)
    make_edge_plot(
        dataset["edges"],
        title="",
        plot=pl,
        line_width=line_width,
        **add_mesh_kwargs,
    )

    pl.subplot(1, 0)
    make_cell_plot(
        dataset["cells"],
        plot=pl,
        font_size=font_size,
        **add_mesh_kwargs,
    )
    pl.subplot(1, 1)
    make_cell_plot(
        dataset["cells"],
        title="",
        plot=pl,
        **add_mesh_kwargs,
    )

    pl.subplot(1, 1)
    pl.add_text("Grid", font_size=font_size)

    return pl
