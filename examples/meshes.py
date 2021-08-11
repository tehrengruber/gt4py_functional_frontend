#!/usr/bin/env python
import os.path

import matplotlib.pyplot as plt

import pyvista as pv


from gtutils import io as gtio
from gtutils import vis as gtvis


## ---- Script to export FVM demo data
# import atlas4py
# from ... import  nabla_setup

# setup = nabla_setup()
# mesh = setup.mesh
# input_field = setup.input_field
# vol_field = setup.vol_field
# S_MXX, S_MYY = setup.S_fields


# fvm_nabla_grid = gtio.make_mesh_from_atlas(mesh)
# fvm_nabla_grid.point_data["input"] = input_field
# fvm_nabla_grid.point_data["vol"] = vol_field
# fvm_nabla_grid.write("fvm_nabla_grid.vtk")

# fvm_nabla_triangles = gtio.make_mesh(
#     points=fvm_nabla_grid.points,
#     point_data=fvm_nabla_grid.point_data,
#     triangle_cells=fvm_nabla_grid.cells_dict["triangle"],
# )
# fvm_nabla_triangles.write("fvm_nabla_triangles.vtk")

# fvm_nabla_edges = gtio.make_mesh(
#     points=fvm_nabla_grid.points,
#     point_data=fvm_nabla_grid.point_data,
#     line_cells=fvm_nabla_grid.cells_dict["line"],
#     line_data={"S_MXX": S_MXX, "S_MYY": S_MYY},
# )
# fvm_nabla_edges.write("fvm_nabla_edges.vtk")
## ----


# Mesh specification for meshio
points = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.0],
    [2.0, 1.0],
]
cells = dict(
    [
        ("triangle", [[0, 1, 2], [1, 3, 2]]),
        ("quad", [[1, 4, 5, 3]]),
    ]
)

m = gtio.make_mesh(
    points,
    point_data={"field_t": [0.3, -1.2, 0.5, 0.7, 0.0, -3.0]},
    triangle_cells=[[0, 1, 2], [1, 3, 2]],
    triangle_data={"field_a": [0.1, 0.2], "field_b": [-0.1, -0.2]},
    quad_cells=[[1, 4, 5, 3]],
    quad_data={"field_a": [1.1], "field_b": [-1.1]},
)
print(m)


# Matplotlib plots
gtvis.mpl_tri_plot(m)
plt.show()


# Pyvista dataset plots
gtvis.start_pyvista()  # Inside jupyter: gtvis.start_pyvista(jupyter=True)

ds = gtvis.make_dataset_from_arrays(
    points, edges=[[1, 4], [4, 5], [5, 3], [3, 1]], cells=cells["triangle"]
)
p = gtvis.make_grid_plot(ds)
p.show(cpos="xy")


fvm_mesh = gtio.read(f"{os.path.dirname(__file__)}/data/fvm_nabla_grid.vtk")
fvm_ds = pv.wrap(fvm_mesh)
p = gtvis.plot_mesh(fvm_ds)
p.show(cpos="xy")
