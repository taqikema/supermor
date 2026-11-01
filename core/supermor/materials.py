"""Assign cellwise VTI tensors through the native distributed DG0 maps.

A caller supplies a sampler with ``sample(points, markers, chunk_size)``;
this module contains no MR3D paths or experiment-specific material rules.
"""
import itertools

import numpy as np


def assign_native_vti(model, sampler, chunk_size=100_000):
    """Set diag(sigma_h, sigma_h, sigma_v) through actual component DOF maps.

    Only owned cell batches and the distributed local tensor vector are held.
    No whole-model float64 copy or UFL tensor expression is constructed.
    """
    import dolfin as df

    mesh = model.FS.mesh
    comm = mesh.mpi_comm()
    space = df.TensorFunctionSpace(mesh, "DG", 0)
    conductivity = df.Function(space)
    values = conductivity.vector().get_local()
    values.fill(0.)
    dofmap = space.dofmap()
    first, last = dofmap.ownership_range()
    element = space.ufl_element()
    components = [element.extract_subelement_component((i, i))[0] for i in range(3)]
    maps = [space.sub(index).dofmap() for index in components]
    markers = model.FS.DOM.domain_func.array()
    cells = (cell for cell in df.cells(mesh) if not cell.is_ghost())
    local_error = None
    try:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        while True:
            batch = list(itertools.islice(cells, chunk_size))
            if not batch:
                break
            positions = np.array([cell.midpoint().array() for cell in batch])
            materials = np.array([markers[cell.index()] for cell in batch])
            horizontal, vertical = sampler.sample(positions, materials, chunk_size)
            if (np.shape(horizontal) != (len(batch),) or np.shape(vertical) != (len(batch),)
                    or not np.isfinite(horizontal).all() or not np.isfinite(vertical).all()
                    or np.any(horizontal <= 0) or np.any(vertical <= 0)):
                raise ValueError("VTI sampler must return finite positive cell conductivities")
            for cell, sh, sv in zip(batch, horizontal, vertical):
                for component_map, value in zip(maps, (sh, sh, sv)):
                    dofs = component_map.cell_dofs(cell.index())
                    if len(dofs) != 1:
                        raise ValueError("VTI requires one DG0 DOF per component and cell")
                    global_dof = dofmap.local_to_global_index(int(dofs[0]))
                    if not first <= global_dof < last:
                        raise ValueError("Owned cell has a non-owned conductivity DOF")
                    values[global_dof-first] = value
    except Exception as error:
        local_error = f"rank {comm.rank}: {type(error).__name__}: {error}"
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        raise ValueError("Native VTI assignment failed: " + "; ".join(
            error for error in errors if error is not None))
    conductivity.vector().set_local(values)
    conductivity.vector().apply("insert")
    df.as_backend_type(conductivity.vector()).update_ghost_values()
    model.FS.S_dgt = space
    model.MP.sigma_func = conductivity
    model.MP.tensor_flag = model.MP.sigma_tensor_flag = True
    return conductivity
