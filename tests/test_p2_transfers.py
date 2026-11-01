"""Independent DOLFIN interpolation oracles; also run with mpiexec -n 2."""
import importlib
import os

import dolfin as df
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc


def builder():
    try:
        module = importlib.import_module("supermor.mor_high_order")
    except ModuleNotFoundError:
        pytest.fail("P2 auxiliary transfer builder is missing")
    return module.build_p2_auxiliary


def spaces(scale=1., reverse=False):
    df.parameters["ghost_mode"] = "shared_facet"
    mesh = df.UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 1)
    x = mesh.coordinates()
    x[:] = scale * np.column_stack((x[:, 0] + .19*x[:, 1] + .07*x[:, 2],
                                    .11*x[:, 0] + 1.2*x[:, 1] + .13*x[:, 2],
                                    .17*x[:, 1] + .9*x[:, 2]))
    if reverse:
        # Reflect the coordinates while retaining DOLFIN's ordered topology.
        x[:, 0] *= -1.
    return mesh, df.FunctionSpace(mesh, "N1curl", 2), df.FunctionSpace(mesh, "CG", 2)


def backend(function):
    return df.as_backend_type(function.vector()).vec()


def relative_error(actual, expected):
    error = actual.copy()
    try:
        error.axpy(-1., expected)
        return error.norm() / max(expected.norm(), 1e-300)
    finally:
        error.destroy()


def random_scalar(S, phase):
    f = df.Function(S)
    first, last = S.dofmap().ownership_range()
    f.vector().set_local(np.sin(np.arange(first, last)*1.731 + phase))
    f.vector().apply("insert")
    f.vector().update_ghost_values()
    return f


class ScalarGradient(df.UserExpression):
    def __init__(self, scalar, **kwargs):
        self.scalar, self.space = scalar, scalar.function_space()
        super().__init__(**kwargs)

    def value_shape(self):
        return (3,)

    def eval_cell(self, values, x, ucell):
        cell = df.Cell(self.space.mesh(), ucell.index)
        coefficients = self.scalar.restrict(self.space.element(), cell)
        derivatives = self.space.element().evaluate_basis_derivatives_all(
            1, x, cell.get_vertex_coordinates(), cell.orientation()).reshape(-1, 3)
        values[:] = coefficients @ derivatives


class ScalarVector(df.UserExpression):
    def __init__(self, components, **kwargs):
        self.components = components
        super().__init__(**kwargs)

    def value_shape(self):
        return (3,)

    def eval_cell(self, values, x, ucell):
        space = self.components[0].function_space()
        cell = df.Cell(space.mesh(), ucell.index)
        basis = space.element().evaluate_basis_all(
            x, cell.get_vertex_coordinates(), cell.orientation()).reshape(-1)
        for c, component in enumerate(self.components):
            values[c] = component.restrict(space.element(), cell) @ basis


@pytest.mark.parametrize("scale,reverse", [(1., False), (1e-14, True), (1e14, False)])
def test_p2_transfers_match_independent_cellwise_interpolation(scale, reverse):
    mesh, V, S = spaces(scale, reverse)
    auxiliary = builder()(V, S)
    G, Pi = auxiliary["G"], auxiliary["Pi"]
    try:
        assert G.getSize() == (V.dim(), S.dim())
        assert Pi.getSize() == (V.dim(), 3*S.dim())
        assert G.getType() in ("seqaij", "mpiaij")
        assert Pi.getType() in ("seqaij", "mpiaij")
        assert G.getOwnershipRange() == V.dofmap().ownership_range()
        assert G.getOwnershipRangeColumn() == S.dofmap().ownership_range()
        first, last = S.dofmap().ownership_range()
        assert Pi.getOwnershipRangeColumn() == (3*first, 3*last)
        assert auxiliary["setup_seconds"] >= 0.
        assert auxiliary["nnz"]["G"] == G.getInfo()["nz_used"]
        assert auxiliary["nnz"]["Pi"] == Pi.getInfo()["nz_used"]
        assert G.getInfo()["mallocs"] == Pi.getInfo()["mallocs"] == 0

        phi = random_scalar(S, .4)
        expected = df.interpolate(ScalarGradient(phi, degree=1), V)
        result = G.createVecLeft()
        G.mult(backend(phi), result)
        assert relative_error(result, backend(expected)) < 1e-11
        result.destroy()

        components = [random_scalar(S, .3+c) for c in range(3)]
        expected = df.interpolate(ScalarVector(components, degree=2), V)
        nodal = Pi.createVecRight()
        nodal.array[:] = np.column_stack([f.vector().get_local()
                                         for f in components]).reshape(-1)
        result = Pi.createVecLeft()
        Pi.mult(nodal, result)
        assert relative_error(result, backend(expected)) < 1e-11
        nodal.destroy()
        result.destroy()
    finally:
        auxiliary["destroy"]()
        auxiliary["destroy"]()
    assert not G and not Pi


def test_p2_exact_sequence_and_polynomial_reproduction():
    mesh, V, S = spaces()
    auxiliary = builder()(V, S)
    G, Pi = auxiliary["G"], auxiliary["Pi"]
    scalar = df.interpolate(df.Expression("x[0]*x[0] + 2*x[0]*x[1] + 3*x[2]*x[2]", degree=2), S)
    gradient = df.interpolate(df.Expression(("2*x[0]+2*x[1]", "2*x[0]", "6*x[2]"), degree=1), V)
    ones, result = G.createVecRight(), G.createVecLeft()
    K = KG = None
    try:
        ones.set(1.)
        G.mult(ones, result)
        assert result.norm() / (G.norm()*ones.norm()) < 1e-11
        G.mult(backend(scalar), result)
        assert relative_error(result, backend(gradient)) < 1e-11
        u, v = df.TrialFunction(V), df.TestFunction(V)
        K = df.as_backend_type(df.assemble(df.inner(df.curl(u), df.curl(v))*df.dx)).mat().copy()
        KG = K.matMult(G)
        assert KG.norm() / (K.norm()*G.norm()) < 1e-11
        for c in range(3):
            nodal = Pi.createVecRight()
            nodal.array[:] = np.tile(np.eye(3)[c], nodal.getLocalSize()//3)
            expected = df.interpolate(df.Constant(tuple(float(j == c) for j in range(3))), V)
            Pi.mult(nodal, result)
            assert relative_error(result, backend(expected)) < 1e-11
            nodal.destroy()
    finally:
        ones.destroy()
        result.destroy()
        if KG is not None:
            KG.destroy()
        if K is not None:
            K.destroy()
        auxiliary["destroy"]()


def test_p2_transfer_on_hdf5_imported_ordered_mesh(mpi_tmp_path):
    mesh, unused_v, unused_s = spaces(reverse=True)
    comm = mesh.mpi_comm()
    filename = str(mpi_tmp_path / "p2_transfers.h5")
    try:
        with df.HDF5File(comm, filename, "w") as handle:
            handle.write(mesh, "/mesh")
        imported = df.Mesh(comm)
        with df.HDF5File(comm, filename, "r") as handle:
            handle.read(imported, "/mesh", False)
        assert imported.ordered()
        V, S = df.FunctionSpace(imported, "N1curl", 2), df.FunctionSpace(imported, "CG", 2)
        auxiliary = builder()(V, S)
        try:
            phi = random_scalar(S, .8)
            expected = df.interpolate(ScalarGradient(phi, degree=1), V)
            result = auxiliary["G"].createVecLeft()
            auxiliary["G"].mult(backend(phi), result)
            assert relative_error(result, backend(expected)) < 1e-11
            result.destroy()
        finally:
            auxiliary["destroy"]()
    finally:
        comm.barrier()
        if comm.rank == 0 and os.path.exists(filename):
            os.unlink(filename)


def test_p2_builder_rejects_other_degrees_and_mismatched_meshes():
    mesh, V, S = spaces()
    build = builder()
    with pytest.raises(ValueError, match="N1curl.*2|degree.*2"):
        build(df.FunctionSpace(mesh, "N1curl", 1), S)
    with pytest.raises(ValueError, match="CG.*2|degree.*2"):
        build(V, df.FunctionSpace(mesh, "CG", 1))
    other = df.UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1)
    with pytest.raises(ValueError, match="same mesh"):
        build(V, df.FunctionSpace(other, "CG", 2))


def test_p2_local_reference_failure_is_reported_collectively(monkeypatch):
    mesh, V, S = spaces()
    module = importlib.import_module("supermor.mor_high_order")
    original = module._reference_transfers

    def fail_one_rank():
        if mesh.mpi_comm().rank == 0:
            raise ValueError("deliberate local FIAT failure")
        return original()

    monkeypatch.setattr(module, "_reference_transfers", fail_one_rank)
    with pytest.raises(ValueError, match="rank 0: deliberate local FIAT failure"):
        module.build_p2_auxiliary(V, S)


def test_p2_local_cell_failure_is_reported_collectively(monkeypatch):
    mesh, V, S = spaces()
    module = importlib.import_module("supermor.mor_high_order")
    original = df.cells

    def fail_one_rank(*args, **kwargs):
        if mesh.mpi_comm().rank == 0:
            raise ValueError("deliberate local cell failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(df, "cells", fail_one_rank)
    with pytest.raises(RuntimeError, match="rank 0: deliberate local cell failure"):
        module.build_p2_auxiliary(V, S)


