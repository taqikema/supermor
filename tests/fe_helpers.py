"""Tiny distributed FE fixtures and independent dense reference calculations.

Gathering is confined to these small test oracles, never production code.
"""
from types import SimpleNamespace

import dolfin as df
import numpy as np
from petsc4py import PETSc

from custEM.core import MOD
from supermor.methods import solve
from supermor.operators import sample_receivers


def small_model(n=1, sigma=None, frequencies=(.1, 1., 100.), *, degree=1):
    mesh = df.UnitCubeMesh(n, n, n)
    space = df.FunctionSpace(mesh, "N1curl", degree)
    nodal = df.FunctionSpace(mesh, "CG", degree)
    mixed = df.FunctionSpace(mesh, df.MixedElement([space.ufl_element()] * 2))
    markers = df.MeshFunction("size_t", mesh, 3, 0)
    for cell in df.cells(mesh):
        markers[cell] = int(cell.midpoint().z() > .5)
    model = MOD.__new__(MOD)
    model.FS = SimpleNamespace(mesh=mesh, V=space, S=nodal, M=mixed,
        U=df.Function(mixed), p=degree, test_mode=True,
        DOM=SimpleNamespace(domain_func=markers))
    model.FE = SimpleNamespace(quasi_static=True, ip=False, n_tx=1, bc=None,
        s_type="path", tx=[[[0., 0., 0.], [1., 0., 0.]]], grounding=True,
        a_tol=1e-8)
    model.MP = SimpleNamespace(approach="E_t", file_format="h5",
        frequencies=list(frequencies), currents=[[1.]] * len(frequencies),
        sigma_func=df.Constant(1.) if sigma is None else sigma,
        mu_inv_func=df.Constant(1.),
        rx=[[[.2, .3, .4], [.5, .5, .5], [.73, .41, .38]]])
    return model


def full_matrix(matrix):
    first, last = matrix.getOwnershipRange()
    local = matrix.getValues(np.arange(first, last, dtype=PETSc.IntType),
        np.arange(matrix.getSize()[1], dtype=PETSc.IntType))
    return np.vstack(matrix.comm.tompi4py().allgather(local))


def full_vector(vector):
    return np.concatenate(vector.comm.tompi4py().allgather(
        vector.getArray(readonly=True).copy()))


def independent_matrices(model):
    """Assemble new UFL forms, independently of MOR's assembly functions."""
    trial, test = df.TrialFunction(model.FS.V), df.TestFunction(model.FS.V)
    stiffness = df.assemble(df.inner(model.MP.mu_inv_func * df.curl(trial),
        df.curl(test)) * df.dx)
    mass = df.assemble(df.inner(model.MP.sigma_func * trial, test) * df.dx)
    return (full_matrix(df.as_backend_type(stiffness).mat()),
        full_matrix(df.as_backend_type(mass).mat()))


def reference_receivers(ops, field, omega):
    real, imag = ops["K"].createVecRight(), ops["K"].createVecRight()
    try:
        first, last = real.getOwnershipRange()
        real.getArray()[:] = field[first:last].real
        imag.getArray()[:] = field[first:last].imag
        return sample_receivers(ops, real, imag, omega)
    finally:
        real.destroy()
        imag.destroy()


def run_method(ops, frequencies, method, **options):
    rows = {}

    def emit(index, real, imag):
        rows[index] = dict(field=full_vector(real) + 1j * full_vector(imag),
            receivers=sample_receivers(ops, real, imag,
                2 * np.pi * frequencies[index]))

    stats = solve(ops, frequencies, method=method, emit=emit, **options)
    return stats, rows
