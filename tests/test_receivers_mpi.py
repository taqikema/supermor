"""Analytic edge-field/curl sampling, including distributed shared faces."""
import dolfin as df
from mpi4py import MPI
import numpy as np
import pytest

from supermor.operators import assemble_mor, sample_receivers, store_field

from .fe_helpers import small_model


def test_receivers_have_one_owner_and_analytic_complex_e_and_h():
    model = small_model(n=2)
    ops = assemble_mor(model)
    field = df.interpolate(df.Expression(("-x[1]/2", "x[0]/2", "0"), degree=1),
        model.FS.V)
    real = df.as_backend_type(field.vector()).vec()
    imag = real.copy()
    imag.scale(2.)
    try:
        comm = model.FS.mesh.mpi_comm()
        owned = np.zeros(len(ops["rx"]), dtype=np.int64)
        for index, *_ in ops["receiver_cells"]:
            owned[index] += 1
        np.testing.assert_array_equal(comm.allreduce(owned, op=MPI.SUM), 1)
        values = sample_receivers(ops, real, imag, omega=2.)
        xyz = values["coordinates"]
        exact_e = np.column_stack((-xyz[:, 1] / 2, xyz[:, 0] / 2, np.zeros(len(xyz))))
        np.testing.assert_allclose(values["E"], (1. + 2j) * exact_e,
            rtol=0., atol=2e-13)
        np.testing.assert_allclose(values["H"],
            np.tile([0., 0., -1. + .5j], (len(xyz), 1)), rtol=0., atol=2e-13)
        for received in comm.allgather(values):
            np.testing.assert_array_equal(received["E"], values["E"])
            np.testing.assert_array_equal(received["H"], values["H"])
        store_field(model, real, imag)
        for component, expected in enumerate((real, imag)):
            extracted = df.Function(model.FS.V)
            df.FunctionAssigner(model.FS.V, model.FS.M.sub(component)).assign(
                extracted, model.FS.U[0].sub(component))
            difference = df.as_backend_type(extracted.vector()).vec().copy()
            try:
                difference.axpy(-1., expected)
                assert difference.norm() < 2e-13
            finally:
                difference.destroy()
    finally:
        imag.destroy()
        ops["destroy"]()


def test_empty_and_outside_receiver_sets():
    model = small_model()
    model.MP.rx = []
    ops = assemble_mor(model)
    try:
        values = sample_receivers(ops, ops["q"], ops["q"], omega=2.)
        assert values["E"].shape == values["H"].shape == (0, 3)
    finally:
        ops["destroy"]()
    model.MP.rx = [[[2., 2., 2.]]]
    with pytest.raises(ValueError, match="outside"):
        assemble_mor(model)
