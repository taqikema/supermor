"""All four public methods against an independent tiny UFL/dense reference."""
import dolfin as df
import numpy as np
import pytest

from supermor.operators import assemble_mor

from .fe_helpers import (
    full_vector, independent_matrices, reference_receivers, run_method, small_model,
)


@pytest.mark.parametrize("method", ["DFOM", "IFOM", "RK", "SUPERMOR"])
@pytest.mark.parametrize("material", ["scalar", "vti"])
def test_public_four_methods_have_true_residual_and_correct_fields(method, material):
    sigma = df.Constant(1.) if material == "scalar" else df.Constant(np.diag([1., 1., .1]))
    model = small_model(sigma=sigma)
    frequencies = model.MP.frequencies
    ops = assemble_mor(model)
    try:
        stiffness, mass = independent_matrices(model)
        q = full_vector(ops["q"])
        stats, rows = run_method(ops, frequencies, method)
        assert stats["status"] == "converged", stats
        assert set(rows) == set(range(len(frequencies)))
        for index, frequency in enumerate(frequencies):
            s = 2j * np.pi * frequency
            rhs, matrix = s * q, stiffness + s * mass
            exact = np.linalg.solve(matrix, rhs)
            actual = rows[index]["field"]
            # Audit actual emitted fields, not a solver-reported statistic.
            residual = np.linalg.norm(rhs - matrix @ actual) / np.linalg.norm(rhs)
            assert residual <= 1.01e-6, (method, material, frequency, residual)
            difference = actual - exact
            error = np.sqrt(np.vdot(difference, mass @ difference).real /
                np.vdot(exact, mass @ exact).real)
            assert error < 1e-3, (method, material, frequency, error)
            reference = reference_receivers(ops, exact, s.imag)
            for field in ("E", "H"):
                expected = reference[field]
                received = rows[index]["receivers"][field]
                relative = np.linalg.norm(received - expected) / np.linalg.norm(expected)
                assert relative < 1e-3, (method, material, frequency, field, relative)
    finally:
        ops["destroy"]()
