"""P2 numerical contracts with independent UFL/dense small-model oracles."""
import dolfin as df
import numpy as np
import pytest

from supermor import mor
from supermor.materials import assign_native_vti
from supermor.methods import solve
from supermor.operators import assemble_mor

from .fe_helpers import (
    full_matrix, full_vector, independent_matrices, reference_receivers,
    run_method, small_model,
)


class LinearVTI:
    def sample(self, points, domains, chunk_size):
        horizontal = 1. + points[:, 0]
        return horizontal, .25 * horizontal


def p2_model(material="scalar"):
    model = small_model(degree=2)
    if material == "vti":
        assign_native_vti(model, LinearVTI(), chunk_size=2)
    return model


@pytest.mark.parametrize("method,options", [
    ("DFOM", {}), ("RK", {}), ("SUPERMOR", {}),
    ("SUPERMOR", {"field_formulation": "total_field"}),
])
@pytest.mark.parametrize("material", ["scalar", "vti"])
def test_p2_supported_matrix_paths_match_independent_reference(method, options, material):
    model = p2_model(material)
    ops = assemble_mor(model)
    try:
        stiffness, mass = independent_matrices(model)
        q = full_vector(ops["q"])
        np.testing.assert_allclose(full_matrix(ops["M"]), mass, rtol=2e-13, atol=2e-14)
        np.linalg.cholesky(mass)
        stats, rows = run_method(ops, model.MP.frequencies, method, **options)
        assert stats["status"] == "converged", stats
        if method == "SUPERMOR":
            assert stats["counts"]["ams_setups"] >= 1
            assert stats["counts"]["shift_factorizations"] == 0
            assert stats["counts"]["P_iterations"] > 0
        for index, frequency in enumerate(model.MP.frequencies):
            s = 2j * np.pi * frequency
            matrix, rhs = stiffness + s * mass, s * q
            expected = np.linalg.solve(matrix, rhs)
            assert np.linalg.norm(rhs - matrix @ expected) / np.linalg.norm(rhs) < 1e-10
            actual = rows[index]["field"]
            residual = np.linalg.norm(rhs - matrix @ actual) / np.linalg.norm(rhs)
            assert residual <= 1.01e-6, (method, material, frequency, residual)
            if method == "DFOM":
                assert residual < 1e-10
                assert stats["per_frequency"][index]["reference_accepted"]
            difference = actual - expected
            error = np.sqrt(np.vdot(difference, mass @ difference).real /
                            np.vdot(expected, mass @ expected).real)
            assert error < 1e-3, (method, material, frequency, error)
            reference = reference_receivers(ops, expected, s.imag)
            for field in ("E", "H"):
                received = rows[index]["receivers"][field]
                assert np.linalg.norm(received-reference[field]) <= (
                    1e-3*np.linalg.norm(reference[field]) + 1e-12)
    finally:
        ops["destroy"]()


@pytest.mark.parametrize("source", ["zero", "gradient"])
def test_p2_static_sources_recover_without_dynamic_basis(source):
    model = p2_model()
    ops = assemble_mor(model)
    gradient = ops["G"].createVecLeft()
    try:
        if source == "zero":
            gradient.set(0.)
            ops["q"].set(0.)
        else:
            phi = df.interpolate(df.Expression("x[0]*x[1]+x[2]*x[2]", degree=2), model.FS.S)
            ops["G"].mult(df.as_backend_type(phi.vector()).vec(), gradient)
            ops["M"].mult(gradient, ops["q"])
        expected, mass = full_vector(gradient), full_matrix(ops["M"])
        stats, rows = run_method(ops, model.MP.frequencies, "SUPERMOR")
        assert stats["status"] == "converged", stats
        assert stats["basis_columns"] == 0
        for row in rows.values():
            if source == "zero":
                np.testing.assert_array_equal(row["field"], 0.)
            else:
                delta = row["field"] - expected
                error = np.sqrt(np.vdot(delta, mass @ delta).real /
                                np.vdot(expected, mass @ expected).real)
                assert error < 1e-6
    finally:
        gradient.destroy()
        ops["destroy"]()


def test_p2_incomplete_basis_emits_finite_unaccepted_fields():
    model = p2_model()
    ops = assemble_mor(model)
    try:
        stats, rows = run_method(ops, model.MP.frequencies, "SUPERMOR", max_basis=1, tol=1e-12)
        assert stats["status"] == "failed", stats
        assert stats["failure"]
        assert set(rows) == set(range(len(model.MP.frequencies)))
        assert all(np.isfinite(row["field"]).all() for row in rows.values())
    finally:
        ops["destroy"]()


@pytest.mark.parametrize("options", [dict(basis_strategy="chain"), dict(residual_correction=True)])
def test_p2_unsupported_modes_fail_before_lazy_assembly(options):
    model = p2_model()
    ops = assemble_mor(model)
    try:
        def forbidden():
            pytest.fail("P2 unsupported mode reached P1-only assembly")
        ops["compute_curl"] = ops["compute_mass_lower_bound"] = forbidden
        with pytest.raises(ValueError, match="P2"):
            mor.solve_split_u(ops, model.MP.frequencies, **options)
    finally:
        ops["destroy"]()


def test_p2_ifom_is_explicitly_unsupported():
    model = p2_model()
    ops = assemble_mor(model)
    try:
        with pytest.raises((NotImplementedError, ValueError), match="P2.*IFOM|IFOM.*P2"):
            solve(ops, model.MP.frequencies, method="IFOM")
    finally:
        ops["destroy"]()


def test_p2_missing_interpolation_on_one_rank_fails_collectively():
    model = p2_model()
    ops = assemble_mor(model)
    try:
        if model.FS.mesh.mpi_comm().rank == 0:
            ops["ams_interpolation"] = None
        with pytest.raises(ValueError, match="P2.*interpolation") as caught:
            mor.solve_split_u(ops, model.MP.frequencies)
        messages = model.FS.mesh.mpi_comm().allgather(str(caught.value))
        assert all(message == messages[0] for message in messages)
    finally:
        ops["destroy"]()


@pytest.mark.parametrize("conductivity", [0., np.diag([1., 1., 0.]),
                                         np.array([[1., .2, 0.], [0., 1., 0.], [0., 0., 1.]])])
def test_p2_invalid_conductivity_is_rejected_before_solving(conductivity):
    model = p2_model()
    model.MP.sigma_func = df.Constant(conductivity)
    with pytest.raises(ValueError, match="positive|symmetric"):
        assemble_mor(model)


def test_p2_poisson_gauge_uses_strongest_global_diagonal():
    model = p2_model()
    ops = assemble_mor(model)
    product = ops["M"].matMult(ops["G"])
    laplacian = ops["G"].transposeMatMult(product)
    diagonal = laplacian.createVecLeft()
    try:
        laplacian.getDiagonal(diagonal)
        values = full_vector(diagonal)
        expected = int(np.argmax(values))
        stats = solve(ops, model.MP.frequencies, method="SUPERMOR")
        assert stats["status"] == "converged", stats
        assert stats["poisson_pinned_dof"] == expected
        assert stats["poisson_pinned_diagonal"] == pytest.approx(values[expected], rel=1e-14)
    finally:
        diagonal.destroy()
        laplacian.destroy()
        product.destroy()
        ops["destroy"]()
