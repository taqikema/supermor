"""Native owned-cell VTI assignment and collective material validation."""
import dolfin as df
import numpy as np
import pytest

from supermor.methods import solve
from supermor.materials import assign_native_vti
from supermor.operators import assemble_mor

from .fe_helpers import independent_matrices, full_matrix, small_model


class LinearVTI:
    def sample(self, points, domains, chunk_size):
        horizontal = 1. + points[:, 0]
        return horizontal, .25 * horizontal


def test_native_dg0_vti_preserves_physical_energy_and_assembled_mass():
    model = small_model(n=2)
    sigma = assign_native_vti(model, LinearVTI(), chunk_size=3)
    direction = df.Constant((1., 2., 3.))
    energy = df.assemble(df.inner(sigma * direction, direction) * df.dx)
    assert energy == pytest.approx(1.5 * (5. + 9. * .25), rel=2e-13)
    assert model.MP.tensor_flag and model.MP.sigma_tensor_flag
    ops = assemble_mor(model)
    try:
        _, expected = independent_matrices(model)
        np.testing.assert_allclose(full_matrix(ops["M"]), expected,
            rtol=2e-13, atol=2e-14)
        np.linalg.cholesky(expected)
        stats = solve(ops, [.1, 1.], method="SUPERMOR")
        assert stats["status"] == "converged", stats
    finally:
        ops["destroy"]()


@pytest.mark.parametrize("invalid", [0., np.nan])
def test_invalid_vti_sampling_on_one_rank_fails_collectively(invalid):
    model = small_model(n=2)
    comm = model.FS.mesh.mpi_comm()

    class InvalidOnLastRank(LinearVTI):
        def sample(self, points, domains, chunk_size):
            horizontal, vertical = super().sample(points, domains, chunk_size)
            if comm.rank == comm.size - 1:
                vertical[:] = invalid
            return horizontal, vertical

    with pytest.raises(ValueError, match="finite positive") as caught:
        assign_native_vti(model, InvalidOnLastRank(), chunk_size=3)
    messages = comm.allgather(str(caught.value))
    assert all(message == messages[0] for message in messages)
    assert f"rank {comm.size - 1}:" in messages[0]


@pytest.mark.parametrize("sigma", [np.diag([1., 1., 0.]),
    np.array([[1., .2, 0.], [0., 1., 0.], [0., 0., 1.]])])
def test_invalid_tensor_conductivity_is_rejected_before_solving(sigma):
    model = small_model(sigma=df.Constant(sigma))
    with pytest.raises(ValueError, match="positive definite|symmetric"):
        assemble_mor(model)


def test_tensor_inverse_permeability_is_outside_the_paper_contract():
    model = small_model()
    model.MP.mu_inv_func = df.Constant(np.eye(3))
    with pytest.raises(ValueError, match="isotropic scalar"):
        assemble_mor(model)
