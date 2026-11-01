"""Small independent dense oracles and MPI helpers shared by core tests.

The algebraic builders are migrated from custEM/tests/test_mor.py; the
least-squares oracles are migrated from test_mor_incremental_qr.py. Only
tiny test matrices/vectors are gathered, never production problem fields.
"""
import numpy as np
import pytest
from petsc4py import PETSc
from scipy.linalg import lstsq

def matrix(values):
    values = np.asarray(values, dtype=float)
    result = PETSc.Mat().createAIJ(size=values.shape, comm=PETSc.COMM_WORLD)
    result.setUp()
    first, last = result.getOwnershipRange()
    columns = np.arange(values.shape[1], dtype=PETSc.IntType)
    for row in range(first, last):
        result.setValues(row, columns, values[row])
    result.assemble()
    return result


def problem(diagonal=False, source="general", size=9):
    rng = np.random.default_rng(17031)
    rotation, _ = np.linalg.qr(rng.normal(size=(size, size)))
    curl = np.sqrt(np.r_[np.zeros(3), np.geomspace(.2, 30., size-3)])[:, None] * rotation.T
    gradient = np.column_stack((-rotation[:, :3].sum(axis=1), rotation[:, :3]))
    if diagonal:
        mass = np.diag(np.linspace(.5, 2., size))
    else:
        factor = rng.normal(size=(size, size))
        mass = np.eye(size) + factor.T @ factor / size
    q = rng.normal(size=size)
    if source == "gradient":
        q = mass @ gradient @ np.arange(4.)
    elif source == "zero":
        q[:] = 0
    stiffness = curl.T @ curl
    K, M, G, C = map(matrix, (stiffness, mass, gradient, curl))
    rhs = M.createVecRight()
    first, last = rhs.getOwnershipRange()
    rhs.array[:] = q[first:last]
    scaled = mass / np.sqrt(np.outer(np.diag(mass), np.diag(mass)))
    ops = dict(K=K, M=M, G=G, q=rhs, curl=C, constants=None,
               mass_lower_bound=.5 * np.linalg.eigvalsh(scaled).min())
    return ops, stiffness, mass, q


def run(algorithm, ops, frequencies, **kwargs):
    """Small test vectors alone are gathered for the independent dense oracle."""
    fields = {}
    comm = PETSc.COMM_WORLD.tompi4py()

    def retain(index, real, imag):
        local = real.array.copy() + 1j * imag.array.copy()
        fields[index] = np.concatenate(comm.allgather(local))

    stats = algorithm(ops, frequencies, emit=retain, **kwargs)
    return stats, fields


def release(ops):
    for key in ("K", "M", "G", "curl", "q"):
        ops[key].destroy()


def rows(values, empty_rank=False):
    """Distribute a dense oracle, optionally giving rank 1 no local rows."""
    comm = PETSc.COMM_WORLD.tompi4py()
    if empty_rank:
        return values if comm.rank == 0 else values[:0]
    boundaries = np.linspace(0, len(values), comm.size + 1, dtype=int)
    return values[boundaries[comm.rank]:boundaries[comm.rank + 1]]


def gather(values):
    return np.concatenate(PETSc.COMM_WORLD.tompi4py().allgather(values))


def _comm():
    return PETSc.COMM_WORLD.tompi4py()


def _full_lstsq(KV, MV, h, frequency):
    """Independent complex least squares, scaled to retain wide columns."""
    s = 2j*np.pi*frequency
    A, b = KV+s*MV, h
    if not A.shape[1]:
        return np.empty(0, complex), np.zeros_like(b, dtype=complex), np.linalg.norm(b)
    scales = np.linalg.norm(A, axis=0)
    coefficients = lstsq(A/scales, b, cond=None, lapack_driver="gelsd")[0]/scales
    image = A@coefficients
    return coefficients, image, np.linalg.norm(b-image)


def _compare_full_space(KV, MV, h, frequencies, coefficients, information,
                        tolerance=3e-11):
    assert coefficients.shape == (KV.shape[1], len(frequencies))
    assert not information["truncated"]
    assert np.isfinite(coefficients).all()
    for index, frequency in enumerate(frequencies):
        s = 2j*np.pi*frequency
        A, b = KV+s*MV, h
        unused, expected_image, expected_norm = _full_lstsq(KV, MV, h, frequency)
        image = A@coefficients[:, index]
        scale = max(np.linalg.norm(b), 1.)
        assert np.linalg.norm(image-expected_image) <= tolerance*scale
        actual_norm = np.linalg.norm(b-image)
        assert actual_norm == pytest.approx(expected_norm, rel=tolerance, abs=tolerance*scale)
        assert information["predicted_residuals"][index] == pytest.approx(
            actual_norm, rel=tolerance, abs=tolerance*scale)
        # Check the independent full-space least-squares optimality condition.
        if A.shape[1]:
            scaled = A/np.linalg.norm(A, axis=0)
            assert np.linalg.norm(scaled.conj().T@(b-image)) <= tolerance*scale


def assert_fields(K, M, q, frequencies, fields, tolerance):
    for index, frequency in enumerate(frequencies):
        s = 2j * np.pi * frequency
        exact = np.linalg.solve(K + s * M, s * q)
        difference = fields[index] - exact
        relative = np.sqrt(np.vdot(difference, M @ difference).real /
                           np.vdot(exact, M @ exact).real)
        assert relative <= tolerance
        residual = np.linalg.norm(s * q - (K + s * M) @ fields[index])
        assert residual / np.linalg.norm(s * q) <= tolerance
