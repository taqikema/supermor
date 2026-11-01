"""Residual projection and incremental QR checked by independent least squares.

Migrated from custEM/tests/test_mor_adaptive.py and
 test_mor_incremental_qr.py. Includes empty ranks, dependent images and
wide scales. Run in serial or MPI2.
"""
import numpy as np
import pytest
from petsc4py import PETSc
from scipy.linalg import lstsq

from supermor import mor
from .helpers import (
    rows as _rows, gather as _gather, _comm, _compare_full_space,
)

@pytest.mark.parametrize("scale", [1e-280, 1., 1e280])
@pytest.mark.parametrize("phase", [0., .37, 1.7, np.pi / 2])
def test_real_residual_direction_is_phase_invariant_and_scale_safe(scale, phase):
    comm = PETSc.COMM_WORLD.tompi4py()
    x = np.array([1., 2., -.5, .75, 0., -.8, .2])
    y = np.array([.2, -.1, .3, -.2, .5, .3, -.1])
    residual = (x + 1j*y) * np.exp(1j*phase) * scale
    local = _rows(residual)
    retained = local.copy()
    direction, info = mor._real_residual_direction(comm, local)
    np.testing.assert_array_equal(local, retained)
    direction = _gather(direction)
    assert np.isrealobj(direction) and np.isfinite(direction).all()
    assert np.linalg.norm(direction) == pytest.approx(1., rel=2e-14)
    # A non-degenerate top eigenvalue makes the best real direction unique
    # up to sign. Multiplying the complex residual by a phase preserves it.
    columns = np.column_stack((x, y))
    unused, eigenvectors = np.linalg.eigh(columns.T @ columns)
    expected = columns @ eigenvectors[:, -1]
    expected /= np.linalg.norm(expected)
    assert abs(float(direction @ expected)) == pytest.approx(1., abs=2e-14)
    assert np.isfinite(info["residual_scale"])
    assert info["residual_scale"] > 0
    assert np.isfinite(info["coefficients"]).all()
    assert np.isfinite(info["eigenvalues"]).all()


@pytest.mark.parametrize("empty_rank", [False, True])
@pytest.mark.parametrize("static_split", [False, True])
def test_residual_mr_matches_full_space_lstsq(empty_rank, static_split):
    comm = PETSc.COMM_WORLD.tompi4py()
    rng = np.random.default_rng(72006)
    size, columns = 19, 5
    factor = rng.normal(size=(size, size))
    mass = factor.T @ factor + np.eye(size)
    rotation = np.linalg.qr(rng.normal(size=(size, size)))[0]
    curl = np.sqrt(np.r_[np.zeros(3), np.geomspace(.5, 35., size-3)])[:, None] * rotation.T
    stiffness = curl.T @ curl
    gradient = rotation[:, :3]
    basis = np.linalg.qr(rng.normal(size=(size, columns)))[0]
    q = rng.normal(size=size)
    g = gradient @ rng.normal(size=3) if static_split else np.zeros(size)
    # The physical split requires the gradient field to lie in ker(K).
    # An arbitrary SPD stiffness and arbitrary g would violate that premise.
    np.testing.assert_allclose(stiffness @ gradient, 0., atol=2e-13)
    h = q-mass@g
    KV, MV = stiffness@basis, mass@basis
    frequencies = np.array([8192., .015625, 1., .015625])
    retained = [value.copy() for value in (KV, MV, h)]
    stats = mor._stats("test_residual_mr", frequencies, -2.)
    coefficients, info = mor._residual_mr(
        comm, *[_rows(value, empty_rank) for value in (KV, MV, h)],
        frequencies, stats)
    assert coefficients.shape == (columns, len(frequencies))
    assert isinstance(info, dict)
    for index, frequency in enumerate(frequencies):
        s = 2j*np.pi*frequency
        A, b = KV+s*MV, h
        expected = lstsq(A, b)[0]
        np.testing.assert_allclose(coefficients[:, index], expected, rtol=2e-11, atol=2e-11)
        residual = b-A@coefficients[:, index]
        field = g+s*(basis@coefficients[:, index])
        equation_residual = s*q-(stiffness+s*mass)@field
        np.testing.assert_allclose(equation_residual/s, residual, rtol=2e-10, atol=2e-10)
        # The small QR problem predicts the unscaled u-equation residual.
        # Divide by ||q||, without another omega, to screen the original E
        # relative residual when the static field belongs to ker(K).
        predicted = info["predicted_residuals"][index]
        assert predicted == pytest.approx(np.linalg.norm(residual), rel=2e-11, abs=2e-11)
        assert predicted/np.linalg.norm(q) == pytest.approx(
            np.linalg.norm(equation_residual)/np.linalg.norm(s*q),
            rel=2e-11, abs=2e-11)
        assert np.linalg.norm(A.conj().T@residual) <= 2e-10*np.linalg.norm(A)*np.linalg.norm(b)
    for actual, expected in zip((KV, MV, h), retained):
        np.testing.assert_array_equal(actual, expected)


def test_residual_mr_nested_spaces_are_monotone_for_each_frequency():
    comm = PETSc.COMM_WORLD.tompi4py()
    rng = np.random.default_rng(72007)
    size, columns = 23, 7
    factor = rng.normal(size=(size, size))
    mass = factor.T@factor+np.eye(size)
    factor = rng.normal(size=(size, size))
    stiffness = factor.T@factor+.5*np.eye(size)
    basis = np.linalg.qr(rng.normal(size=(size, columns)))[0]
    h = rng.normal(size=size)
    KV, MV = stiffness@basis, mass@basis
    frequencies = np.array([.015625, 1., 8192.])
    previous = np.full(len(frequencies), np.inf)
    for width in (1, 3, 7):
        stats = mor._stats("test_nested_mr", frequencies, -2.)
        coefficients, unused = mor._residual_mr(
            comm, _rows(KV[:, :width]), _rows(MV[:, :width]), _rows(h),
            frequencies, stats)
        actual = np.array([np.linalg.norm(h-
            (KV[:, :width]+2j*np.pi*f*MV[:, :width])@coefficients[:, index])
            for index, f in enumerate(frequencies)])
        assert np.all(actual <= previous+1e-11)
        previous = actual


@pytest.mark.parametrize("empty_rank", [False, True])
def test_incremental_residual_mr_all_prefixes_match_full_space_lstsq(empty_rank):
    rng = np.random.default_rng(8217001)
    rows, columns = 29, 8
    KV, MV = rng.normal(size=(2, rows, columns))
    h = rng.normal(size=rows)
    frequencies = np.array([8192., .015625, 1., .015625])
    local = [_rows(value, empty_rank).copy() for value in (KV, MV, h)]
    retained = [value.copy() for value in local]
    stats = mor._stats("incremental_residual_qr", frequencies, -2.)
    qr = mor._ResidualQR(_comm(), local[0].shape[0], columns, stats)
    try:
        # Jumps append all intervening pairs; a repeated prefix is a read.
        for width in (0, 1, 3, 3, 8):
            coefficients, information = mor._residual_mr(
                _comm(), local[0][:, :width], local[1][:, :width],
                local[2], frequencies, stats, qr=qr)
            _compare_full_space(KV[:, :width], MV[:, :width], h,
                                frequencies, coefficients, information)
            for index, frequency in enumerate(frequencies):
                shift = 2j*np.pi*frequency
                weights = np.empty(1+2*width, dtype=complex)
                weights[0] = 1.
                weights[1::2] = -coefficients[:, index]
                weights[2::2] = -shift*coefficients[:, index]
                actual = qr.combination(weights)
                expected = h-(KV[:, :width]+shift*MV[:, :width])@coefficients[:, index]
                np.testing.assert_allclose(actual, _rows(expected, empty_rank),
                                           rtol=3e-11, atol=3e-11)
                norm = np.sqrt(_comm().allreduce(float(np.vdot(actual, actual).real)))
                assert norm == pytest.approx(information['predicted_residuals'][index],
                                             rel=3e-11, abs=3e-11)
            # Cached source columns and old trial images are factored once;
            # repeated prefix 3 must add neither reflectors nor work counts.
            assert stats["counts"]["residual_qr_columns"] == 1+2*width
            assert stats["counts"]["residual_qr_updates"] == 1+width
            for actual, original in zip(local, retained):
                np.testing.assert_array_equal(actual, original)
    finally:
        qr.destroy()


@pytest.mark.parametrize("images", ["exact_related", "near_related", "wide_scales"])
@pytest.mark.parametrize("zero_source", [False, True])
def test_incremental_residual_mr_handles_dependent_joint_images_and_scales(images, zero_source):
    rng = np.random.default_rng(8217002)
    rows, columns = 27, 7
    common, independent = rng.normal(size=(2, rows, columns))
    if images == "exact_related":
        # Joint W is rank deficient; the complex trial image is full rank.
        KV, MV = common, common*np.linspace(.5, 2., columns)
    elif images == "near_related":
        KV, MV = common+1e-10*independent, common
    else:
        scales = np.geomspace(1e-70, 1e70, columns)
        KV, MV = common*scales, (.2*common+independent)*scales
    h = rng.normal(size=rows)
    if zero_source:
        h[:] = 0
    frequencies = np.array([.015625, 1., 8192.])
    stats = mor._stats("incremental_dependent_images", frequencies, -2.)
    local = [_rows(value).copy() for value in (KV, MV, h)]
    qr = mor._ResidualQR(_comm(), local[0].shape[0], columns, stats)
    try:
        for width in (1, 4, columns):
            coefficients, information = mor._residual_mr(
                _comm(), local[0][:, :width], local[1][:, :width],
                local[2], frequencies, stats, qr=qr)
            _compare_full_space(KV[:, :width], MV[:, :width], h,
                                frequencies, coefficients, information)
            weights = np.ones(1+2*width, complex)
            # Cancel the artificial image scales, preserving a meaningful
            # reconstruction check even for the 1e-70 ... 1e70 input case.
            weights /= qr.scales[:qr.width]
            interleaved = np.column_stack([h]+[
                v for j in range(width) for v in (KV[:, j], MV[:, j])])
            np.testing.assert_allclose(qr.combination(weights), _rows(interleaved@weights),
                                       rtol=2e-12, atol=2e-12)
    finally:
        qr.destroy()


@pytest.mark.parametrize("rows", [0, 1, 3, 9])
def test_incremental_local_qr_preserves_norms_with_fewer_rows_than_columns(rows):
    rng = np.random.default_rng(8217003+rows)
    columns = 5
    # Each rank owns this many rows. The QR must support an empty factor as
    # well as a wide local matrix without inventing or deleting directions.
    KV, MV = rng.normal(size=(2, rows, columns))
    h = rng.normal(size=rows)
    stats = mor._stats("incremental_local_factor", np.array([1.]), -2.)
    qr = mor._ResidualQR(_comm(), rows, columns, stats)
    try:
        for width in (0, 1, columns):
            R, scales = qr(KV[:, :width], MV[:, :width], h)
            interleaved = np.column_stack(
                [h]+[v for index in range(width) for v in (KV[:, index], MV[:, index])])
            trial = rng.normal(size=(1+2*width, 4))
            image = (interleaved/scales)@trial
            reduced_image = R@trial
            np.testing.assert_allclose(reduced_image.T@reduced_image, image.T@image,
                                       rtol=5e-13, atol=5e-13)
    finally:
        qr.destroy()


