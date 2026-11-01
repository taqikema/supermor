"""Restarted rational Krylov regression against independent dense oracles.

Migrated from custEM/tests/test_mor.py. The production API calls this route
CPC; the paper names the same route restarted RK. Run in serial or MPI2.
"""
import numpy as np
import pytest
from petsc4py import PETSc

from supermor import mor
from supermor.mor import solve_cpc, solve_split_u
from .helpers import matrix, problem, release, run


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("route", ["cpc"])
def test_complete_mass_against_dense_full_order(route, diagonal):
    ops, K, M, q = problem(diagonal)
    frequencies = np.array([.1, 1., 100.])
    try:
        algorithm = solve_cpc if route == "cpc" else solve_split_u
        options = (dict(steps=8) if route == "cpc" else
                   dict(direct=True, check_every=1, pole_factor=1., basis_strategy='chain'))
        stats, fields = run(algorithm, ops, frequencies, **options)
        assert stats["converged"], stats
        expected_pole = -2*np.pi*np.sqrt(frequencies.min()*frequencies.max())
        assert stats["xi0"] == pytest.approx(expected_pole)
        for index, frequency in enumerate(frequencies):
            s = 2j*np.pi*frequency
            exact = np.linalg.solve(K+s*M, s*q)
            difference = fields[index]-exact
            assert np.sqrt(np.vdot(difference, M@difference).real /
                           np.vdot(exact, M@exact).real) < 1e-6
            assert np.linalg.norm(s*q-(K+s*M)@fields[index])/np.linalg.norm(s*q) < 1e-6
    finally:
        release(ops)


@pytest.mark.parametrize("source", ["zero", "gradient"])
def test_cpc_real_source_handles_zero_and_invariant_gradient(source):
    ops, K, M, q = problem(source=source)
    try:
        stats, fields = run(solve_cpc, ops, [.1, 1., 100.])
        assert stats["converged"], stats
        assert stats["counts"].get("P_rhs", 0) <= (source == "gradient")
        for index, frequency in enumerate([.1, 1., 100.]):
            exact = np.linalg.solve(K+2j*np.pi*frequency*M, 2j*np.pi*frequency*q)
            np.testing.assert_allclose(fields[index], exact, rtol=1e-10, atol=1e-11)
    finally:
        release(ops)


@pytest.mark.parametrize("steps", [1, 3])
def test_cpc_real_arnoldi_relation_and_tail_correction(monkeypatch, steps):
    """Audit the actual shared basis and its one-cycle tail independently."""
    import supermor.mor as mor
    original = mor._source_basis
    captured = []
    comm = PETSc.COMM_WORLD.tompi4py()

    def retain(comm, mass, shifted, seed, columns, stats):
        basis = original(comm, mass, shifted, seed, columns, stats)
        V, Hbar, eta, tail, beta, updates, gram, breakdown = basis
        local = np.column_stack(V) if isinstance(V, list) else np.asarray(V)
        full_V = np.vstack(comm.allgather(local))
        full_tail = None if tail is None else np.concatenate(comm.allgather(tail))
        captured.append((full_V, Hbar.copy(), eta, full_tail, beta, updates, gram, breakdown))
        return basis

    monkeypatch.setattr(mor, "_source_basis", retain)
    ops, K, M, q = problem()
    frequencies = np.array([.1, 1., 100.])
    try:
        stats, fields = run(solve_cpc, ops, frequencies, steps=steps, max_cycles=1, tol=1e-14)
        assert not stats["converged"], stats
        assert len(captured) == 1
        V, Hbar, eta, tail, beta, updates, gram, breakdown = captured[0]
        assert V.shape == (len(q), steps)
        assert Hbar.shape == (steps+1, steps)
        assert updates == steps and breakdown is None and tail is not None
        assert not np.iscomplexobj(V) and not np.iscomplexobj(tail)
        extended = np.column_stack((V, tail))
        np.testing.assert_allclose(extended.T@M@extended, np.eye(steps+1),
                                   rtol=0., atol=1e-12)
        pole = stats["xi0"]
        images = np.linalg.solve(K-pole*M, M@V)
        np.testing.assert_allclose(images, extended@Hbar, rtol=1e-11, atol=1e-13)
        assert gram <= 1e-10
        assert stats["counts"]["P_rhs"] == steps
        assert stats["counts"]["M_rhs"] == 1
        e1 = np.zeros(steps)
        e1[0] = eta
        H = Hbar[:steps]
        for index, frequency in enumerate(frequencies):
            s = 2j*np.pi*frequency
            mu = pole+s
            y = np.linalg.solve(np.eye(steps)+mu*H, e1)
            d = beta*y[-1]
            expected = s*(V@(H@y)+d*tail)
            np.testing.assert_allclose(fields[index], expected, rtol=1e-11, atol=1e-12)
            residual = s*q-(K+s*M)@fields[index]
            theoretical = -s*mu*d*(M@tail)
            np.testing.assert_allclose(residual, theoretical, rtol=1e-10, atol=1e-11)
    finally:
        release(ops)


def test_cpc_shares_direct_inverse_chain_across_duplicate_frequencies():
    ops, _, _, _ = problem()
    try:
        single, before = run(solve_cpc, ops, [.1, 1., 100.], steps=8)
        duplicated, after = run(solve_cpc, ops, [.1, 1., 100., .1, 1., 100.], steps=8)
        assert single["converged"] and duplicated["converged"]
        assert single["counts"]["P_rhs"] == duplicated["counts"]["P_rhs"]
        assert single["counts"]["M_rhs"] == duplicated["counts"]["M_rhs"] == 1
        assert single["counts"]["P_factorizations"] == duplicated["counts"]["P_factorizations"] == 1
        for index in range(6):
            np.testing.assert_allclose(after[index], before[index % 3], rtol=1e-12, atol=1e-12)
    finally:
        release(ops)


def test_cpc_later_restart_failure_keeps_last_finite_fields(monkeypatch):
    import supermor.mor as mor
    ops, _, _, _ = problem()
    try:
        first, expected = run(solve_cpc, ops, [.1, 1., 100.], steps=1,
                              max_cycles=1, tol=1e-14)
        assert not first["converged"]
        original, calls = mor._residual, []

        def fail_second_cycle(*args, **kwargs):
            calls.append(True)
            # Three checks complete cycle 1. Fail after cycle 2 has already
            # changed one frequency's residual and appended its pending block.
            if len(calls) == 5:
                raise mor.MORFailure("controlled shared-source restart failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(mor, "_residual", fail_second_cycle)
        stats, fields = run(solve_cpc, ops, [.1, 1., 100.], steps=1,
                             max_cycles=3, tol=1e-14)
        assert not stats["converged"]
        assert "controlled shared-source restart failure" in stats["failure"]
        assert len(calls) == 5
        assert stats["retained_basis_columns"] == first["retained_basis_columns"]
        np.testing.assert_allclose(stats["relative_residuals"], first["relative_residuals"],
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(stats["physical_relative_residuals"], first["physical_relative_residuals"],
                                   rtol=1e-13, atol=1e-13)
        assert [row["cycles"] for row in stats["per_frequency"]] == [1, 1, 1]
        assert len(fields) == len(expected) == 3
        for index in expected:
            assert np.isfinite(fields[index]).all()
            np.testing.assert_allclose(fields[index], expected[index], rtol=1e-13, atol=1e-13)
    finally:
        release(ops)


@pytest.mark.parametrize("route", ["cpc"])
def test_frequency_order_and_duplicates(route):
    ops, K, M, q = problem()
    frequencies = [100., .1, 1., .1]
    try:
        algorithm = solve_cpc if route == "cpc" else solve_split_u
        options = dict(steps=8) if route == "cpc" else dict(direct=True)
        stats, fields = run(algorithm, ops, frequencies, **options)
        assert stats["converged"], stats
        np.testing.assert_allclose(fields[1], fields[3], rtol=1e-10, atol=1e-12)
        for index, frequency in enumerate(frequencies):
            s = 2j*np.pi*frequency
            np.testing.assert_allclose(fields[index], np.linalg.solve(K+s*M, s*q),
                                       rtol=1e-6, atol=1e-9)
    finally:
        release(ops)


