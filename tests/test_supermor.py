"""P1 SUPERMOR split-u, total-field and pole regression oracles.

Migrated from custEM/tests/test_mor.py, test_mor_split_u.py and
 test_mor_adaptive.py. Dense full-order solutions independently check
returned fields and true residuals. Run in serial or MPI2.
"""
import numpy as np
import pytest
from petsc4py import PETSc

from supermor import mor
from supermor.mor import solve_cpc, solve_split_u
from .helpers import matrix, problem, release, run
from .helpers import assert_fields as _assert_fields


@pytest.mark.parametrize("formulation", ["split_u", "total_field"])
def test_formulations_match_dense_fom_with_complete_nondiagonal_mass(formulation):
    ops, K, M, q = problem(size=15)
    frequencies = [8192., .015625, .5, 128., .015625]
    try:
        stats, fields = run(mor.solve_split_u, ops, frequencies,
            field_formulation=formulation, direct=True, max_basis=15, tol=1e-10)
        assert stats["converged"], stats
        assert stats["gram_error"] <= 1e-10
        assert stats["residual_verified"]
        assert stats["residual_kind"] == "assembled_matrix"
        assert stats["dynamic_basis_columns"] == stats["basis_columns"]
        assert stats["counts"]["direction_defect_checks"] == 0
        _assert_fields(K, M, q, frequencies, fields, 1e-9)
    finally:
        release(ops)


def test_total_field_retains_gradient_source_without_explicit_poisson_or_cleanup(monkeypatch):
    ops, K, M, q = problem(source="gradient")
    frequencies = [.1, 1., 100.]
    gradient = ops["G"]

    class NoExplicitPoissonGradient:
        def __getattr__(self, name):
            return getattr(gradient, name)

        def transposeMatMult(self, unused):
            pytest.fail("total_field must not construct an explicit nodal Poisson matrix")

    initialize, action = mor._LinearSolve.__init__, mor._Action.__call__

    def no_poisson(self, matrix, stats, name, *args, **kwargs):
        assert name != "poisson"
        return initialize(self, matrix, stats, name, *args, **kwargs)

    def no_gradient_action(self, *args, **kwargs):
        assert self.matrix is not ops["G"], "total_field must not apply gradient cleanup"
        return action(self, *args, **kwargs)

    try:
        split, unused_fields = run(mor.solve_split_u, ops, frequencies, direct=True)
        assert split["converged"] and split["basis_columns"] == 0
        ops["G"] = NoExplicitPoissonGradient()
        monkeypatch.setattr(mor._LinearSolve, "__init__", no_poisson)
        monkeypatch.setattr(mor._Action, "__call__", no_gradient_action)
        total, fields = run(mor.solve_split_u, ops, frequencies,
                            field_formulation="total_field", direct=True)
        assert total["converged"], total
        assert total["counts"]["P_rhs"] >= 1
        assert total["basis_columns"] >= 1
        assert total["counts"]["poisson_setups"] == 0
        assert total["counts"].get("poisson_rhs", 0) == 0
        assert not total["static_split"] and not total["gradient_cleanup"]
        assert total["static_vector_count"] == 0
        for direction in total["directions"]:
            assert direction["scaled_nodal_defect"] is None
            assert direction["projection_error"] is None
            assert not direction["gradient_constraint_checked"]
        _assert_fields(K, M, q, frequencies, fields, 1e-8)
    finally:
        ops["G"] = gradient
        release(ops)


def test_total_field_basis_budget_failure_verifies_and_emits_partial_fields():
    ops, K, M, q = problem(size=18)
    frequencies = [.015625, 1., 8192.]
    try:
        stats, fields = run(mor.solve_split_u, ops, frequencies,
            field_formulation="total_field", direct=True, max_basis=2, tol=1e-30)
        assert not stats["converged"]
        assert "exhausted 2 basis columns" in stats["failure"]
        assert stats["basis_columns"] == stats["emitted_basis_columns"] == 2
        assert stats["residual_verified"]
        assert stats["counts"]["true_residual_checks"] == len(frequencies)
        for index, frequency in enumerate(frequencies):
            s = 2j * np.pi * frequency
            residual = np.linalg.norm(s * q - (K + s * M) @ fields[index]) / np.linalg.norm(s * q)
            assert stats["relative_residuals"][index] == pytest.approx(residual, rel=1e-11)
    finally:
        release(ops)


def test_residual_mr_strategy_matches_independent_dense_fom_on_complete_mass():
    ops, K, M, q = problem(diagonal=False, size=15)
    frequencies = [.015625, .5, 128., 8192.]
    try:
        stats, fields = run(mor.solve_split_u, ops, frequencies,
            basis_strategy="residual_mr", direct=True, poles=(-80.,),
            check_every=9, max_basis=15, tol=1e-10)
        assert stats["converged"], stats
        assert stats["gram_error"] <= 1e-10
        assert stats["counts"]["direction_defect_checks"] == 0
        for index, frequency in enumerate(frequencies):
            s = 2j*np.pi*frequency
            exact = np.linalg.solve(K+s*M, s*q)
            delta = fields[index]-exact
            relative_M_error = np.sqrt(np.vdot(delta, M@delta).real/np.vdot(exact, M@exact).real)
            assert relative_M_error <= 1e-9
            assert np.linalg.norm(s*q-(K+s*M)@fields[index])/np.linalg.norm(s*q) <= 1e-10
    finally:
        release(ops)


@pytest.mark.parametrize("source", ["zero", "gradient"])
def test_residual_mr_strategy_static_sources_do_not_request_shifted_inverse(source):
    ops, K, M, q = problem(source=source)
    try:
        stats, fields = run(mor.solve_split_u, ops, [.1, 1., 100.],
            basis_strategy="residual_mr", direct=True)
        assert stats["converged"], stats
        assert stats["counts"].get("P_rhs", 0) == 0
        assert stats["residual_verified"]
        assert stats["residual_kind"] == "assembled_matrix"
        assert stats["counts"]["true_residual_checks"] == 3
        for index, frequency in enumerate([.1, 1., 100.]):
            s = 2j*np.pi*frequency
            np.testing.assert_allclose(fields[index], np.linalg.solve(K+s*M, s*q),
                                       rtol=1e-8, atol=1e-10)
    finally:
        release(ops)


def test_residual_mr_false_predicted_convergence_is_verified_and_continues(monkeypatch):
    ops, unused_K, unused_M, unused_q = problem(size=18)
    frequencies = [.015625, 1., 8192.]
    original = mor._residual_mr

    def underestimate_first(comm, KV, *args, **kwargs):
        coefficients, info = original(comm, KV, *args, **kwargs)
        if coefficients.shape[0] == 1:
            info["predicted_residuals"] = [0.]*len(frequencies)
        return coefficients, info

    monkeypatch.setattr(mor, "_residual_mr", underestimate_first)
    try:
        stats, fields = run(mor.solve_split_u, ops, frequencies,
            basis_strategy="residual_mr", direct=True, poles=(-80.,),
            max_basis=5, tol=1e-30)
        assert stats["counts"]["P_rhs"] == 5
        first = next(row for row in stats["trace"] if row["basis_columns"] == 1)
        assert first["residual_verified"]
        assert first["predicted_relative_residuals"] == [0.]*len(frequencies)
        assert first["max_relative_residual"] > 1e-6
        assert not stats["converged"]
        assert stats["counts"]["full_residual_checkpoints"] == 2
        assert stats["counts"]["true_residual_checks"] == 2*len(frequencies)
        assert len(fields) == len(frequencies)
    finally:
        release(ops)


def test_static_solution_does_not_depend_on_reference_node():
    ops, _, _, _ = problem()
    gradient = ops["G"]
    try:
        original, before = run(solve_split_u, ops, [.1, 1., 100.], direct=True)
        assert original["converged"], original
        # Reordering nodes changes the pinned global node but leaves grad space.
        first, last = gradient.getOwnershipRange()
        local = gradient.getValues(np.arange(first, last, dtype=PETSc.IntType),
                                   np.arange(4, dtype=PETSc.IntType))
        dense = np.vstack(PETSc.COMM_WORLD.tompi4py().allgather(local))
        ops["G"] = matrix(dense[:, [2, 0, 1, 3]])
        changed, after = run(solve_split_u, ops, [.1, 1., 100.], direct=True)
        assert changed["converged"], changed
        for index in before:
            np.testing.assert_allclose(after[index], before[index], rtol=1e-8, atol=1e-10)
    finally:
        if ops["G"] is not gradient:
            gradient.destroy()
        release(ops)


@pytest.mark.parametrize('frequencies,expected', [
    ([8192., .015625, 8., 16., .015625, 8192.],
     [-2*np.pi*np.sqrt(16.*8192.), -2*np.pi*np.sqrt(.015625*8.)]),
    ([1., 4., 16.], [-2*np.pi*16., -2*np.pi*2.]),
    ([3., 3.], [-2*np.pi*3.]),
])
def test_split_u_default_poles_use_full_frequency_range_and_high_first(frequencies, expected):
    ops, _, _, _ = problem(source='zero')
    try:
        stats, fields = run(solve_split_u, ops, frequencies)
        assert stats['converged'], stats
        np.testing.assert_allclose(stats['poles'], expected, rtol=1e-14)
        assert stats['pole'] == stats['xi0'] == pytest.approx(expected[0])
        assert stats['pole_every'] == 20
        assert stats['pole_schedule'] == 'hold_first'
        assert stats['basis_strategy'] == 'residual_mr'
        assert stats['basis_budget'] == 400
        assert stats['check_every'] == 1
        assert stats['basis_columns'] == 0
        assert stats['counts'].get('P_setups', 0) == 0
        assert all(np.count_nonzero(field) == 0 for field in fields.values())
    finally:
        release(ops)


@pytest.mark.parametrize('formulation', ['split_u', 'total_field'])
def test_split_u_default_switches_at_20_then_40_accepted_columns(monkeypatch, formulation):
    import supermor.mor as mor
    ops, _, _, _ = problem(size=48)
    inverse = mor._LinearSolve.__call__
    rng = np.random.default_rng(93014)
    independent = rng.normal(size=(41, 48))
    calls = []

    def observe(self, rhs, *args, **kwargs):
        solution = inverse(self, rhs, *args, **kwargs)
        if self.name == 'P':
            calls.append(self.stats['current_pole'])
            lo, hi = ops['q'].getOwnershipRange()
            # Isolate the scheduling decision from numerical deflation. Real
            # inverse/Poisson calls and matrix assembly still run; the complete
            # numerical method has independent direct-FOM tests below.
            return independent[len(calls)-1, lo:hi].copy()
        return solution

    monkeypatch.setattr(mor._LinearSolve, '__call__', observe)
    try:
        stats, _ = run(solve_split_u, ops, [.015625, 1., 8192.],
                       direct=True, max_basis=41, tol=1e-30,
                       field_formulation=formulation)
        high, low = stats['poles']
        assert stats['pole_every'] == 20
        assert stats['pole_schedule'] == 'hold_first'
        assert stats['basis_strategy'] == 'residual_mr'
        assert stats['basis_columns'] == 41, stats
        assert calls == [high]*20 + [low]*20 + [high]
        assert stats['counts']['P_factorizations'] == 3
        assert [row['cycle'] for row in stats['directions']] == [1]*20 + [2]*20 + [3]
        assert stats['gram_error'] <= 1e-10
    finally:
        release(ops)


def test_split_u_two_poles_matches_dense_fom_with_complete_mass_and_unordered_frequencies():
    ops, K, M, q = problem(size=15)
    frequencies = [100., .1, 1., .1]
    try:
        stats, fields = run(solve_split_u, ops, frequencies, direct=True,
                             poles=(-80., -2.), pole_every=2, check_every=2,
                             max_basis=20, tol=1e-11)
        assert stats['converged'], stats
        assert stats['basis_columns'] > 2
        assert stats['gram_error'] <= 1e-10
        assert stats['counts']['P_rhs'] == stats['basis_columns']
        assert stats['counts']['P_factorizations'] >= 2
        assert stats['counts']['poisson_factorizations'] == 1
        np.testing.assert_allclose(fields[1], fields[3], rtol=1e-13, atol=1e-13)
        for index, frequency in enumerate(frequencies):
            s = 2j*np.pi*frequency
            exact = np.linalg.solve(K+s*M, s*q)
            difference = fields[index]-exact
            assert np.sqrt(np.vdot(difference, M@difference).real /
                           np.vdot(exact, M@exact).real) <= 1e-9
            assert np.linalg.norm(s*q-(K+s*M)@fields[index])/np.linalg.norm(s*q) <= 1e-11
    finally:
        release(ops)


