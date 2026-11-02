"""Independent P1 oracles for the historical sphere MPI Kg kernel."""
from types import SimpleNamespace

import numpy as np
import pytest
from petsc4py import PETSc

from supermor import mor, mor_mpi_legacy as legacy
from supermor.operators import assemble_mor

from .helpers import problem, release, run, assert_fields, rows
from .fe_helpers import small_model, independent_matrices, full_vector


def test_legacy_mr_targets_h_minus_kg_over_s():
    rng = np.random.default_rng(7321)
    KV, MV = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    h, kg = rng.normal(size=20), rng.normal(size=20)
    frequencies = [.015625, 1., 8192.]
    stats = dict(seconds={}, counts={})
    coefficients, information = legacy._residual_mr(PETSc.COMM_WORLD.tompi4py(),
        rows(KV), rows(MV), rows(h), rows(kg), frequencies, stats)
    for index, frequency in enumerate(frequencies):
        s = 2j * np.pi * frequency
        A, b = KV + s*MV, h - kg/s
        expected = np.linalg.lstsq(A, b, rcond=None)[0]
        assert np.allclose(A@coefficients[:, index], A@expected, rtol=2e-11, atol=2e-11)
        norm = np.linalg.norm(b-A@coefficients[:, index])
        assert information['predicted_residuals'][index] == pytest.approx(norm, rel=2e-11, abs=2e-11)
    # The nonzero Kg term changes this target; the no-Kg solve is not substituted.
    no_kg = np.linalg.lstsq(KV + 2j*np.pi*frequencies[0]*MV, h, rcond=None)[0]
    assert np.linalg.norm(coefficients[:, 0]-no_kg) > .1


def test_legacy_matrix_fields_satisfy_original_maxwell_system():
    ops, stiffness, mass, source = problem(size=15)
    frequencies = [.015625, 1., 8192.]
    try:
        stats, fields = run(legacy.solve_split_u, ops, frequencies,
                           direct=True, max_basis=15, tol=1e-8)
        assert stats['status'] == 'converged', stats
        assert stats['historical_kernel']['retained_Kg_MR_term'] is True
        assert_fields(stiffness, mass, source, frequencies, fields, 1e-7)
    finally:
        release(ops)


def test_legacy_finite_element_fields_satisfy_independent_ufl_matrices():
    model = small_model(frequencies=(.015625, 1., 8192.))
    ops = assemble_mor(model)
    try:
        stiffness, mass = independent_matrices(model)
        source = full_vector(ops['q'])
        stats, fields = run(legacy.solve_split_u, ops, model.MP.frequencies,
                           direct=True, max_basis=model.FS.V.dim(), tol=1e-8)
        assert stats['status'] == 'converged', stats
        assert_fields(stiffness, mass, source, model.MP.frequencies, fields, 1e-7)
    finally:
        ops['destroy']()


def test_legacy_rejects_p2_before_any_solver_setup():
    with pytest.raises(NotImplementedError, match='P1 only'):
        legacy.solve_split_u(dict(fe_degree=2), [1.])


def test_sphere_entry_selects_distinct_experiment_kernels(monkeypatch, tmp_path):
    from runs.case2_layered_sphere import run_methods_mpi as entry
    monkeypatch.setattr(entry, 'get_case_mesh', lambda *a, **kw:
        dict(metadata=tmp_path/'mesh.json', receivers=tmp_path/'receivers.csv'))
    mpi = entry.parse_args(['--method', 'SUPERMOR'])
    frequency = entry.parse_args(['--method', 'SUPERMOR', '--experiment', 'test2', '--n-frequency', '20'])
    current = entry.parse_args(['--method', 'SUPERMOR', '--kernel', 'current'])
    assert mpi.kernel == 'paper_mpi' and entry.kernel_module(mpi) is legacy
    assert frequency.kernel == 'current' and entry.kernel_module(frequency) is mor
    assert entry.kernel_module(current) is mor
    with pytest.raises(SystemExit):
        entry.parse_args(['--method', 'SUPERMOR', '--experiment', 'test2', '--n-frequency', '20', '--kernel', 'paper_mpi'])
