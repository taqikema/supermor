"""Block drawing adapters preserve actual responses and reject incomplete runs."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from figs.common.block_inputs import (
    COMPONENTS, CROSS_FREQUENCIES, STRATEGY_FREQUENCIES, STRATEGIES,
    component_nrmsd, cross_validation_inputs, expected_receivers, strategy_inputs,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def save_archive(path, frequencies, electric, magnetic, **extras):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=np.arange(len(frequencies)),
        frequencies_hz=frequencies, rx_xyz=expected_receivers(),
        E=electric, H=magnetic, **extras)


@pytest.fixture
def cross_run(tmp_path):
    run = tmp_path / 'cross'
    write_json(run / 'summary.json', {'status': 'complete'})
    base = np.ones((4, 3321, 3), complex) * (2 + 3j)
    # Keep component-dependent complex phases so using amplitudes in the
    # NRMSD formula or concatenating H/E in the wrong order is detectable.
    base *= np.array([1., 2j, 3 - 1j])
    for method, offset in (('DFOM', 0.), ('SUPERMOR', .001j), ('EMG3D', .02j)):
        save_archive(run / method / 'responses.npz', CROSS_FREQUENCIES,
            base + offset, .2j * base + 2 * offset,
            physical_hash='physical', receivers_hash='receivers', frequencies_hash='frequencies',
            meta=json.dumps([{'status': 'complete', 'residual': 1e-8}] * 4))
    return run


def test_complex_nrmsd_symmetry_and_undefined_zero_denominator():
    a, b = np.array([1j, 2 + 2j, 0.]), np.array([-1j, 1 + 1j, 0.])
    np.testing.assert_array_equal(component_nrmsd(a, b), component_nrmsd(b, a))
    np.testing.assert_allclose(component_nrmsd(a, b)[:2], [200., 200 / 3])
    assert np.isnan(component_nrmsd(a, b)[2])


def test_cross_export_uses_complex_component_errors_and_actual_crossline(cross_run):
    target = cross_validation_inputs(cross_run)
    with np.load(cross_run / 'DFOM/responses.npz') as a, np.load(cross_run / 'EMG3D/responses.npz') as b:
        expected = component_nrmsd(np.concatenate((a['E'], a['H']), axis=2),
                                   np.concatenate((b['E'], b['H']), axis=2))
        with np.load(target / 'dfom_vs_emg3d/pointwise_data.npz') as output:
            np.testing.assert_array_equal(output['components'], COMPONENTS)
            np.testing.assert_array_equal(output['pointwise_nrmsd_percent'], expected)
        with (target / 'crossline_10hz/line_responses.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 41
        assert [float(row['y_m']) for row in rows] == list(np.arange(-1000., 1001., 50.))
        index = np.flatnonzero((expected_receivers()[:, 0] == 3000.)
                              & (expected_receivers()[:, 1] == -1000.))[0]
        actual = complex(float(rows[0]['dfom_Ex_real']), float(rows[0]['dfom_Ex_imag']))
        assert actual == a['E'][2, index, 0]
    provenance = json.loads((target / 'source_metadata.json').read_text())
    assert provenance['run_directory'] == str(cross_run)
    assert len(provenance['inputs']) == 4
    assert all(record['sha256'] for record in provenance['inputs'])
    assert target == cross_run / 'plot_inputs/cross_validation'


def test_cross_export_rejects_incomplete_run_before_writing(cross_run):
    write_json(cross_run / 'summary.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='not complete'):
        cross_validation_inputs(cross_run)
    assert not (cross_run / 'plot_inputs').exists()


def test_cross_export_rejects_physical_identity_mismatch(cross_run):
    path = cross_run / 'EMG3D/responses.npz'
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays['physical_hash'] = 'different model'
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='physical input identity'):
        cross_validation_inputs(cross_run)
    assert not (cross_run / 'plot_inputs').exists()


def test_cross_export_rejects_synthetic_responses(cross_run):
    path = cross_run / 'SUPERMOR/responses.npz'
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    np.savez_compressed(path, **arrays, synthetic=True)
    with pytest.raises(ValueError, match='Synthetic responses'):
        cross_validation_inputs(cross_run)
    assert not (cross_run / 'plot_inputs').exists()


@pytest.fixture
def strategy_run(tmp_path):
    run = tmp_path / 'strategy'
    reference = run / 'reference/receivers.npz'
    frequencies = STRATEGY_FREQUENCIES
    fields = np.ones((20, 3321, 3), complex) * 1j
    fields[:, :, 1:] = 0
    fields[:, 0, 0] = 100j
    save_archive(reference, frequencies, fields, 2 * fields)
    identity = dict(K='K', M='M', q='q', receivers='rx')
    records = [dict(index=i, frequency=float(f), verified=i != 0,
                    field_origin='independent_ufl_mumps', residual=5e-7 if i == 0 else 1e-12)
               for i, f in enumerate(frequencies)]
    write_json(reference.with_name('stats.json'), dict(route='independent_mixed_ufl',
        per_frequency=records, input_sha256=identity, status='unverified'))
    summaries = []
    for name in STRATEGIES['gradient']:
        actual = fields.copy()
        actual[:, -1, 0] *= 2
        save_archive(run / name / 'receivers.npz', frequencies, actual, 2 * actual)
        write_json(run / name / 'stats.json', dict(status='converged', input_sha256=identity,
            dynamic_basis_columns=50, complete_forward_s=12.5))
        write_json(run / name / 'resources.json', dict(returncode=0, peak_job_pss_bytes=2**30))
        summaries.append(dict(strategy=name, returncode=0, status='converged', reference=str(reference)))
    write_json(run / 'summary.json', summaries)
    return run


def test_strategy_export_uses_pointwise_vector_max_and_preserves_reference_flags(strategy_run):
    target = strategy_inputs(strategy_run, 'gradient')
    with (target / 'frequency_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60
    assert {float(row['receiver_E_max_pointwise_pct']) for row in rows} == {100.}
    assert {float(row['receiver_H_max_pointwise_pct']) for row in rows} == {100.}
    report = json.loads((target / 'comparison.json').read_text())
    assert report['reference']['native_reference_verified'] is False
    assert report['reference']['controller_summary']['per_frequency'][0]['verified'] is False
    assert report['reference']['maximum_recorded_residual'] == 5e-7
    assert report['provenance']['run_directory'] == str(strategy_run)


def test_strategy_export_rejects_assembled_input_mismatch(strategy_run):
    path = strategy_run / 'total_field_pcg/stats.json'
    stats = json.loads(path.read_text())
    stats['input_sha256']['K'] = 'other-mesh'
    write_json(path, stats)
    with pytest.raises(ValueError, match='input identity mismatch'):
        strategy_inputs(strategy_run, 'gradient')
    assert not (strategy_run / 'plot_inputs').exists()


def test_strategy_export_rejects_failed_reference_field(strategy_run):
    path = strategy_run / 'reference/stats.json'
    stats = json.loads(path.read_text())
    stats['per_frequency'][0]['field_origin'] = 'zero_fallback_after_solve_error'
    write_json(path, stats)
    with pytest.raises(ValueError, match='reference lacks complete'):
        strategy_inputs(strategy_run, 'gradient')
    assert not (strategy_run / 'plot_inputs').exists()


def test_strategy_export_rejects_zero_reference_vector_without_masking(strategy_run):
    path = strategy_run / 'reference/receivers.npz'
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays['E'][0, -1] = 0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='pointwise maximum undefined'):
        strategy_inputs(strategy_run, 'gradient')
    assert not (strategy_run / 'plot_inputs').exists()


def test_strategy_export_rejects_missing_method_instead_of_using_paper_data(strategy_run):
    with pytest.raises(ValueError, match='missing required poles'):
        strategy_inputs(strategy_run, 'poles')
    assert not (strategy_run / 'plot_inputs').exists()
