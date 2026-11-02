"""Prepare block drawing inputs from one explicitly selected complete run.

No solver is imported. Paper inputs are read as supplied; fresh run exports
remain under that run's ``plot_inputs`` directory and retain source hashes.
"""
from pathlib import Path
import csv
import json

import numpy as np

from figs.common.paths import fingerprint


CROSS_FREQUENCIES = np.array([.1, 1., 10., 100.])
STRATEGY_FREQUENCIES = 2. ** np.arange(-6, 14)
COMPONENTS = np.array(['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'])
STRATEGIES = {
    'gradient': {'split_u_pcg': 'split_u_pcg_1e-6',
                 'total_field_pcg': 'total_field_pcg_1e-6',
                 'total_field_lu': 'total_field_direct_lu'},
    'poles': {'single_pole': 'single_geometric', 'dual_pole': 'dual_existing'},
}


def expected_receivers():
    return np.column_stack((np.tile(np.arange(1000., 5001., 50.), 41),
                            np.repeat(np.arange(-1000., 1001., 50.), 81),
                            -np.ones(3321)))


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _archive(path, frequencies):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    if 'synthetic' in result and bool(result['synthetic']):
        raise ValueError('Synthetic responses cannot replace numerical run results: ' + str(path))
    if (not np.array_equal(result['frequencies_hz'], frequencies)
            or not np.array_equal(result['indices'], np.arange(len(frequencies)))
            or not np.array_equal(result['rx_xyz'], expected_receivers())):
        raise ValueError('Run archive requires the complete ordered paper frequencies and receivers: ' + str(path))
    for field in ('E', 'H'):
        values = result[field]
        if values.shape != (len(frequencies), 3321, 3) or not np.isfinite(values).all():
            raise ValueError('Incomplete or nonfinite receiver responses: ' + str(path))
    return result


def component_nrmsd(actual, reference):
    """Original complex scalar symmetric NRMSD; zero denominators stay undefined."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    denominator = np.abs(actual) + np.abs(reference)
    result = np.full(denominator.shape, np.nan)
    np.divide(200 * np.abs(actual - reference), denominator,
              out=result, where=denominator != 0)
    return result


def _provenance(run, paths):
    return {'input_kind': 'new_run', 'run_directory': str(run),
            'inputs': [fingerprint(path) for path in dict.fromkeys(paths)],
            'adapter': fingerprint(__file__)}


def cross_validation_inputs(run_dir):
    """Export figures 2--4 inputs from a full three-method cross-validation run."""
    run = Path(run_dir).expanduser().resolve()
    summary_path = run / 'summary.json'
    summary = _json(summary_path)
    if summary.get('status') != 'complete':
        raise ValueError('Cross-validation run is not complete')
    sources = [summary_path]
    archives = {}
    for method in ('DFOM', 'SUPERMOR', 'EMG3D'):
        path = run / method / 'responses.npz'
        archives[method.lower()] = _archive(path, CROSS_FREQUENCIES)
        sources.append(path)
        metadata = json.loads(str(archives[method.lower()]['meta']))
        if (len(metadata) != 4 or any(row.get('status') != 'complete' or row.get('synthetic', False)
                or not isinstance(row.get('residual'), (int, float))
                or not np.isfinite(row['residual']) or not 0 <= row['residual'] <= 1e-6
                for row in metadata)):
            raise ValueError('Cross-validation archive lacks accepted native diagnostics: ' + method)
    for key in ('physical_hash', 'receivers_hash', 'frequencies_hash'):
        values = [str(archive[key]) for archive in archives.values()]
        if len(set(values)) != 1:
            raise ValueError('Cross-validation physical input identity mismatch: ' + key)
    payloads = {}
    for pair, roles in (('dfom_vs_emg3d', ('dfom', 'emg3d')),
                        ('supermor_vs_dfom', ('supermor', 'dfom'))):
        a, b = [np.concatenate((archives[role]['E'], archives[role]['H']), axis=2)
                for role in roles]
        errors = component_nrmsd(a, b)
        plotted = errors[:, :, [0, 4]]
        if not np.isfinite(plotted).all():
            raise ValueError('Undefined Ex/Hy NRMSD; complete maps cannot be plotted')
        payloads[pair] = (dict(frequencies_hz=CROSS_FREQUENCIES,
            receiver_xyz=expected_receivers(), components=COMPONENTS,
            a_role=roles[0], b_role=roles[1], a_complex=a, b_complex=b,
            a_amplitude=np.abs(a), b_amplitude=np.abs(b),
            a_phase_deg=np.angle(a, deg=True), b_phase_deg=np.angle(b, deg=True),
            pointwise_nrmsd_percent=errors, synthetic=False),
            dict(synthetic=False, a_role=roles[0], b_role=roles[1],
                 pointwise_formula='200*abs(a-b)/(abs(a)+abs(b))',
                 pointwise_scale=dict(minimum_percent=0., maximum_percent=float(plotted.max()),
                                      actual_maximum_percent=float(plotted.max()), clipped=False)))
    rows = []
    coords = expected_receivers()
    for index in np.flatnonzero(coords[:, 0] == 3000.):
        row = dict(synthetic=False, frequency_hz=10., x_m=3000.,
                   y_m=float(coords[index, 1]), z_m=-1.)
        for role in ('dfom', 'emg3d'):
            for field, components in (('E', ('Ex', 'Ey', 'Ez')), ('H', ('Hx', 'Hy', 'Hz'))):
                for component, value in zip(components, archives[role][field][2, index]):
                    for name, number in (('real', value.real), ('imag', value.imag),
                                         ('amplitude', abs(value)), ('phase_deg', np.angle(value, deg=True))):
                        row[f'{role}_{component}_{name}'] = float(number)
        rows.append(row)
    target = run / 'plot_inputs/cross_validation'
    target.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(run, sources)
    for pair, (arrays, metadata) in payloads.items():
        folder = target / pair
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / 'pointwise_data.npz', **arrays)
        _write_json(folder / 'summary.json', {**metadata, 'provenance': provenance})
    (target / 'crossline_10hz').mkdir(exist_ok=True)
    _write_csv(target / 'crossline_10hz/line_responses.csv', rows)
    _write_json(target / 'source_metadata.json', provenance)
    return target


def _reference_records(stats, frequencies):
    records = stats.get('per_frequency', [])
    if (len(records) != len(frequencies)
            or [row.get('index') for row in records] != list(range(len(frequencies)))
            or [row.get('frequency') for row in records] != frequencies.tolist()
            or any(not isinstance(row.get('verified'), bool)
                   or row.get('field_origin') != 'independent_ufl_mumps'
                   or not isinstance(row.get('residual'), (int, float))
                   or not np.isfinite(row['residual']) or not 0 <= row['residual'] <= 1e-6
                   for row in records)):
        raise ValueError('Independent reference lacks complete acceptable native diagnostics')
    return records


def strategy_inputs(run_dir, group):
    """Export gradient or pole plots while preserving native reference diagnostics."""
    run = Path(run_dir).expanduser().resolve()
    summary_path = run / 'summary.json'
    summary = _json(summary_path)
    if not isinstance(summary, list):
        raise ValueError('Expected a strategy-run summary list')
    entries = {row['strategy']: row for row in summary}
    names = STRATEGIES[group]
    if not all(name in entries for name in names):
        raise ValueError('Run is missing required ' + group + ' strategies')
    references = {str(Path(entries[name]['reference']).expanduser().resolve()) for name in names}
    if len(references) != 1:
        raise ValueError('Strategies must use one explicit same-input reference')
    reference_path = Path(references.pop())
    reference = _archive(reference_path, STRATEGY_FREQUENCIES)
    reference_stats_path = reference_path.with_name('stats.json')
    reference_stats = _json(reference_stats_path)
    if reference_stats.get('route') != 'independent_mixed_ufl':
        raise ValueError('Reference must use the independent UFL/MUMPS route')
    records = _reference_records(reference_stats, STRATEGY_FREQUENCIES)
    identity = reference_stats.get('input_sha256')
    if not identity or set(identity) != {'K', 'M', 'q', 'receivers'}:
        raise ValueError('Missing assembled reference input identity')
    sources = [summary_path, reference_path, reference_stats_path]
    costs, metrics = [], []
    for name, method in names.items():
        entry = entries[name]
        if entry.get('returncode') != 0 or entry.get('status') != 'converged':
            raise ValueError('Strategy run failed or did not converge: ' + name)
        response_path = run / name / 'receivers.npz'
        stats_path = run / name / 'stats.json'
        resources_path = run / name / 'resources.json'
        archive = _archive(response_path, STRATEGY_FREQUENCIES)
        stats, resources = _json(stats_path), _json(resources_path)
        if stats.get('input_sha256') != identity:
            raise ValueError('Strategy and reference assembled input identity mismatch: ' + name)
        if stats.get('status') != 'converged' or resources.get('returncode') != 0:
            raise ValueError('Native strategy diagnostics did not converge: ' + name)
        sources.extend((response_path, stats_path, resources_path))
        cost = dict(method=method, successful_forward=True,
                    dynamic_basis_columns=stats['dynamic_basis_columns'],
                    complete_forward_s=stats['complete_forward_s'],
                    peak_job_pss_GiB=resources['peak_job_pss_bytes'] / 2**30)
        if any(not isinstance(cost[key], (int, float)) or not np.isfinite(cost[key]) or cost[key] <= 0
               for key in ('dynamic_basis_columns', 'complete_forward_s', 'peak_job_pss_GiB')):
            raise ValueError('Missing or invalid measured strategy costs: ' + name)
        costs.append(cost)
        for index, frequency in enumerate(STRATEGY_FREQUENCIES):
            row = dict(method=method, index=index, frequency_hz=float(frequency),
                       receiver_count=3321, reference_available=True, observation_errors='')
            for field in ('E', 'H'):
                scale = np.linalg.norm(reference[field][index], axis=1)
                if np.any(scale == 0):
                    raise ValueError('Zero reference vector makes the pointwise maximum undefined')
                difference = np.linalg.norm(archive[field][index] - reference[field][index], axis=1)
                error = 100 * difference / scale
                row[f'receiver_{field}_max_pointwise_pct'] = float(error.max())
                row[f'receiver_{field}_zero_reference_points'] = 0
                row[f'receiver_{field}_max_pointwise_status'] = 'defined'
            metrics.append(row)
    provenance = _provenance(run, sources)
    metadata = dict(reference=dict(reference_available=True,
        native_reference_verified=all(row['verified'] for row in records),
        maximum_recorded_residual=max(row['residual'] for row in records),
        controller_summary=dict(per_frequency=records), native_stats=reference_stats),
        provenance=provenance)
    target = run / 'plot_inputs' / group
    target.mkdir(parents=True, exist_ok=True)
    _write_csv(target / 'comparison.csv', costs)
    _write_csv(target / 'frequency_metrics.csv', metrics)
    _write_json(target / 'comparison.json', metadata)
    _write_json(target / 'source_metadata.json', provenance)
    return target


def source_provenance(input_dir):
    """Return a fresh-run identity when supplied; frozen paper records stay untouched."""
    path = Path(input_dir) / 'source_metadata.json'
    if path.is_file():
        value = _json(path)
        if value.get('input_kind') == 'new_run':
            return value
    return None
