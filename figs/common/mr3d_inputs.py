"""Validate frozen MR3D measurements and normalize a new measured campaign."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics

from figs.common.paths import fingerprint

LEVELS = ('M1', 'M2', 'M3', 'M4', 'M5')
METHODS = ('DFOM', 'CPC', 'IFOM', 'SUMOR')
ALIASES = {'DFOM': 'DFOM', 'IFOM': 'IFOM', 'RK': 'CPC', 'CPC': 'CPC',
           'SUPERMOR': 'SUMOR', 'SUMOR': 'SUMOR'}
FIELDS = ('level', 'tetrahedra', 'real_edge_dof', 'method', 'complete_forward_s',
          'peak_pss_gib', 'repeats', 'time_statistic', 'memory_statistic',
          'measurement_status')


def _campaign_key(row, *, repeat=False):
    method = ALIASES.get(str(row.get('method', '')).upper())
    level = row.get('level')
    if method is None or level not in LEVELS:
        raise ValueError('Campaign contains an unknown method or mesh')
    key = method, level
    if repeat:
        number = int(row['repeat'])
        if number < 1:
            raise ValueError('Campaign repeats must be positive')
        key += (number,)
    return key


def _validate_completed_campaign(run_dir, raw, summary):
    """Reject partial runs and stale CSV summaries before normalizing them."""
    campaign_path = run_dir / 'campaign.json'
    if not campaign_path.is_file():
        raise FileNotFoundError(f'Missing MR3D campaign plan: {campaign_path}')
    campaign = json.loads(campaign_path.read_text(encoding='utf-8'))
    planned, actual = {}, {}
    for row in campaign.get('runs', ()):
        key = _campaign_key(row, repeat=True)
        if key in planned:
            raise ValueError('Campaign contains duplicate planned repetitions')
        planned[key] = row
    records = summary.get('runs')
    if not planned or not isinstance(records, list):
        raise ValueError('Campaign plan and per-run summary records are required')
    for row in records:
        if row.get('status') != 'passed':
            raise ValueError('MR3D campaign contains a failed or unfinished run')
        key = _campaign_key(row, repeat=True)
        if key in actual:
            raise ValueError('Campaign contains duplicate measured repetitions')
        actual[key] = row
    if set(planned) != set(actual):
        raise ValueError('MR3D campaign has incomplete planned repetitions')
    measured = {}
    for key, row in actual.items():
        seconds, pss = float(row['complete_forward_s']), float(row['peak_job_pss_bytes'])
        residual = float(row['max_independent_residual'])
        if not (math.isfinite(seconds) and seconds > 0 and math.isfinite(pss) and pss > 0
                and math.isfinite(residual) and 0 <= residual <= 1e-6):
            raise ValueError('Campaign runs require finite, positive measurements and accepted residuals')
        measured.setdefault(key[:2], []).append(row)
    groups = {}
    for group in summary.get('groups', ()):
        key = _campaign_key(group)
        if key in groups:
            raise ValueError('Summary contains duplicate method/mesh groups')
        groups[key] = group
    csv_groups = {}
    for row in raw:
        key = _campaign_key(row)
        if key in csv_groups:
            raise ValueError('Expected unique known method/mesh run summaries')
        csv_groups[key] = row
    if set(groups) != set(measured) or set(csv_groups) != set(measured):
        raise ValueError('CSV, summary groups and measured run configurations disagree')
    for key, records in measured.items():
        group, row = groups[key], csv_groups[key]
        expected_time = statistics.median(float(record['complete_forward_s']) for record in records)
        expected_pss = max(float(record['peak_job_pss_bytes']) for record in records) / 2**30
        if int(row['successful_repeats']) != len(records) or int(group['successful_repeats']) != len(records):
            raise ValueError('Successful repeat counts disagree with measured run records')
        for field, expected in (('complete_forward_s', expected_time), ('peak_pss_gib', expected_pss)):
            if not all(math.isclose(float(item[field]), expected, rel_tol=1e-12, abs_tol=0.)
                       for item in (row, group)):
                raise ValueError(f'CSV or summary {field} does not match measured run records')
        if any(int(row[field]) != int(group[field]) for field in ('cells', 'real_block_dof')):
            raise ValueError('CSV and summary mesh descriptions disagree')
    return campaign


def read_measurements(source, *, paper=True):
    """Read the paper schema without estimating any missing measurement.

    Frozen paper inputs retain the original repeat/count checks. A normalized
    new campaign retains its actual repeat counts and optional large DFOM runs.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f'Missing Figure 12 measurements: {source.resolve()}')
    with source.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    values, meshes = {}, {}
    for row in rows:
        method, level = row['method'], row['level']
        key = method, level
        if key in values or method not in METHODS or level not in LEVELS:
            raise ValueError('Expected unique method/mesh configurations')
        mesh = (int(row['tetrahedra']), int(row['real_edge_dof']))
        if min(mesh) <= 0 or (level in meshes and meshes[level] != mesh):
            raise ValueError('All methods must use the same mesh at each level')
        meshes[level] = mesh
        if row['measurement_status'] == 'not_measured':
            if method != 'DFOM' or level not in LEVELS[2:]:
                raise ValueError('Only DFOM M3-M5 may be unmeasured')
            if row['complete_forward_s'] or row['peak_pss_gib']:
                raise ValueError('Unmeasured configurations must have empty metrics')
            values[key] = None
            continue
        if row['measurement_status'] != 'measured':
            raise ValueError('Unexpected measurement status')
        metrics = {name: float(row[name])
                   for name in ('complete_forward_s', 'peak_pss_gib')}
        if not all(math.isfinite(value) and value > 0 for value in metrics.values()):
            raise ValueError('Time and memory must be finite, positive measurements')
        repeats = int(row['repeats'])
        if paper:
            expected_repeats = 3 if method in ('SUMOR', 'IFOM') else 1
            if repeats != expected_repeats:
                raise ValueError('Expected three repeats for SUPERMOR/IFOM, otherwise one')
        elif repeats < 1 or (method in ('CPC', 'DFOM') and repeats != 1):
            raise ValueError('Expected positive repeats and one RK/DFOM run')
        values[key] = metrics
    if set(values) != {(method, level) for method in METHODS for level in LEVELS}:
        raise ValueError('Expected all twenty measured/unmeasured configurations')
    if paper and sum(value is not None for value in values.values()) != 17:
        raise ValueError('Expected seventeen measured configurations')
    return rows, values, meshes


def normalize_run_inputs(run_dir):
    """Copy a complete release run's summaries into its own plot_inputs cache.

    Numeric strings from performance.csv are copied unchanged. Missing large
    DFOM configurations are explicitly unmeasured; all plotted methods and
    DFOM M1-M2 must exist. Solver output files are never modified.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    source, summary_path = run_dir / 'performance.csv', run_dir / 'summary.json'
    for path in (source, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing MR3D run summary: {path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    with source.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'method', 'level', 'cells', 'real_block_dof', 'successful_repeats',
                    'complete_forward_s', 'peak_pss_gib'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError('Run performance.csv does not use the MR3D release schema')
        raw = list(reader)
    campaign = _validate_completed_campaign(run_dir, raw, summary)
    rows, seen, mesh_by_level = [], set(), {}
    for row in raw:
        method = ALIASES.get(row['method'].upper())
        level = row['level']
        if method is None or level not in LEVELS or (method, level) in seen:
            raise ValueError('Expected unique known method/mesh run summaries')
        seen.add((method, level))
        mesh = int(row['cells']), int(row['real_block_dof'])
        if level in mesh_by_level and mesh_by_level[level] != mesh:
            raise ValueError('All methods must use the same mesh at each level')
        mesh_by_level[level] = mesh
        rows.append(dict(level=level, tetrahedra=row['cells'],
            real_edge_dof=row['real_block_dof'], method=method,
            complete_forward_s=row['complete_forward_s'], peak_pss_gib=row['peak_pss_gib'],
            repeats=row['successful_repeats'],
            time_statistic='median' if method in ('SUMOR', 'IFOM') else 'single',
            memory_statistic='maximum' if method in ('SUMOR', 'IFOM') else 'single',
            measurement_status='measured'))
    mandatory = {(method, level) for method in METHODS for level in LEVELS
                 if method != 'DFOM' or level in LEVELS[:2]}
    if not mandatory.issubset(seen):
        missing = sorted(mandatory - seen)
        raise ValueError(f'MR3D campaign is incomplete; missing measured configurations: {missing}')
    for level in LEVELS[2:]:
        if ('DFOM', level) not in seen:
            cells, dof = mesh_by_level[level]
            rows.append(dict(level=level, tetrahedra=str(cells), real_edge_dof=str(dof),
                method='DFOM', complete_forward_s='', peak_pss_gib='', repeats='',
                time_statistic='', memory_statistic='', measurement_status='not_measured'))
    rows.sort(key=lambda row: (LEVELS.index(row['level']), METHODS.index(row['method'])))
    cache = run_dir / 'plot_inputs'
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / 'performance.csv'
    # Validate before publishing the cache, without writing a partial CSV.
    import io
    contents = io.StringIO(newline='')
    writer = csv.DictWriter(contents, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    temporary = cache / 'performance.csv.tmp'
    try:
        temporary.write_text(contents.getvalue(), encoding='utf-8')
        read_measurements(temporary, paper=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    inputs = {'performance': fingerprint(source), 'summary': fingerprint(summary_path)}
    inputs['campaign'] = fingerprint(run_dir / 'campaign.json')
    metadata = {
        'data_origin': 'new_run', 'source_inputs': inputs,
        'measurement_implementation': summary.get('measurement_implementation'),
        'historical_performance_reproduced': False,
        'time_aggregation': summary.get('time_aggregation'),
        'memory_aggregation': summary.get('memory_aggregation'),
        'MPI_ranks': campaign.get('mpi_ranks'),
        'threads_per_rank': campaign.get('threads_per_rank'),
        'normalization': 'method/column aliases only; measurements copied without rounding; missing large DFOM remains unmeasured',
        'successful_repeats': {f"{row['method']} {row['level']}": int(row['repeats'])
                               for row in rows if row['measurement_status'] == 'measured'},
    }
    (cache / 'source_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return destination
