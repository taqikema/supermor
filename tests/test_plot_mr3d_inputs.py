"""Check MR3D plotting inputs without launching a forward calculation."""
import csv
import json

import numpy as np
import pytest

from figs.common.mr3d_inputs import normalize_run_inputs, read_measurements
from figs.case3_mr3d.plot_mr3d_section import original_section, load_sections


def campaign(tmp_path, *, repeats=3, large_dfom=False):
    rows = []
    for number in range(1, 6):
        for method in ('SUPERMOR', 'IFOM', 'RK', 'DFOM'):
            if method == 'DFOM' and number > 2 and not large_dfom:
                continue
            rows.append(dict(method=method, level=f'M{number}', cells=str(number * 100),
                real_block_dof=str(number * 240),
                successful_repeats=str(repeats if method in ('SUPERMOR', 'IFOM') else 1),
                complete_forward_s=f'{number}.123456789123',
                peak_pss_gib=f'{number}.987654321987', max_independent_residual='1e-8'))
    write_campaign(tmp_path, rows)
    return rows


def write_campaign(folder, rows):
    with (folder / 'performance.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    planned, measured = [], []
    for row in rows:
        for repeat in range(1, int(row['successful_repeats']) + 1):
            key = dict(method=row['method'], level=row['level'], repeat=repeat)
            planned.append(key)
            measured.append(dict(key, status='passed', complete_forward_s=float(row['complete_forward_s']),
                peak_job_pss_bytes=float(row['peak_pss_gib']) * 2**30,
                max_independent_residual=1e-8))
    (folder / 'campaign.json').write_text(json.dumps(dict(
        runs=planned, mpi_ranks=8, threads_per_rank=1)))
    (folder / 'summary.json').write_text(json.dumps(dict(runs=measured, groups=rows,
        measurement_implementation='portable_release_v1',
        historical_performance_reproduced=False, time_aggregation='median',
        memory_aggregation='maximum')))


def test_run_normalization_preserves_values_missing_data_and_source_files(tmp_path):
    original = campaign(tmp_path)
    source_before = (tmp_path / 'performance.csv').read_bytes()
    summary_before = (tmp_path / 'summary.json').read_bytes()
    normalized = normalize_run_inputs(tmp_path)
    rows, values, _ = read_measurements(normalized, paper=False)
    assert normalized == tmp_path / 'plot_inputs/performance.csv'
    assert len(rows) == 20
    assert sum(value is not None for value in values.values()) == 17
    source_values = {(row['method'], row['level']): row for row in original}
    labels = {'SUMOR': 'SUPERMOR', 'CPC': 'RK', 'DFOM': 'DFOM', 'IFOM': 'IFOM'}
    for row in rows:
        if row['measurement_status'] == 'not_measured':
            assert row['method'] == 'DFOM' and row['level'] in ('M3', 'M4', 'M5')
            assert row['complete_forward_s'] == row['peak_pss_gib'] == ''
        else:
            source = source_values[labels[row['method']], row['level']]
            for field in ('complete_forward_s', 'peak_pss_gib'):
                assert row[field] == source[field]
    assert (tmp_path / 'performance.csv').read_bytes() == source_before
    assert (tmp_path / 'summary.json').read_bytes() == summary_before
    provenance = json.loads(normalized.with_name('source_metadata.json').read_text())
    assert provenance['data_origin'] == 'new_run'
    assert provenance['measurement_implementation'] == 'portable_release_v1'
    assert provenance['historical_performance_reproduced'] is False
    assert provenance['source_inputs']['performance']['sha256']


def test_actual_repeat_counts_are_not_relabelled_as_paper_measurements(tmp_path):
    campaign(tmp_path, repeats=1)
    normalized = normalize_run_inputs(tmp_path)
    rows, _, _ = read_measurements(normalized, paper=False)
    assert all(row['repeats'] == '1' for row in rows if row['measurement_status'] == 'measured')
    with pytest.raises(ValueError, match='three repeats'):
        read_measurements(normalized, paper=True)


def test_optional_large_dfom_measurements_are_retained(tmp_path):
    campaign(tmp_path, large_dfom=True)
    normalized = normalize_run_inputs(tmp_path)
    _, values, _ = read_measurements(normalized, paper=False)
    assert all(value is not None for value in values.values())
    with pytest.raises(ValueError, match='seventeen measured'):
        read_measurements(normalized, paper=True)


def test_partial_plotted_campaign_is_rejected(tmp_path):
    rows = campaign(tmp_path)
    rows = [row for row in rows if (row['method'], row['level']) != ('RK', 'M4')]
    write_campaign(tmp_path, rows)
    with pytest.raises(ValueError, match='incomplete.*CPC.*M4'):
        normalize_run_inputs(tmp_path)


def test_cross_method_mesh_mismatch_is_rejected(tmp_path):
    rows = campaign(tmp_path)
    rows[1]['real_block_dof'] = '242'
    write_campaign(tmp_path, rows)
    with pytest.raises(ValueError, match='same mesh'):
        normalize_run_inputs(tmp_path)


@pytest.mark.parametrize('bad_value', ['nan', 'inf', '0', '-1'])
def test_invalid_measurement_never_becomes_a_curve(tmp_path, bad_value):
    rows = campaign(tmp_path)
    rows[0]['complete_forward_s'] = bad_value
    write_campaign(tmp_path, rows)
    with pytest.raises(ValueError, match='finite, positive'):
        normalize_run_inputs(tmp_path)
    assert not (tmp_path / 'plot_inputs/performance.csv').exists()


def test_duplicate_aliases_are_rejected(tmp_path):
    rows = campaign(tmp_path)
    extra = dict(rows[0], method='SUMOR')
    write_campaign(tmp_path, rows + [extra])
    with pytest.raises(ValueError, match='duplicate|unique known'):
        normalize_run_inputs(tmp_path)


def test_failed_planned_run_is_rejected(tmp_path):
    campaign(tmp_path)
    path = tmp_path / 'summary.json'
    summary = json.loads(path.read_text())
    summary['runs'][0]['status'] = 'failed'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='failed or unfinished'):
        normalize_run_inputs(tmp_path)


def test_missing_planned_repeat_is_rejected_even_when_all_csv_groups_exist(tmp_path):
    campaign(tmp_path)
    path = tmp_path / 'summary.json'
    summary = json.loads(path.read_text())
    summary['runs'].pop(0)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='incomplete planned repetitions'):
        normalize_run_inputs(tmp_path)


def test_stale_csv_measurement_is_rejected(tmp_path):
    campaign(tmp_path)
    path = tmp_path / 'performance.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows[0]['complete_forward_s'] = '123456'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='does not match measured run records'):
        normalize_run_inputs(tmp_path)


def test_new_measurements_outside_paper_ranges_are_visible(tmp_path):
    from figs.case3_mr3d.plot_sumor_mr3d_scaling_four_methods import draw

    run = tmp_path / 'run'
    run.mkdir()
    campaign(run, repeats=1)
    source = normalize_run_inputs(run)
    metadata = draw(source, tmp_path / 'rendered', paper=False)
    assert metadata['data_origin'] == 'new_run'
    assert metadata['typography_check']['text_within_canvas']
    for axis, metric in zip(metadata['axis_limits'], ('time_s', 'peak_memory_GiB')):
        observed = [value for curve in metadata['curves'].values() for value in curve[metric]]
        assert axis['y'][0] < min(observed) <= max(observed) < axis['y'][1]
    assert set(metadata['repetitions'].values()) == {1}


def test_missing_section_input_reports_its_path(tmp_path):
    missing = tmp_path / 'Horizontal_resistivity.sgy'
    with pytest.raises(FileNotFoundError, match='Horizontal_resistivity.sgy'):
        load_sections(missing, tmp_path / 'vertical.sgy', tmp_path / 'survey.npz')


def test_segy_section_uses_log_interpolation_and_actual_negative_depths(tmp_path):
    import segyio

    path = tmp_path / 'two_lines.sgy'
    depths = np.arange(60., 6201., 5.)
    spec = segyio.spec()
    spec.samples = depths
    spec.format = 5
    spec.tracecount = 6
    # Unsorted x traces make the coordinate/data sorting observable.
    xs = np.array([25, -25, 0])
    with segyio.create(str(path), spec) as file:
        for line in range(2):
            for ix, x in enumerate(xs):
                trace = line * 3 + ix
                file.header[trace] = {
                    segyio.TraceField.INLINE_3D: line + 1,
                    segyio.TraceField.CDP_X: int(390275 + x),
                    segyio.TraceField.CDP_Y: int(7518065 + (-300, -225)[line]),
                    segyio.TraceField.SourceGroupScalar: 1,
                    segyio.TraceField.DelayRecordingTime: 60,
                }
                file.trace[trace] = np.asarray(
                    10. ** (line + 1. + x / 1000. + (depths - 60.) / 10000.),
                    dtype=np.float32)
    x, z, section, metadata = original_section(path, -253.)
    np.testing.assert_array_equal(x, [-25., 0., 25.])
    np.testing.assert_array_equal(z, -depths[::-1])
    expected = 1. + 47. / 75. + x[:, None] / 1000. + (depths[::-1][None, :] - 60.) / 10000.
    np.testing.assert_allclose(section, expected, rtol=0, atol=4e-7)
    assert metadata['bracketing_y_m'] == [-300., -225.]
    assert metadata['shape_x_z'] == [3, 1229]
    assert metadata['upper_line_weight'] == 47. / 75.
