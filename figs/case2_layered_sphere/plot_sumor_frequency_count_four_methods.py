#!/usr/bin/env python3
"""Plot the frozen four-method frequency-count measurements for the paper.

This script only reads measurements and renders figures.  It does not launch
solvers, combine historical measurements, fit curves, or estimate crossings.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figs.common.sumor_paper_figure_style import configure, finalize, specification
from figs.common.paths import ROOT, fingerprint
from figs.common.method_styles import STYLES
from figs.common.sphere_inputs import load_frequency_inputs

DEFAULT_OUTPUT = Path(__file__).resolve().parent
COUNTS = (2, 8, 10, 16, 20, 30, 40)
EXTENSIONS = ('png', 'pdf', 'svg')


def read_measurements(source, rows=None):
    if rows is None:
        with Path(source).open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
    values = {}
    for row in rows:
        key = (row['method'], int(row['n_frequency']))
        if key in values or int(row['mpi_ranks']) != 8 or row['data_origin'] != 'measured':
            raise ValueError('Expected unique measured configurations, each with 8 MPI ranks')
        metrics = {k: float(row[k]) for k in ('complete_forward_s', 'peak_pss_gib')}
        if not all(math.isfinite(v) and v > 0 for v in metrics.values()):
            raise ValueError('Time and PSS must be finite positive measured values')
        if not math.isclose(metrics['peak_pss_gib'],
                            int(row['peak_job_pss_bytes']) / 2**30, rel_tol=1e-12):
            raise ValueError('PSS bytes and GiB disagree')
        values[key] = metrics
    expected = {(method, n) for method in STYLES for n in COUNTS}
    if set(values) != expected:
        raise ValueError('Expected exactly four methods at each of the seven frequency counts')
    return rows, values


def ratios(values, metric):
    return {STYLES[m][0]: [values[m, n][metric] / values['SUMOR', n][metric]
                          for n in COUNTS] for m in ('DFOM', 'IFOM', 'CPC')}


def measured_reversals(values, first, second):
    """Return adjacent measured counts with opposite time order, without fitting."""
    differences = [values[first, n]['complete_forward_s'] -
                   values[second, n]['complete_forward_s'] for n in COUNTS]
    return [[a, b] for a, b, da, db in
            zip(COUNTS, COUNTS[1:], differences, differences[1:]) if da * db < 0]


def axis_style(axis):
    axis.set_xlabel(r'Number of frequencies $N_f$')
    axis.set_xticks(COUNTS)
    axis.set_xlim(0, 42)
    axis.grid(True, which='major', color='#b4b4b4', alpha=.28, linewidth=.55)
    axis.tick_params(direction='out', length=3, width=.7)
    for spine in axis.spines.values():
        spine.set_linewidth(.7)


def save_figure(figure, output, stem):
    outputs = {}
    for extension in EXTENSIONS:
        path = output / f'{stem}.{extension}'
        figure.savefig(path, dpi=320, facecolor='white', bbox_inches=None)
        outputs[extension] = fingerprint(path)
        print(path)
    return outputs


def check_typography(figure):
    """Verify the final-size lettering after the shared style is applied."""
    from matplotlib.text import Text
    fonts, sizes = set(), set()
    for item in figure.findobj(match=Text):
        if item.get_text():
            fonts.add(tuple(item.get_fontfamily()))
            sizes.add(float(item.get_fontsize()))
    if fonts != {('Times New Roman',)} or sizes != {7.0}:
        raise ValueError(f'Unexpected final figure typography: {fonts}, {sizes}')
    return {'font_families': [list(x) for x in sorted(fonts)],
            'font_sizes_pt': sorted(sizes)}


def draw(source, output, *, rows=None, input_files=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    rows, values = read_measurements(source, rows=rows)
    output.mkdir(parents=True, exist_ok=True)
    configure()

    performance, axes = plt.subplots(1, 2, figsize=(15.92 / 2.54, 3.05))
    performance.subplots_adjust(left=.084, right=.988, bottom=.18, top=.875, wspace=.285)
    for method, (label, color, marker) in STYLES.items():
        for axis, metric in zip(axes, ('complete_forward_s', 'peak_pss_gib')):
            axis.plot(COUNTS, [values[method, n][metric] for n in COUNTS],
                      color=color, marker=marker, linestyle='-', label=label,
                      linewidth=1.3, markersize=3.5, markeredgewidth=.85)
    axes[0].set_yscale('log')
    axes[0].set_ylim(60, 12000)
    axes[0].set_ylabel('Complete forward time / s')
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel('Peak memory / GiB')
    for axis, panel in zip(axes, ('(a)', '(b)')):
        axis_style(axis)
        axis.set_title(panel, loc='left', pad=5)
    handles, labels = axes[0].get_legend_handles_labels()
    performance.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.0),
                       ncol=4, frameon=False, handlelength=2.1, columnspacing=1.8)
    finalize(performance)
    type_check = check_typography(performance)
    outputs = {'performance': save_figure(performance, output, 'fig09_frequency_performance')}
    limits = {'performance': [{'x': list(a.get_xlim()), 'y': list(a.get_ylim()),
                              'y_scale': a.get_yscale()} for a in axes]}
    plt.close(performance)

    time_ratios = ratios(values, 'complete_forward_s')
    relative, axis = plt.subplots(figsize=(15.92 / 2.54, 3.25))
    relative.subplots_adjust(left=.105, right=.985, bottom=.17, top=.965)
    for method in ('DFOM', 'IFOM', 'CPC'):
        label, color, marker = STYLES[method]
        axis.plot(COUNTS, time_ratios[label], color=color, marker=marker,
                  linestyle='-', label=label, linewidth=1.3,
                  markersize=3.5, markeredgewidth=.85)
    axis.axhline(1, color='#707070', linestyle='--', linewidth=.9, zorder=1)
    axis.set_yscale('log')
    axis.set_ylim(.18, 30)
    axis.set_yticks([.2, .5, 1, 2, 5, 10, 20])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    axis.set_ylabel(r'Speedup $T_j/T_{\mathrm{SUPERMOR}}$')
    axis_style(axis)
    axis.legend(loc='lower right', frameon=False, handlelength=2.1)
    finalize(relative)
    check_typography(relative)
    outputs['relative_speedup'] = save_figure(relative, output, 'fig10_relative_speedup')
    limits['relative_speedup'] = {'x': list(axis.get_xlim()), 'y': list(axis.get_ylim()),
                                  'y_scale': axis.get_yscale()}
    plt.close(relative)

    v40 = {m: values[m, 40] for m in STYLES}
    summary = {
        'speedup_at_40': {m: time_ratios[m][-1] for m in time_ratios},
        'PSS_reduction_vs_DFOM_pct_at_40':
            100 * (1 - v40['SUMOR']['peak_pss_gib'] / v40['DFOM']['peak_pss_gib']),
        'PSS_reduction_vs_RK_pct_at_40':
            100 * (1 - v40['SUMOR']['peak_pss_gib'] / v40['CPC']['peak_pss_gib']),
        'PSS_increase_vs_IFOM_pct_at_40':
            100 * (v40['SUMOR']['peak_pss_gib'] / v40['IFOM']['peak_pss_gib'] - 1),
        'SUPERMOR_time_increase_2_to_40_pct':
            100 * (v40['SUMOR']['complete_forward_s'] /
                   values['SUMOR', 2]['complete_forward_s'] - 1),
        'RK_time_increase_2_to_40_pct':
            100 * (v40['CPC']['complete_forward_s'] /
                   values['CPC', 2]['complete_forward_s'] - 1),
        'measured_time_order_reversals': {
            'IFOM_SUPERMOR': measured_reversals(values, 'IFOM', 'SUMOR'),
            'RK_DFOM': measured_reversals(values, 'CPC', 'DFOM'),
            'RK_IFOM': measured_reversals(values, 'CPC', 'IFOM'),
        },
    }
    validation_path = source.with_name('validation_summary.json')
    new_run = any('worker_provenance' in row for row in rows)
    thread_values = {row['worker_provenance'].get('threads_per_rank')
                     for row in rows if 'worker_provenance' in row}
    threads = next(iter(thread_values)) if len(thread_values) == 1 else None
    metadata = {
        'description': 'Four-method frequency-count performance and SUPERMOR speedup',
        'paper_figures': {'performance': 9, 'relative_speedup': 10},
        'script': fingerprint(__file__), 'input': fingerprint(source),
        'inputs': [fingerprint(p) for p in (input_files or [source])],
        'typography': specification(), 'typography_check': type_check,
        'frequency_counts': list(COUNTS), 'MPI_ranks': 8,
        'threads_per_rank': threads if new_run else 1,
        'input_kind': 'new_run' if new_run else 'paper',
        'kernel_selections': sorted({row['worker_provenance'].get('kernel_selection', 'unrecorded')
                                     for row in rows if 'worker_provenance' in row}),
        'method_labels': {k: v[0] for k, v in STYLES.items()},
        'measurements': rows, 'time_ratios_other_over_SUPERMOR': time_ratios,
        'PSS_ratios_other_over_SUPERMOR': ratios(values, 'peak_pss_gib'),
        'summary': summary, 'axis_limits': limits,
        'curve_style': {'line_style': 'solid', 'marker_size_pt': 3.5,
                        'all_seven_frequency_counts_marked': True,
                        'error_bars': False, 'fitted_crossings': False},
        'outputs': outputs,
        'data_policy': 'All 28 current measured configurations; no historical rows, fits or estimates',
    }
    if validation_path.is_file():
        metadata['validation'] = fingerprint(validation_path)
    for stem in ('fig09_frequency_performance', 'fig10_relative_speedup'):
        (output / (stem + '_metadata.json')).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--input-dir', type=Path, help='Frozen paper drawing inputs')
    group.add_argument('--run-dir', type=Path, help='One complete new frequency-count run')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, inputs = load_frequency_inputs(input_dir=args.input_dir, run_dir=args.run_dir)
    draw(inputs[0], args.output_dir, rows=rows, input_files=inputs)


if __name__ == '__main__':
    main()
