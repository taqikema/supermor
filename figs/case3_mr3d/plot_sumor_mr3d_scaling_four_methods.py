#!/usr/bin/env python3
"""Render frozen paper or new MR3D measurements with shared method styles.

The CSV retains DFOM's two measured configurations for the paper table.
Only the three methods with all five measured meshes are plotted.  Empty
DFOM configurations remain unmeasured; they are never filled or estimated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from figs.common.method_styles import STYLES
from figs.common.sumor_paper_figure_style import configure, finalize, specification
from figs.common.paths import fingerprint
from figs.common.mr3d_inputs import read_measurements, normalize_run_inputs

DEFAULT_INPUT = ROOT / 'outputs/case3_mr3d/mesh_scaling/paper/performance.csv'
DEFAULT_OUTPUT = Path(__file__).resolve().parent
LEVELS = ('M1', 'M2', 'M3', 'M4', 'M5')
PLOTTED_METHODS = ('SUMOR', 'IFOM', 'CPC')


def typography_check(figure):
    texts = [item for item in figure.findobj(match=Text)
             if item.get_visible() and item.get_text()]
    families = {tuple(item.get_fontfamily()) for item in texts}
    sizes = {float(item.get_fontsize()) for item in texts}
    if families != {('Times New Roman',)} or sizes != {7.}:
        raise ValueError(f'Unexpected typography: {families}, {sizes}')
    width, height = figure.get_size_inches() * figure.dpi
    renderer = figure.canvas.get_renderer()
    clipped = []
    for item in texts:
        bbox = item.get_window_extent(renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > width + 1 or bbox.y1 > height + 1:
            clipped.append(item.get_text())
    if clipped:
        raise ValueError(f'Text falls outside the figure: {clipped}')
    return {'font_families': [list(x) for x in sorted(families)],
            'font_sizes_pt': sorted(sizes), 'text_within_canvas': True}


def draw(source=DEFAULT_INPUT, output=DEFAULT_OUTPUT, *, paper=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    rows, values, meshes = read_measurements(source, paper=paper)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    fig, axes = plt.subplots(1, 2, figsize=(15.92 / 2.54, 3.05))
    fig.subplots_adjust(left=.084, right=.988, bottom=.22, top=.875, wspace=.285)
    x = [meshes[level][1] / 1e6 for level in LEVELS]
    for method in PLOTTED_METHODS:
        label, color, marker = STYLES[method]
        for axis, metric in zip(axes, ('complete_forward_s', 'peak_pss_gib')):
            axis.plot(x, [values[method, level][metric] for level in LEVELS],
                      color=color, marker=marker, linestyle='-', label=label,
                      linewidth=1.3, markersize=3.5, markeredgewidth=.85)
    axes[0].set(ylabel='Complete forward time / s', ylim=(10, 10000))
    axes[1].set(ylabel='Peak memory / GiB', ylim=(4, 800))
    reference_lines = []
    for value, linestyle in ((32, '--'), (64, ':')):
        reference_lines.append(axes[1].axhline(
            value, color='#777777', linestyle=linestyle, linewidth=.85,
            label=f'{value} GiB RAM', zorder=1))
    axes[1].legend(handles=reference_lines, loc='upper right',
                   bbox_to_anchor=(.99, .70), frameon=True, facecolor='white',
                   edgecolor='none', framealpha=.95, handlelength=2.1,
                   borderpad=.25, labelspacing=.35)
    for axis, label in zip(axes, ('(a)', '(b)')):
        axis.set_yscale('log')
        axis.set_xlabel(r'Real edge DOF / $10^6$')
        axis.set_xlim(0, 20)
        axis.set_xticks(x, [f'{value:.2f}\n{level}' for value, level in zip(x, LEVELS)])
        axis.set_title(label, loc='left', pad=5)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which='major', color='#b4b4b4', alpha=.28, linewidth=.55)
        axis.set_axisbelow(True)
        axis.tick_params(direction='out', length=3, width=.7)
        for spine in axis.spines.values():
            spine.set_linewidth(.7)
    axes[0].set_yticks([10, 100, 1000, 10000])
    axes[1].set_yticks([4, 8, 16, 32, 64, 128, 256, 512])
    if not paper:
        # New hardware measurements may lie outside the frozen paper limits.
        for axis, metric in zip(axes, ('complete_forward_s', 'peak_pss_gib')):
            measured = [values[method, level][metric]
                        for method in PLOTTED_METHODS for level in LEVELS]
            bounds = measured + ([32., 64.] if metric == 'peak_pss_gib' else [])
            low, high = min(bounds) * .75, max(bounds) * 1.35
            ticks = [base * 10. ** exponent
                     for exponent in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
                     for base in (1., 2., 5.) if low <= base * 10. ** exponent <= high]
            axis.set_yticks(ticks)
            axis.set_ylim(low, high)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.),
               ncol=3, frameon=False, handlelength=2.1, columnspacing=1.8)
    finalize(fig)
    type_check = typography_check(fig)
    outputs = {}
    for suffix in ('png', 'pdf', 'svg'):
        path = output / f'fig12_mesh_scaling.{suffix}'
        fig.savefig(path, dpi=320, facecolor='white', bbox_inches=None)
        outputs[suffix] = fingerprint(path)
        print(path)
    summary = {'speedup_over_SUPERMOR': {}, 'memory_reduction_SUPERMOR_pct': {}}
    for method in ('IFOM', 'CPC', 'DFOM'):
        label = STYLES[method][0]
        summary['speedup_over_SUPERMOR'][label] = {
            level: values[method, level]['complete_forward_s'] /
                   values['SUMOR', level]['complete_forward_s']
            for level in LEVELS if values[method, level] is not None}
        summary['memory_reduction_SUPERMOR_pct'][label] = {
            level: 100 * (1 - values['SUMOR', level]['peak_pss_gib'] /
                          values[method, level]['peak_pss_gib'])
            for level in LEVELS if values[method, level] is not None}
    summary['SUPERMOR_memory_margin_GiB'] = {
        '32GiB_at_M3': 32 - values['SUMOR', 'M3']['peak_pss_gib'],
        '64GiB_at_M5': 64 - values['SUMOR', 'M5']['peak_pss_gib']}
    metadata = {
        'description': 'MR3D grid-scaling comparison; DFOM is table-only',
        'paper_figure': 12, 'script': fingerprint(__file__),
        'input': fingerprint(source), 'input_rows': rows,
        'measured_configurations': sum(value is not None for value in values.values()),
        'unmeasured_configurations': [f'{method} {level}' for (method, level), value in values.items() if value is None],
        'data_origin': 'frozen_paper' if paper else 'new_run',
        'data_policy': ('Frozen paper measurements; no missing values estimated' if paper else
                        'New campaign measurements; aliases normalized, numeric values retained, no missing values estimated'),
        'method_styles': {method: {'label': STYLES[method][0], 'color': STYLES[method][1],
                                   'marker': STYLES[method][2]}
                          for method in PLOTTED_METHODS},
        'style_reference': fingerprint(ROOT / 'figs/common/method_styles.py'),
        'curves': {STYLES[method][0]: {
            'x_real_edge_dof': [meshes[level][1] for level in LEVELS],
            'time_s': [values[method, level]['complete_forward_s'] for level in LEVELS],
            'peak_memory_GiB': [values[method, level]['peak_pss_gib'] for level in LEVELS]}
            for method in PLOTTED_METHODS},
        'MPI_ranks': 8, 'threads_per_rank': 1, 'frequency_count': 19,
        'frequencies_hz': [index / 16 for index in range(2, 21)],
        'repetitions': {f"{STYLES[row['method']][0]} {row['level']}": int(row['repeats'])
                        for row in rows if row['measurement_status'] == 'measured'},
        'statistics': {'SUPERMOR_IFOM': ('median time and maximum sampled peak aggregate PSS of three repeats'
                                        if paper else 'median time and maximum sampled peak aggregate PSS of the recorded successful repeats'),
                       'RK_DFOM': 'single measured run'},
        'axis_limits': [{'x': list(axis.get_xlim()), 'y': list(axis.get_ylim()),
                         'y_scale': axis.get_yscale()} for axis in axes],
        'axis_labels': {'x': 'Real edge DOF / 10^6',
                        'y': ['Complete forward time / s', 'Peak memory / GiB']},
        'reference_lines_GiB': [32, 64], 'reference_legend_anchor': [.99, .70],
        'curve_style': {'line_style': 'solid', 'marker_size_pt': 3.5,
                        'all_five_meshes_marked': True, 'error_bars': False},
        'typography': specification(), 'typography_check': type_check,
        'summary': summary, 'outputs': outputs,
    }
    provenance = source.with_name('source_metadata.json')
    if provenance.is_file():
        metadata['source_metadata'] = fingerprint(provenance)
        if not paper:
            metadata['measurement_provenance'] = json.loads(provenance.read_text(encoding='utf-8'))
    (output / 'fig12_mesh_scaling_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    plt.close(fig)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--input-dir', type=Path,
                           help='Frozen paper input directory containing performance.csv')
    selection.add_argument('--run-dir', type=Path,
                           help='New MR3D campaign containing performance.csv, summary.json and campaign.json')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.run_dir is not None:
        source = normalize_run_inputs(args.run_dir)
    else:
        source = args.input_dir / 'performance.csv' if args.input_dir else DEFAULT_INPUT
    draw(source, args.output_dir, paper=args.run_dir is None)


if __name__ == '__main__':
    main()
