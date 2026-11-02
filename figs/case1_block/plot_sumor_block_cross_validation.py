#!/usr/bin/env python3
"""Reproduce the supplied compact-v2 block-validation figures from local data.

The source figures use Matplotlib's DejaVu Sans defaults.  This entry point
preserves that supplied layout, including the fixed phase limits, compact
equal-aspect maps and original full-range colour scales.  It needs only NumPy
and Matplotlib; no solver, campaign module or external drive is imported.
Default redraws are written beside this script; frozen numerical inputs
are read from outputs/case1_block/cross_validation/paper/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figs.common.paths import fingerprint
from figs.common.block_inputs import cross_validation_inputs, source_provenance

DEFAULT_DATA = ROOT / 'outputs/case1_block/cross_validation/paper'
DEFAULT_OUTPUT = Path(__file__).resolve().parent
FREQUENCIES = np.array([.1, 1., 10., 100.])
COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
PAIRS = ('supermor_vs_dfom', 'dfom_vs_emg3d')
AXES = {
    'Ex': {'amplitude': [0., 3.2e-8], 'phase_deg': [5., 30.], 'unit': 'V/m', 'exponent': -8},
    'Hy': {'amplitude': [0., 1e-6], 'phase_deg': [130., 145.], 'unit': 'A/m', 'exponent': -7},
}


def save(figure, output, stem, *, png_dpi):
    outputs = {}
    for extension in ('pdf', 'png'):
        path = output / f'{stem}.{extension}'
        kwargs = {'dpi': png_dpi} if extension == 'png' else {}
        figure.savefig(path, bbox_inches='tight', **kwargs)
        outputs[extension] = fingerprint(path)
        print(path)
    return outputs


def plot_maps(data_dir, output_dir, pair):
    source = data_dir / pair
    data_path, summary_path = source / 'pointwise_data.npz', source / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    with np.load(data_path, allow_pickle=False) as data:
        expected_receivers = np.column_stack((
            np.tile(np.arange(1000., 5001., 50.), 41),
            np.repeat(np.arange(-1000., 1001., 50.), 81), -np.ones(3321)))
        if (bool(data['synthetic'])
                or not np.array_equal(data['frequencies_hz'], FREQUENCIES)
                or not np.array_equal(data['components'], COMPONENTS)
                or not np.array_equal(data['receiver_xyz'], expected_receivers)
                or data['pointwise_nrmsd_percent'].shape != (4, 3321, 6)):
            raise ValueError('Requires complete four-frequency response maps')
        values = [data['pointwise_nrmsd_percent'][:, :, i].copy() for i in (0, 4)]
    scale = summary['pointwise_scale']
    upper = float(scale['maximum_percent'])
    if (scale['minimum_percent'] != 0. or scale['clipped']
            or not np.isfinite(values).all() or np.any(np.asarray(values) < 0)
            or float(np.max(values)) != upper):
        raise ValueError('Requires the original, defined and unclipped Ex/Hy colour scale')
    target = output_dir
    target.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(16.5, 4.3), sharex=True, sharey=True)
    figure.subplots_adjust(left=.065, right=.9, top=.94, bottom=.14, wspace=.12, hspace=.12)
    x, y = np.arange(1000., 5001., 50.), np.arange(-1000., 1001., 50.)
    cmap = plt.get_cmap('viridis').with_extremes(bad='#b8b8b8')
    norm = Normalize(0., upper)
    for ci, component in enumerate(('Ex', 'Hy')):
        for fi, frequency in enumerate(FREQUENCIES):
            axis = axes[ci, fi]
            image = axis.pcolormesh(x, y, values[ci][fi].reshape(41, 81),
                                    shading='nearest', cmap=cmap, norm=norm, rasterized=True)
            axis.set(xlim=(1000, 5000), ylim=(-1000, 1000), aspect='equal',
                     xticks=[1000, 2000, 3000, 4000, 5000], yticks=[-1000, 0, 1000])
            axis.tick_params(labelsize=9)
            if ci == 0:
                axis.set_title(f'{frequency:g} Hz', fontsize=13)
            else:
                axis.set_xlabel('x (m)')
            if fi == 0:
                axis.set_ylabel(component + '\ny (m)')
    color_axis = figure.add_axes([.915, .15, .012, .78])
    figure.colorbar(image, cax=color_axis).set_label('pointwise NRMSD (%)')
    stem = 'fig03_dfom_vs_emg3d' if pair == 'dfom_vs_emg3d' else 'fig04_supermor_vs_dfom'
    outputs = save(figure, target, stem, png_dpi=200)
    plt.close(figure)
    return {
        'inputs': [fingerprint(data_path), fingerprint(summary_path)],
        'components': ['Ex', 'Hy'], 'frequencies_hz': FREQUENCIES.tolist(),
        'receiver_count_per_panel': 3321,
        'formula': '200 * abs(a-b) / (abs(a)+abs(b)); complex scalar components',
        'color_scale_percent': [0., upper], 'colorbar_label': 'pointwise NRMSD (%)',
        'data_values_unchanged': True, 'overall_title': False,
        'subtitle': False, 'footnote': False,
        'layout': {'figsize_inches': [16.5, 4.3], 'wspace': .12, 'hspace': .12, 'aspect': 'equal'},
        'outputs': outputs,
    }


def plot_crossline(data_dir, output_dir):
    source = data_dir / 'crossline_10hz/line_responses.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    y = np.array([float(row['y_m']) for row in rows])
    if (not np.array_equal(y, np.arange(-1000., 1001., 50.))
            or any(float(row['x_m']) != 3000. or float(row['z_m']) != -1.
                   or float(row['frequency_hz']) != 10. or row['synthetic'] != 'False'
                   for row in rows)):
        raise ValueError('Requires the original 41-point x=3000 m, 10 Hz line')
    target = output_dir
    target.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11.7, 7.8), sharex=True)
    figure.subplots_adjust(left=.10, right=.98, bottom=.13, top=.83, hspace=.34, wspace=.27)
    styles = [dict(color='#228b22', linestyle='-', marker='^'),
              dict(color='#e87516', linestyle='--', marker='s')]
    extrema = {}
    for ri, (component, settings) in enumerate(AXES.items()):
        extrema[component] = {}
        for role, label, style in zip(('dfom', 'emg3d'), ('DFOM', 'emg3d'), styles):
            samples = np.array([complex(float(row[f'{role}_{component}_real']),
                                        float(row[f'{role}_{component}_imag'])) for row in rows])
            values = {'amplitude': np.abs(samples), 'phase_deg': np.angle(samples, deg=True)}
            for ci, (kind, response) in enumerate(values.items()):
                low, high = settings[kind]
                if not np.isfinite(response).all() or (response < low).any() or (response > high).any():
                    raise ValueError('Fixed axes must include every original sample')
                axes[ri, ci].plot(y, response, label=label, linewidth=1.4, markersize=4.3,
                                 markerfacecolor='none', **style)
            extrema[component][role] = {
                kind: [float(value.min()), float(value.max())] for kind, value in values.items()}
        for ci, kind in enumerate(('amplitude', 'phase_deg')):
            panel = axes[ri, ci]
            panel.set(title=component + (' amplitude' if ci == 0 else ' phase'),
                      ylabel='Amplitude (' + settings['unit'] + ')' if ci == 0 else 'Phase (degrees)',
                      ylim=settings[kind], xlim=(-1000, 1000), xticks=[-1000, -500, 0, 500, 1000])
            if ci == 0:
                panel.ticklabel_format(axis='y', style='sci',
                                       scilimits=(settings['exponent'], settings['exponent']),
                                       useMathText=True)
            else:
                panel.set_yticks(np.arange(settings[kind][0], settings[kind][1] + 1., 5.))
            panel.grid(alpha=.25)
            panel.legend(fontsize=9)
            if ri == 1:
                panel.set_xlabel('y (m)')
    outputs = save(figure, target, 'fig02_crossline_10hz', png_dpi=220)
    plt.close(figure)
    return {
        'input': fingerprint(source), 'frequency_hz': 10., 'point_count_per_curve': 41,
        'axes': AXES, 'actual_extrema': extrema,
        'interpolation': False, 'smoothing': False, 'clipped_points': 0,
        'overall_title': False, 'footnote': False, 'outputs': outputs,
    }


def draw(data_dir=DEFAULT_DATA, output_dir=DEFAULT_OUTPUT):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    # Match the supplied compact-v2 figures, which intentionally use the
    # source's DejaVu defaults rather than the manuscript's shared Times style.
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {pair: plot_maps(data_dir, output_dir, pair) for pair in PAIRS}
    metadata['crossline_10hz'] = plot_crossline(data_dir, output_dir)
    metadata['script'] = fingerprint(__file__)
    metadata['style_policy'] = 'Preserve the explicitly supplied compact-v2 DejaVu Sans layout and axes.'
    provenance = data_dir / 'source_metadata.json'
    if provenance.is_file():
        metadata['source_metadata'] = fingerprint(provenance)
    run_provenance = source_provenance(data_dir)
    for key, stem in (('crossline_10hz', 'fig02_crossline_10hz'),
                      ('dfom_vs_emg3d', 'fig03_dfom_vs_emg3d'),
                      ('supermor_vs_dfom', 'fig04_supermor_vs_dfom')):
        record = {**metadata[key], 'script': metadata['script'],
                  'style_policy': metadata['style_policy']}
        if run_provenance:
            record['run_provenance'] = run_provenance
        (output_dir / (stem + '_metadata.json')).write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input-dir', type=Path)
    source.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = cross_validation_inputs(args.run_dir) if args.run_dir else (args.input_dir or DEFAULT_DATA)
    draw(inputs, args.output_dir)


if __name__ == '__main__':
    main()
