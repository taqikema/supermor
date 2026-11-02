#!/usr/bin/env python3
"""Render Figure 11 directly from the original MR3D SEG-Y resistivity volumes.

Only the two y lines bracketing the source plane are read from each full model.
No finite-element grid or benchmark summary is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from figs.common.paths import fingerprint
from figs.common.sumor_paper_figure_style import TEXT_PT, configure, finalize, specification

TRANSLATION = np.array([-390275., -7518065., 0.])
DEFAULT_RAW = ROOT / 'data/raw_data/case3_mr3d'
DEFAULT_SURVEY = ROOT / 'data/mesh_data/case3_mr3d/prepared/survey.npz'
DEFAULT_OUTPUT = Path(__file__).resolve().parent

GEOLOGICAL_LABELS = (
    {'label': 'Undulating seafloor', 'target_km': [-8., -.6991],
     'text_km': [-4.8, -.27]},
    {'label': 'Sedimentary layers', 'target_km': [-5., -1.7],
     'text_km': [-5.8, -1.27]},
    {'label': 'Reservoir', 'target_km': [1.0054, -2.645],
     'text_km': [4.8, -1.95]},
    {'label': 'Salt body', 'target_km': [-9., -3.87],
     'text_km': [0., -4.85]},
)

def scaled_coordinates(values, scalar):
    values = np.asarray(values, dtype=float)
    scalar = np.asarray(scalar, dtype=float)
    factor = np.where(scalar > 0, scalar,
                      np.where(scalar < 0, 1. / np.maximum(abs(scalar), 1), 1.))
    return values * factor

def original_section(path, section_y):
    """Interpolate log10 resistivity between adjacent native 75 m y lines."""
    import segyio

    with segyio.open(str(path), 'r', ignore_geometry=True) as segy:
        first = segy.header[0][segyio.TraceField.INLINE_3D]
        last = segy.header[segy.tracecount - 1][segyio.TraceField.INLINE_3D]
        line_count = abs(last - first) + 1
        if segy.tracecount % line_count:
            raise ValueError('Original SEG-Y must contain complete inline rows')
        width = segy.tracecount // line_count
        starts = np.arange(0, segy.tracecount, width)
        native_y = np.array([
            scaled_coordinates(segy.header[int(start)][segyio.TraceField.CDP_Y],
                               segy.header[int(start)][segyio.TraceField.SourceGroupScalar])
            + TRANSLATION[1] for start in starts])
        order = np.argsort(native_y)
        ys = native_y[order]
        if not np.allclose(np.diff(ys), 75.):
            raise ValueError('Expected the original model with 75 m y spacing')
        upper = int(np.searchsorted(ys, section_y))
        if not 0 < upper < len(ys):
            raise ValueError('Section plane must lie inside the material model')
        rows, axes = [], []
        for iy in (upper - 1, upper):
            start = int(starts[order[iy]])
            stop = start + width
            scalar = segy.attributes(segyio.TraceField.SourceGroupScalar)[start:stop]
            x = scaled_coordinates(segy.attributes(segyio.TraceField.CDP_X)[start:stop],
                                   scalar) + TRANSLATION[0]
            y = scaled_coordinates(segy.attributes(segyio.TraceField.CDP_Y)[start:stop],
                                   scalar) + TRANSLATION[1]
            if not np.all(y == ys[iy]):
                raise ValueError('Original SEG-Y y lines must be contiguous')
            ix = np.argsort(x)
            axes.append(x[ix])
            values = segy.trace.raw[start:stop][ix]
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError('Resistivity samples must be finite and positive')
            rows.append(np.log10(values))
        if not np.array_equal(axes[0], axes[1]):
            raise ValueError('Bracketing SEG-Y lines have different x samples')
        depth = np.asarray(segy.samples, dtype=float)
        if not np.allclose(np.diff(depth), 5.) or depth[0] != 60. or depth[-1] != 6200.:
            raise ValueError('Expected original actual depth samples 60:5:6200 m')
        weight = float((section_y - ys[upper - 1]) / (ys[upper] - ys[upper - 1]))
        section = (1. - weight) * rows[0] + weight * rows[1]
        metadata = {
            'bracketing_y_m': ys[upper - 1:upper + 1].tolist(),
            'upper_line_weight': weight,
            'depth_samples_m': [float(depth[0]), float(depth[-1])],
            'shape_x_z': list(section.shape),
            'resistivity_range_ohm_m': [float(10. ** section.min()),
                                       float(10. ** section.max())]}
        return axes[0], -depth[::-1], section[:, ::-1], metadata

def save_figure(fig, stem):
    finalize(fig)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(stem.with_suffix('.' + suffix), dpi=320, facecolor='white',
                    bbox_inches=None, metadata={'Creator': Path(__file__).name})

def model_figure(x, z, horizontal, vertical, receivers, source, output):
    import matplotlib.pyplot as plt

    if not np.isclose(source[1], -253.):
        raise ValueError('Geological annotation targets are verified at y=-253 m')
    fig, axes = plt.subplots(1, 2, figsize=(15.92 / 2.54, 3.1), sharey=True,
                             constrained_layout=True)
    for index, (ax, section) in enumerate(zip(axes, (horizontal, vertical))):
        field = ax.pcolormesh(x / 1000., z / 1000., section.T, shading='nearest',
                              cmap='turbo', vmin=-1., vmax=4., rasterized=True)
        # Data start at 60 m inside seawater; only this display gap is filled.
        water_log = float(np.median(section[:, -1]))
        ax.axhspan((z[-1] + 2.5) / 1000., 0.,
                   facecolor=field.cmap(field.norm(water_log)), zorder=0)
        ax.scatter(receivers[:, 0] / 1000., receivers[:, 2] / 1000.,
                   s=4, color='#e64b35', zorder=4, label='Receivers')
        ax.scatter(source[0] / 1000., source[2] / 1000., marker='*', s=105,
                   color='#ffe24d', edgecolor='black', linewidth=.7,
                   zorder=5, label='Source')
        label = r'(a) $\rho_h$' if index == 0 else r'(b) $\rho_v$'
        ax.set_title(label, loc='left', pad=8, color='black')
        ax.set(xlabel=r'$x$ / km', xlim=(-12., 11.4), ylim=(-6.2, 0.))
        ax.set_aspect(3.2)
        ax.tick_params(labelsize=TEXT_PT)
        for item in GEOLOGICAL_LABELS:
            ax.annotate(item['label'], xy=item['target_km'],
                        xytext=item['text_km'], ha='center', va='center',
                        fontsize=TEXT_PT, color='#171717', zorder=6,
                        bbox={'boxstyle': 'round,pad=.22', 'facecolor': 'white',
                              'edgecolor': 'none', 'alpha': .92},
                        arrowprops={'arrowstyle': '->', 'color': '#171717',
                                    'linewidth': .85, 'shrinkA': 3, 'shrinkB': 1})
    axes[0].set_ylabel(r'$z$ / km')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], loc='lower left', fontsize=TEXT_PT,
                   framealpha=.9, borderpad=.4, handletextpad=.4)
    bar = fig.colorbar(field, ax=axes, shrink=.78, pad=.015)
    bar.set_ticks([-1, 0, 1, 2, 3, 4],
                  labels=['0.1', '1', '10', '100', '1000', '10000'])
    bar.set_label(r'Resistivity / $\Omega\,\mathrm{m}$')
    bar.ax.tick_params(labelsize=TEXT_PT)
    save_figure(fig, output / 'fig11_mr3d_section')
    plt.close(fig)

def load_sections(horizontal_sgy, vertical_sgy, survey_path):
    """Read the unchanged paper interpolation and source/receiver geometry."""
    paths = [Path(horizontal_sgy), Path(vertical_sgy), Path(survey_path)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Figure 11 input: {path.resolve()}")
    with np.load(paths[2], allow_pickle=False) as survey:
        receivers = survey['inline'].copy()
        source = survey['source_endpoints'].mean(axis=0)
    if receivers.shape != (204, 3) or not np.allclose(receivers[:, 1], source[1]):
        raise ValueError('Expected 204 inline receivers in the source section plane')
    horizontal = original_section(paths[0], float(source[1]))
    vertical = original_section(paths[1], float(source[1]))
    if not all(np.array_equal(horizontal[i], vertical[i]) for i in (0, 1)):
        raise ValueError('Horizontal and vertical material coordinates differ')
    return horizontal, vertical, receivers, source


def draw(horizontal_sgy=DEFAULT_RAW / 'Horizontal_resistivity.sgy',
         vertical_sgy=DEFAULT_RAW / 'Vertical_Resistivity.sgy',
         survey=DEFAULT_SURVEY, output_dir=DEFAULT_OUTPUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    horizontal_sgy, vertical_sgy, survey = (Path(p).expanduser().resolve()
        for p in (horizontal_sgy, vertical_sgy, survey))
    output_dir = Path(output_dir).expanduser().resolve()
    horizontal, vertical, receivers, source = load_sections(horizontal_sgy, vertical_sgy, survey)
    configure()
    plt.rcParams.update({'axes.linewidth': .8})
    output_dir.mkdir(parents=True, exist_ok=True)
    model_figure(horizontal[0], horizontal[1], horizontal[2], vertical[2],
                 receivers, source, output_dir)
    metadata = {
        'paper_figure': 11, 'script': fingerprint(__file__),
        'inputs': {name: fingerprint(path) for name, path in
                   (('horizontal_sgy', horizontal_sgy), ('vertical_sgy', vertical_sgy),
                    ('survey', survey))},
        'section': {'y_m': float(source[1]), 'source_m': source.tolist(),
                    'inline_receivers': len(receivers),
                    'method': 'linear interpolation of log10 resistivity between native y lines',
                    'native_spacing_m': [25, 75, 5],
                    'top_water_display_fill_m': [0., 57.5],
                    'horizontal': horizontal[3], 'vertical': vertical[3],
                    'geological_annotations': list(GEOLOGICAL_LABELS),
                    'color_log10_limits': [-1, 4], 'vertical_exaggeration': 3.2},
        'sources': {'material': 'https://zenodo.org/records/400233'},
        'typography': specification(),
        'outputs': {suffix: fingerprint(output_dir / f'fig11_mr3d_section.{suffix}')
                    for suffix in ('png', 'pdf', 'svg')}}
    (output_dir / 'fig11_mr3d_section_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizontal-sgy', type=Path,
                        default=DEFAULT_RAW / 'Horizontal_resistivity.sgy')
    parser.add_argument('--vertical-sgy', type=Path,
                        default=DEFAULT_RAW / 'Vertical_Resistivity.sgy')
    parser.add_argument('--survey', type=Path, default=DEFAULT_SURVEY)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = draw(args.horizontal_sgy, args.vertical_sgy, args.survey, args.output_dir)
    print(json.dumps({'output_dir': str(args.output_dir.resolve()),
                      'section_y_m': result['section']['y_m']}, indent=2))


if __name__ == '__main__':
    main()
