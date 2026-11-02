#!/usr/bin/env python3
"""Draw the current Figure 1 shifted-block geometry without a mesh.

Read the locally frozen model parameters; no FEM mesh, solver, or external
campaign module is required. The historical filename is retained because
the manuscript and reproduction index already refer to this entry point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figs.common.paths import fingerprint
from figs.common.sumor_paper_figure_style import configure, finalize, specification


DEFAULT_PARAMETERS = ROOT / 'data/figure_data/case1_block/geometry/model_parameters.json'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'fig01_block_model'
RED, BLUE, TEAL, GREY = '#c0443c', '#1676b7', '#278a98', '#5e5b51'


def read_parameters(path):
    parameters = json.loads(Path(path).read_text(encoding='utf-8'))
    physical, sampling = parameters['physical'], parameters['sampling']
    start = np.asarray(physical['source']['start_m'], dtype=float)
    end = np.asarray(physical['source']['end_m'], dtype=float)
    bounds = np.asarray(physical['block_bounds_m'], dtype=float)
    assert physical['conductivity_S_per_m'] == {'air': 1e-8, 'background': .01, 'block': 1.}
    assert physical['domain_bounds_m'] == [[-100000., 100000.]] * 3
    assert np.array_equal(start, [-2500., 0., 0.])
    assert np.array_equal(end, [-2000., 0., 0.])
    assert np.array_equal(bounds, [[1300., 3500.], [-500., 500.], [-700., -300.]])
    assert physical['source']['current_A'] == 1.
    assert sampling['x_range_m'] == [1000., 5000.]
    assert sampling['y_range_m'] == [-1000., 1000.]
    assert sampling['spacing_m'] == 50. and sampling['z_m'] == -1.
    assert sampling['shape_xy'] == [81, 41] and sampling['receiver_count'] == 3321
    assert sampling['highlighted_line'] is None
    return parameters, start, end, bounds


def typography_check(figure):
    texts = [text for text in figure.findobj(match=Text)
             if text.get_visible() and text.get_text()]
    assert {tuple(text.get_fontfamily()) for text in texts} == {('Times New Roman',)}
    assert {float(text.get_fontsize()) for text in texts} == {7.}
    renderer = figure.canvas.get_renderer()
    width, height = figure.get_size_inches() * figure.dpi
    clipped = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            clipped.append(text.get_text())
    assert not clipped, f'Text outside figure canvas: {clipped}'
    return {'font_family': 'Times New Roman', 'font_size_pt': 7.,
            'all_text_within_canvas': True}


def draw(parameters=DEFAULT_PARAMETERS, output=DEFAULT_OUTPUT, *, metadata_path=None,
         entry_script=None):
    parameters, output = Path(parameters).resolve(), Path(output).resolve()
    model, start, end, bounds = read_parameters(parameters)
    physical, sampling = model['physical'], model['sampling']
    b, start_km, end_km = bounds / 1000., start / 1000., end / 1000.
    plane_x = np.asarray(sampling['x_range_m']) / 1000.
    plane_y = np.asarray(sampling['y_range_m']) / 1000.
    plane_z = sampling['z_m'] / 1000.
    configure()
    fig, (plan, section) = plt.subplots(1, 2, figsize=(15.92 / 2.54, 3.2))
    fig.subplots_adjust(left=.079, right=.985, bottom=.16, top=.92, wspace=.28)

    plan.add_patch(Rectangle((plane_x[0], plane_y[0]), np.ptp(plane_x), np.ptp(plane_y),
                             facecolor='#e5f0f6', edgecolor=BLUE, linewidth=.85))
    plan.add_patch(Rectangle((b[0, 0], b[1, 0]), np.ptp(b[0]), np.ptp(b[1]),
                             facecolor='none', edgecolor=TEAL, linewidth=.95,
                             linestyle=(0, (4, 2.5))))
    plan.text(3., 1.13, 'Receiver plane\n3321 points, 50 m spacing',
              ha='center', va='bottom', color=BLUE, linespacing=1.15)
    plan.text(2.4, -.68, 'Conductive block\n(plan projection)',
              ha='center', va='top', color=TEAL)
    plan_source_label = plan.text(-2.1, .33, 'Line source', ha='center', va='bottom', color=RED)
    plan.text(-2.25, -.12, r'$+x$', ha='center', va='top', color=RED)
    plan.add_patch(FancyArrowPatch(start_km[:2], end_km[:2], arrowstyle='-|>',
                                   mutation_scale=6.5, linewidth=1.35, color=RED,
                                   shrinkA=0, shrinkB=0, zorder=6))
    plan.set(ylim=(-1.35, 1.65), yticks=np.arange(-1., 1.6, .5), ylabel=r'$y$ / km')

    section.axhspan(0., .35, facecolor='#edf6fc', zorder=0)
    section.axhspan(-1., 0., facecolor='#f0ede1', zorder=0)
    section.axhline(0., color=GREY, linewidth=.7, zorder=1)
    section.add_patch(Rectangle((b[0, 0], b[2, 0]), np.ptp(b[0]), np.ptp(b[2]),
                                facecolor='#c5e7e8', edgecolor=TEAL, linewidth=1., zorder=2))
    block_label = section.text(2.4, -.50, 'Conductive\nblock\n' + r'$\sigma=1$ S/m',
                               ha='center', va='center', color=TEAL)
    section.text(3., .19, 'Air\n' + r'$\sigma=10^{-8}$ S/m',
                  ha='center', va='center', color=GREY)
    section.text(-1., -.73, 'Background\n' + r'$\sigma=0.01$ S/m',
                  ha='center', va='center', color=GREY)
    # Preserve the actual receiver-plane intersection; do not exaggerate its depth.
    receiver_line, = section.plot(plane_x, [plane_z, plane_z], color=BLUE,
                                  linewidth=1.15, zorder=4)
    section.annotate('Receiver plane', xy=(4.4, plane_z), xytext=(4.05, -.105),
                     ha='center', va='top', color=BLUE,
                     arrowprops={'arrowstyle': '-', 'color': BLUE, 'lw': .65})
    section.add_patch(FancyArrowPatch(start_km[[0, 2]], end_km[[0, 2]], arrowstyle='-|>',
                                      mutation_scale=6.5, linewidth=1.35, color=RED,
                                      shrinkA=0, shrinkB=0, zorder=6))
    section_source_label = section.text(-2.1, .10, 'Line source\n500 m, 1 A',
                                        ha='center', va='bottom', color=RED)
    section.text(.98, .98, r'$y=0$ m section', transform=section.transAxes,
                  ha='right', va='top', color=GREY)
    section.set(ylim=(-1., .35), yticks=[-1., -.75, -.5, -.25, 0., .25],
                ylabel=r'$z$ / km')
    for ax, label in zip((plan, section), ('(a)', '(b)')):
        ax.set(xlim=(-3., 5.5), xticks=np.arange(-3., 6.), xlabel=r'$x$ / km')
        ax.set_title(label, loc='left', pad=5)
        ax.grid(False)
        ax.tick_params(direction='out', length=3., width=.65)
        for spine in ax.spines.values():
            spine.set_linewidth(.65)
    finalize(fig)
    type_check = typography_check(fig)
    renderer = fig.canvas.get_renderer()
    block_box = block_label.get_window_extent(renderer)
    block_corners = section.transData.transform([[b[0, 0], b[2, 0]], [b[0, 1], b[2, 1]]])
    assert (block_box.x0 > block_corners[0, 0] and block_box.x1 < block_corners[1, 0]
            and block_box.y0 > block_corners[0, 1] and block_box.y1 < block_corners[1, 1])
    for ax, source_label in ((plan, plan_source_label), (section, section_source_label)):
        assert source_label.get_window_extent(renderer).x0 > ax.get_window_extent(renderer).x0
    annotations = [text.get_text() for ax in (plan, section) for text in ax.texts]
    assert not any('z=' in text.replace(' ', '') or 'depth' in text.lower()
                   for text in annotations)
    assert not plan.lines, 'No highlighted receiver line or sampling lattice in plan view'
    assert np.array_equal(receiver_line.get_ydata(), [plane_z, plane_z])
    assert not any(line.get_visible() for ax in (plan, section)
                   for line in ax.get_xgridlines() + ax.get_ygridlines())
    output.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for extension in ('png', 'pdf', 'svg'):
        path = output.with_suffix('.' + extension)
        fig.savefig(path, dpi=320, facecolor='white', bbox_inches=None)
        outputs[extension] = fingerprint(path)
        print(path)
    metadata = {
        'purpose': 'Current Figure 1: shifted-block plan view and y=0 section, geometry only',
        'script': fingerprint(__file__), 'parameters': fingerprint(parameters),
        'physical': physical, 'sampling_plane': sampling,
        'source_start_m': start.tolist(), 'source_end_m': end.tolist(),
        'source_length_m': float(np.linalg.norm(end - start)),
        'source_current_A': physical['source']['current_A'],
        'sigma_air_S_m': physical['conductivity_S_per_m']['air'],
        'sigma_background_S_m': physical['conductivity_S_per_m']['background'],
        'sigma_anomaly_S_m': physical['conductivity_S_per_m']['block'],
        'block_bounds_m': bounds.tolist(), 'section_y_m': 0.,
        'receivers': {'count': sampling['receiver_count'],
                      'x_range_m': sampling['x_range_m'], 'y_range_m': sampling['y_range_m'],
                      'x_spacing_m': [sampling['spacing_m']], 'actual_z_m': sampling['z_m'],
                      'projection': 'Actual intersection of receiver plane with y=0 section; no depth offset',
                      'highlighted_line': None},
        'section': {'plane': 'y=0 m', 'receiver_intersection_true_z_m': sampling['z_m'],
                    'receiver_vertical_shift_m': 0.,
                    'receiver_feature': 'Intersection of sampling plane with section'},
        'presentation': {'mesh_information_shown': False, 'background_coordinate_grid_shown': False,
                         'receiver_plane_lattice_shown': False, 'highlighted_receiver_line_shown': False,
                         'receiver_z_annotation_shown': False, 'total_title_shown': False,
                         'left_source_length_and_current_shown': False,
                         'panel_labels': ['(a)', '(b)'], 'lettering_language': 'English',
                         'axes_geometry': 'Independent horizontal and vertical scales; exact data coordinates',
                         'panel_axis_limits': [{'x': list(ax.get_xlim()), 'y': list(ax.get_ylim())}
                                               for ax in (plan, section)]},
        'typography': specification(), 'typography_check': type_check,
        'rendered_annotations': annotations, 'outputs': outputs,
    }
    provenance = parameters.with_name('source_metadata.json')
    if provenance.is_file():
        metadata['source_metadata'] = fingerprint(provenance)
    if entry_script is not None:
        metadata['entry_script'] = fingerprint(entry_script)
    metadata_path = (Path(metadata_path).resolve() if metadata_path is not None
                     else output.with_name(output.name + '_metadata.json'))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')
    plt.close(fig)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT.parent)
    args = parser.parse_args()
    draw(args.parameters, args.output_dir / 'fig01_block_model')


if __name__ == '__main__':
    main()
