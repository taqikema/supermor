"""Shared final-size typography for the SUPERMOR manuscript figures.

Ordinary lettering uses Times New Roman; mathematical variables retain italics.
Use configure() before constructing a figure and finalize() before saving it.
Save without bbox_inches='tight' so the declared physical width is preserved.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.text import Text

FONT_NAME = 'Times New Roman'
WIDTH_CM = 15.92
AXIS_PT = 7.
TEXT_PT = 7.
PANEL_PT = 7.
_PANEL = re.compile(r'^\s*\([a-z]\)(?:\s|$)')


def configure():
    """Register the system Times faces and configure text/math typography."""
    directories = [Path('/mnt/c/Windows/Fonts'), Path('C:/Windows/Fonts')]
    if os.environ.get('SUMOR_PAPER_FONT_DIR'):
        directories.insert(0, Path(os.environ['SUMOR_PAPER_FONT_DIR']))
    for directory in directories:
        for filename in ('times.ttf', 'timesi.ttf', 'timesbd.ttf', 'timesbi.ttf'):
            path = directory / filename
            if path.is_file():
                font_manager.fontManager.addfont(str(path))
    font_manager.findfont(FONT_NAME, fallback_to_default=False)
    mpl.rcParams.update({
        'font.family': FONT_NAME,
        'font.serif': [FONT_NAME],
        'font.sans-serif': [FONT_NAME],
        'font.size': TEXT_PT,
        'axes.labelsize': AXIS_PT,
        'axes.titlesize': AXIS_PT,
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'xtick.labelsize': TEXT_PT,
        'ytick.labelsize': TEXT_PT,
        'legend.fontsize': TEXT_PT,
        'legend.title_fontsize': TEXT_PT,
        'figure.titlesize': AXIS_PT,
        'mathtext.fontset': 'custom',
        'mathtext.rm': FONT_NAME,
        'mathtext.it': FONT_NAME + ':italic',
        'mathtext.bf': FONT_NAME + ':bold',
        'mathtext.sf': FONT_NAME,
        'mathtext.tt': FONT_NAME,
        # Script alphabets and large operators need mathematical glyphs that
        # Times does not contain. Preserve that notation rather than replacing
        # a script symbol with an ordinary italic letter.
        'mathtext.cal': 'STIXGeneral:italic',
        'mathtext.fallback': 'stix',
        'text.usetex': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'savefig.dpi': 320,
        'savefig.bbox': None,
        'axes.unicode_minus': True,
    })
    return {'font': FONT_NAME, 'axis_pt': AXIS_PT, 'text_pt': TEXT_PT,
            'panel_pt': PANEL_PT, 'default_width_cm': WIDTH_CM}


def _font(text, size, weight='normal'):
    text.set_fontfamily(FONT_NAME)
    text.set_fontsize(size)
    text.set_fontweight(weight)
    text.set_fontstyle('normal')
    if hasattr(text, 'set_math_fontfamily'):
        text.set_math_fontfamily('custom')


def finalize(fig, width_cm=WIDTH_CM, *, legend_pt=TEXT_PT):
    """Apply role-based sizes to all text, including inset/subfigure labels.

    Height is chosen by the figure's renderer so multi-panel figures can reflow
    without reducing type size. Lettering defaults to 7 pt; compact legends can
    specify a smaller size. Panel identifiers retain bold weight.
    """
    fig.set_size_inches(float(width_cm) / 2.54, fig.get_figheight(), forward=True)
    for item in fig.findobj(match=Text):
        _font(item, TEXT_PT)
    for ax in fig.findobj(match=Axes):
        for axis in (ax.xaxis, ax.yaxis, getattr(ax, 'zaxis', None)):
            if axis is None:
                continue
            _font(axis.label, AXIS_PT)
            _font(axis.get_offset_text(), TEXT_PT)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                _font(tick.label1, TEXT_PT)
                _font(tick.label2, TEXT_PT)
        for title in (ax.title, ax._left_title, ax._right_title):
            _font(title, AXIS_PT)
        legend = ax.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                _font(item, legend_pt)
            _font(legend.get_title(), legend_pt)
    for item in fig.findobj(match=Text):
        if _PANEL.match(item.get_text()):
            _font(item, PANEL_PT, 'bold')
    fig.canvas.draw()
    return fig


def specification(width_cm=WIDTH_CM):
    return {'font_family': FONT_NAME, 'width_cm': float(width_cm),
            'axis_and_colorbar_title_pt': AXIS_PT,
            'tick_legend_annotation_pt': TEXT_PT,
            'panel_label_pt': PANEL_PT, 'panel_label_weight': 'bold',
            'math_variables': 'italic; units upright'}
