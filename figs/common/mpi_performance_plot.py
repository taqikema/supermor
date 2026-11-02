"""Shared renderer for the paper's four-method MPI performance comparison.

The figure entry point validates inputs, applies paper typography and saves
outputs. This module supplies only the method styles and three-panel renderer.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter


RANKS = (1, 4, 8, 16, 32, 64)
METHODS = ("IFOM", "SUMOR", "CPC", "DFOM")
DISPLAY_NAMES = {"IFOM": "IFOM", "SUMOR": "SUPERMOR", "CPC": "CPC", "DFOM": "DFOM"}
STYLES = {
    "IFOM": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "SUMOR": {"color": "#009E73", "marker": "^", "linestyle": "-."},
    "CPC": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    "DFOM": {"color": "#CC79A7", "marker": "D", "linestyle": ":"},
}


def make_figure(summary, data, rows, baselines):
    """三个子图只连接已有实测点，共享方法颜色、标记和线型。"""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "figure.facecolor": "white",
                         "axes.facecolor": "white"}):
        figure, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
        figure.subplots_adjust(left=0.058, right=0.985, bottom=0.14,
                               top=0.84, wspace=0.28)
        handles = [Line2D([], [], label=DISPLAY_NAMES[method], linewidth=1.9, markersize=6.8,
                          markerfacecolor="white", markeredgewidth=1.5,
                          **STYLES[method]) for method in METHODS]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                      ncol=4, frameon=False, columnspacing=3.1, handlelength=3)

        metrics = ("complete_forward_s", "peak_pss_gib", "speedup")
        titles = ("(a) Complete forward time", "(b) Sampled peak memory",
                  "(c) Same-method speedup")
        labels = ("Complete forward time (s)", "Sampled peak aggregate PSS (GiB)",
                  r"Speedup $T(1\,\mathrm{MPI})/T(p\,\mathrm{MPI})$")
        for axis, metric, title, ylabel in zip(axes, metrics, titles, labels):
            for method in METHODS:
                points = [data[(ranks, method)] for ranks in RANKS
                          if (ranks, method) in data
                          and data[(ranks, method)][metric] is not None]
                if points:
                    axis.plot([row["mpi_ranks"] for row in points],
                              [row[metric] for row in points], linewidth=1.9,
                              markersize=6.7, markerfacecolor="white",
                              markeredgewidth=1.5, zorder=3, **STYLES[method])
            axis.set_xscale("log", base=2)
            axis.set(xlim=(0.85, 76), xticks=RANKS, title=title,
                     xlabel="MPI ranks, p", ylabel=ylabel)
            axis.xaxis.set_major_formatter(ScalarFormatter())
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.grid(True, which="major", color="#d9d9d9", linewidth=0.7)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(direction="out", length=4, color="#555555")

        axes[0].set_yscale("log", base=10)
        times = [row["complete_forward_s"] for row in rows]
        axes[0].set_ylim(10**math.floor(math.log10(min(times) * 0.9)),
                         10**math.ceil(math.log10(max(times) * 1.1)))
        axes[0].yaxis.set_major_formatter(ScalarFormatter())
        axes[0].yaxis.set_minor_formatter(NullFormatter())
        axes[0].grid(True, which="minor", axis="y", color="#eeeeee", linewidth=0.45)
        axes[1].set_ylim(0, max(row["peak_pss_gib"] for row in rows) * 1.15)
        flagged = [row for row in rows if row["sampling_warning"]]
        for index, row in enumerate(flagged):
            axes[1].annotate("*", (row["mpi_ranks"], row["peak_pss_gib"]),
                             xytext=((-9 if index % 2 == 0 else 9),
                                     (-15 if index % 2 == 0 else 7)),
                             textcoords="offset points", ha="center", fontsize=15,
                             fontweight="bold", color=STYLES[row["method"]]["color"])
        axes[2].set_yscale("log", base=2)
        speedups = [row["speedup"] for row in rows if row["speedup"] is not None]
        axes[2].set(ylim=(min([1] + speedups) * 0.8, max([64] + speedups) * 1.25),
                    yticks=(1, 2, 4, 8, 16, 32, 64))
        axes[2].yaxis.set_major_formatter(ScalarFormatter())
        axes[2].plot(RANKS, RANKS, color="#9a9a9a", linewidth=1.1,
                     linestyle="--", zorder=1)
        axes[2].text(28, 48, "Ideal S = p", color="#777777", fontsize=9)
        missing_baselines = [method for method in METHODS if method not in baselines]
        if missing_baselines:
            axes[2].text(0.04, 0.95,
                         ", ".join(DISPLAY_NAMES[method] for method in missing_baselines)
                         + ": awaiting 1-MPI baseline",
                         transform=axes[2].transAxes, va="top", fontsize=9,
                         color="#555555")
        return figure
