#!/usr/bin/env python3
"""Plot the paper's four-method MPI comparison with uniform method markers.

Uses the saved 23-configuration rerun and the previous measured DFOM1 baseline.
Launches no solvers and does not alter measurements or fit missing values.
Explicit legacy inputs containing the earlier fitted estimates remain supported.

Example: python figs/case2_layered_sphere/plot_supermor_paper_mpi_performance.py \
    --run-dir outputs/case2_layered_sphere/mpi_scaling/RUN

With no arguments, reads the frozen paper inputs and saves beside this script.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figs.common import mpi_performance_plot as base
from figs.common.sumor_paper_figure_style import configure, finalize, specification
from figs.common.paths import ROOT, fingerprint
from figs.common.sphere_inputs import load_mpi_inputs

OUTPUT = Path(__file__).resolve().parent
NAME = "fig08_mpi_scaling"
LEGEND_PT = 7.0
MARKER_PT = 3.5
PAPER_DISPLAY_NAMES = {**base.DISPLAY_NAMES, "CPC": "RK"}


def display_path(path, relative_to=ROOT):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(relative_to.resolve()))
    except ValueError:
        return str(path)


def load_comparison(source):
    """Validate the complete comparison and derive its stored one-process baselines."""
    summary = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    rows = summary["rows"]
    data = {(row["mpi_ranks"], row["method"]): row for row in rows}
    expected = {(p, method) for p in base.RANKS for method in base.METHODS}
    if len(rows) != 24 or len(data) != 24 or set(data) != expected:
        raise ValueError("The paper comparison requires 24 unique method/MPI combinations")
    baselines = {method: data[(1, method)]["complete_forward_s"]
                 for method in base.METHODS}
    for row in rows:
        for metric in ("complete_forward_s", "peak_pss_gib", "speedup"):
            if not math.isfinite(row[metric]) or row[metric] <= 0:
                raise ValueError(f"Invalid {metric} for {row['method']} / {row['mpi_ranks']} MPI")
        for metric in ("complete_forward_s_origin", "peak_pss_gib_origin"):
            if row[metric] not in ("measured", "model_estimate"):
                raise ValueError(f"Invalid origin {metric}: {row[metric]!r}")
        baseline = baselines[row["method"]]
        if not math.isclose(row["speedup"], baseline / row["complete_forward_s"], rel_tol=1e-13):
            raise ValueError(f"Inconsistent same-method speedup for {row['method']} / {row['mpi_ranks']} MPI")
    return summary, data, rows, baselines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input-dir", type=Path, help="Frozen paper MPI drawing inputs")
    group.add_argument("--run-dir", type=Path, help="One complete new MPI-scaling run")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--name", default=NAME, help="Output filename stem")
    args = parser.parse_args(argv)
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a nonempty filename stem without directories")
    loaded, input_files = load_mpi_inputs(input_dir=args.input_dir, run_dir=args.run_dir)
    source = input_files[0].resolve()
    output = args.output_dir.resolve()
    name = args.name
    summary, data, rows, baselines = load_comparison(loaded)
    plot_rows = [{**row, "sampling_warning": False} for row in rows]
    configure()
    figure = base.make_figure(summary, data, plot_rows, baselines)
    for axis in figure.axes:
        for line in axis.lines:
            line.set(linestyle="-", markersize=MARKER_PT, markeredgewidth=1)
            if line.get_marker() in (None, "None", "none", ""):
                line.set_marker(".")
    # The reusable plot creates its figure legend in an 11 pt context. Rebuild
    # the same entries at the manuscript size so its frame and spacing agree.
    for legend in tuple(figure.legends):
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = legend.legendHandles
        for handle in handles:
            handle.set(linestyle="-", markersize=MARKER_PT, markeredgewidth=1)
        display_labels = {base.DISPLAY_NAMES[method]: PAPER_DISPLAY_NAMES[method]
                          for method in base.METHODS}
        labels = [display_labels.get(text.get_text(), text.get_text())
                  for text in legend.get_texts()]
        for handle, label in zip(handles, labels):
            handle.set_label(label)
        legend.remove()
        figure.legend(handles=handles, labels=labels, loc="upper center",
                      bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
                      columnspacing=1.8, handlelength=3, fontsize=LEGEND_PT)
    # Use larger, nearly square panels with less space between them.
    figure.set_size_inches(15.92 / 2.54, 2.5)
    figure.axes[0].set_ylim(min(row["complete_forward_s"] for row in rows) * 0.75,
                            max(row["complete_forward_s"] for row in rows) * 1.25)
    figure.axes[1].set(title="(b) Peak memory", ylabel="Peak memory / GiB")
    for axis, panel in zip(figure.axes, ("(a)", "(b)", "(c)")):
        axis.set_title(panel)
    for text in figure.axes[2].texts:
        if text.get_text() == "Ideal S = p":
            text.set_transform(figure.axes[2].transAxes)
            text.set_position((.04, .95))
            text.set_ha("left")
            text.set_va("top")
    figure.subplots_adjust(left=.085, right=.985, bottom=.20, top=.84, wspace=.30)
    finalize(figure)

    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for extension in ("png", "pdf", "svg"):
        path = output / f"{name}.{extension}"
        figure.savefig(path, dpi=320, facecolor="white", bbox_inches=None)
        outputs[extension] = fingerprint(path)
        print(display_path(path))
    base.plt.close(figure)
    metadata_path = output / f"{name}_metadata.json"
    all_measured = all(row[metric] == "measured" for row in rows for metric in
                       ("complete_forward_s_origin", "peak_pss_gib_origin"))
    companion_names = (("results.json", "old_DFOM1_measured.json", "mpi_performance_measured.csv")
                       if all_measured else ("snapshot.json", "mpi_performance_estimated.csv"))
    companions = [source.parent / filename for filename in companion_names]
    metadata = {
        "description": summary.get("description", "Four-method MPI performance measurements"),
        "typography": specification(),
        "script": fingerprint(Path(__file__).resolve()),
        "base_plot_script": fingerprint(Path(base.__file__)),
        "inputs": [fingerprint(path) for path in input_files],
        "optional_companion_files_missing": ([path.name for path in companions if not path.is_file()]
                                             if summary.get("input_kind") != "new_run" else []),
        "cells": summary["cells"], "complex_field_dofs": summary["field_dofs"],
        "node": summary.get("node"),
        "nodes": sorted({row["node"] for row in rows if row.get("node")}),
        "threads_per_rank": summary["threads_per_rank"],
        "frequency_count": len(summary["frequencies_hz"]), "mpi_ranks": base.RANKS,
        "display_names": PAPER_DISPLAY_NAMES,
        "baseline_1mpi_s": baselines,
        "measured_configuration_count": sum(
            row["complete_forward_s_origin"] == row["peak_pss_gib_origin"] == "measured"
            for row in rows),
        "source_metadata": {key: value for key, value in summary.items() if key != "rows"},
        "marker_definitions": {"method": "uniform color, line style and open marker shape per method",
                               "size_pt": MARKER_PT, "line_style": "solid",
                               "edge_width_pt": 1,
                               "ideal_reference": "thin grey solid line with dot markers",
                               "data_origin": "retained in rows and source files; not encoded visually"},
        "layout": {"width_cm": 15.92, "height_in": 2.5,
                   "margins": {"left": .085, "right": .985, "bottom": .20, "top": .84},
                   "wspace": .30,
                   "panel_size_in": [[axis.get_position().width * figure.get_figwidth(),
                                       axis.get_position().height * figure.get_figheight()]
                                      for axis in figure.axes]},
        "limitations": ["Sampled PSS peaks retain the source measurement definition"],
        "rows": rows, "outputs": outputs,
    }
    if not all_measured:
        estimation = summary.get("estimation", {})
        metadata.update({
            "models": estimation.get("models", {}),
            "measured_configuration_count": estimation.get(
                "observed_row_count", metadata["measured_configuration_count"]),
            "estimated_missing_configurations": estimation.get("estimated_missing_configurations", []),
            "limitations": ["Estimates do not change failed or unexecuted experiment statuses",
                            "All DFOM speedups use an estimated one-rank time baseline",
                            "Sparse PSS sampling may miss peaks; fitted memory is not a measured correction"],
        })
        original = data[(64, "DFOM")].get("original_measurement")
        if original:
            metadata.update({
                "dfom64_original_sample_GiB": original["peak_pss_gib"],
                "dfom64_estimated_memory_GiB": data[(64, "DFOM")]["peak_pss_gib"],
                "dfom64_maximum_sample_gap_s": original.get("actual_max_inter_sample_s"),
            })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(display_path(metadata_path))


if __name__ == "__main__":
    main()
