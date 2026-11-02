#!/usr/bin/env python3
"""Plot the 2026-11-01 shifted-block poles strategy comparison.

Read the frozen complete-forward costs and per-frequency maxima of pointwise
three-component complex-vector relative errors. No FEM solve is launched.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figs.common.paths import fingerprint
from figs.common.block_inputs import strategy_inputs, source_provenance
from figs.common.sumor_paper_figure_style import configure, finalize, specification


DEFAULT_REPORT = ROOT / "outputs/case1_block/strategy/paper/poles"
DEFAULT_OUTPUT = Path(__file__).resolve().parent
NAME = "fig06_poles"
LEGEND_PT = 7.0
GROUPS = (
    ("single_geometric", "Single pole", "#0072B2", "-", "o"),
    ("dual_existing", "Dual poles", "#D55E00", "-", "s"),
)
COSTS = (
    ("dynamic_basis_columns", "Dynamic basis columns"),
    ("complete_forward_s", "Complete forward time / s"),
    ("peak_job_pss_GiB", "Peak memory / GiB"),
)
FIELDS = (
    ("E", "receiver_E_max_pointwise_pct"),
    ("H", "receiver_H_max_pointwise_pct"),
)


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def draw(report=DEFAULT_REPORT, output=DEFAULT_OUTPUT):
    report, output = Path(report).resolve(), Path(output).resolve()
    comparison_csv = report / "comparison.csv"
    frequency_csv = report / "frequency_metrics.csv"
    costs = {row["method"]: row for row in read_csv(comparison_csv)}
    comparison = json.loads((report / "comparison.json").read_text(encoding="utf-8"))
    rows = read_csv(frequency_csv)
    frequencies = [2.0**index for index in range(-6, 14)]
    per_method = {}
    for method, _, _, _, _ in GROUPS:
        if costs[method]["successful_forward"] != "True":
            raise ValueError(f"Expected a successful complete forward run: {method}")
        for column, _ in COSTS:
            value = float(costs[method][column])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid forward cost: {method}, {column}")
        points = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["index"]),
        )
        if [int(row["index"]) for row in points] != list(range(20)):
            raise ValueError(f"Expected all 20 unique frequency indices: {method}")
        if [float(row["frequency_hz"]) for row in points] != frequencies:
            raise ValueError(f"Unexpected frequency grid: {method}")
        for row in points:
            if (int(row["receiver_count"]) != 3321 or row["reference_available"] != "True"
                    or row["observation_errors"]):
                raise ValueError(f"Invalid receiver/reference data: {method}, {row['index']}")
            for field, column in FIELDS:
                if (row[f"receiver_{field}_max_pointwise_status"] != "defined"
                        or int(row[f"receiver_{field}_zero_reference_points"]) != 0):
                    raise ValueError("A zero reference vector makes the full-plane maximum undefined")
                value = float(row[column])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid pointwise maximum: {method}, {row['index']}")
        per_method[method] = points
    expected_pairs = {(method, index) for method, *_ in GROUPS for index in range(20)}
    if {(row["method"], int(row["index"])) for row in rows} != expected_pairs or len(rows) != len(expected_pairs):
        raise ValueError("Expected exactly one complete twenty-frequency curve per strategy")
    reference = comparison["reference"]
    if not reference["reference_available"]:
        raise ValueError("The same-mesh numerical reference is unavailable")
    reference_records = reference["controller_summary"]["per_frequency"]
    if len(reference_records) != 20:
        raise ValueError("Expected all twenty native reference diagnostic records")

    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 240,
    })
    configure()
    figure = plt.figure(figsize=(15.92 / 2.54, 4.7))
    grid = figure.add_gridspec(2, 6, height_ratios=(.64, 1))
    figure.subplots_adjust(left=.115, right=.985, bottom=.108, top=.91,
                           wspace=1.8, hspace=.556)
    figure.text(.5, .975, "(a) Basis size and complete forward costs",
                ha="center", va="top")
    figure.text(.5, .535, "(b) Maximum pointwise E/H relative errors at each frequency",
                ha="center", va="top")

    # Preserve all three metrics, bar values and colors from the original cost plot.
    cost_axes = [figure.add_subplot(grid[0, 2*i:2*i+2]) for i in range(3)]
    for axis, (column, label) in zip(cost_axes, COSTS):
        values = []
        for number, (method, _, color, _, _) in enumerate(GROUPS):
            value = float(costs[method][column])
            values.append(value)
            bars = axis.bar(number, value, width=0.5, color=color)
            axis.bar_label(bars, labels=[f"{value:.5g}"], padding=3)
        axis.set_ylim(0, 1.2 * max(values))
        # Keep the original category positions while narrowing the bars.
        axis.set_xlim(-0.49, 1.49)
        axis.set_xticks(range(2), [group[1] for group in GROUPS])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

    receiver_axes = [figure.add_subplot(grid[1, :3]),
                     figure.add_subplot(grid[1, 3:])]
    curve_metadata = []
    for axis, (field, column) in zip(receiver_axes, FIELDS):
        for method, label, color, linestyle, marker in GROUPS:
            values = [float(row[column]) for row in per_method[method]]
            # Mark every measured frequency with compact open symbols.
            axis.loglog(
                frequencies, values, color=color, linestyle=linestyle,
                linewidth=1.8, marker=marker, markersize=3.5,
                markerfacecolor="white", markeredgewidth=1,
                markevery=1, label=label,
            )
            curve_metadata.append({
                "method": method, "field": field, "column": column,
                "color": color, "linestyle": linestyle, "legend": label,
                "marker": marker, "marker_size_pt": 3.5, "marker_count": len(values),
                "panel": f"bottom {'left' if field == 'E' else 'right'}",
                "frequency_count": len(values), "max_pointwise_relative_error_pct": max(values),
                "values_pct": values, "zero_error_values": values.count(0.0),
            })
        axis.set(xlabel="Frequency / Hz",
                 ylabel=rf"Max. pointwise ${field}$ relative error / %")
        axis.grid(True, which="major", alpha=0.22)
        axis.grid(True, which="minor", alpha=0.08)
        axis.set_xlim(frequencies[0] / 1.3, frequencies[-1] * 1.3)

    # The E and H panels share the same pole strategies and marker styles.
    receiver_axes[0].legend(loc="lower right", fontsize=LEGEND_PT,
                            handlelength=3.2, framealpha=0.95, edgecolor="0.8")

    finalize(figure, legend_pt=LEGEND_PT)
    from matplotlib.text import Text
    texts = [item for item in figure.findobj(match=Text) if item.get_visible() and item.get_text()]
    if {tuple(item.get_fontfamily()) for item in texts} != {("Times New Roman",)} or {float(item.get_fontsize()) for item in texts} != {7.0}:
        raise ValueError("Unexpected final typography")
    for axis in receiver_axes:
        if len(axis.lines) != len(GROUPS) or any(len(line.get_xdata()) != 20 for line in axis.lines):
            raise ValueError("Expected one twenty-marker curve per strategy and field")
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        destination = output / f"{NAME}.{suffix}"
        figure.savefig(destination, dpi=320, bbox_inches=None)
        outputs[suffix] = fingerprint(destination)
        print(relative(destination))
    plt.close(figure)

    metadata_path = output / f"{NAME}_metadata.json"
    metadata = {
        "description": "HPC single/dual-pole cost comparison above separate E and H frequency panels",
        "typography": {**specification(), "legend_pt": LEGEND_PT},
        "script": fingerprint(__file__),
        "inputs": [fingerprint(comparison_csv), fingerprint(frequency_csv),
                   fingerprint(report / "comparison.json")],
        "source_report": relative(report),
        "layout": "Two rows: (a) three cost axes; (b) E on the left, H on the right, two strategies per axis",
        "cost_bar_width": 0.5,
        "receiver_legend": "One shared strategy legend at the lower right of the E panel",
        "frequency_hz": frequencies,
        "receiver_metric": "max over all 3321 receivers of 100 * norm(F - F_ref) / norm(F_ref); three complex components within each point; percent",
        "reference": "Independent same-mesh UFL/MUMPS numerical reference from job 20466531",
        "reference_caveat": "All twenty frequencies retained; lowest two fail the native 1e-10 strict diagnostic; no reference flags altered",
        "maximum_reference_actual_residual": reference["maximum_recorded_residual"],
        "reference_native_verified": reference["native_reference_verified"],
        "reference_strict_unverified_frequencies_hz": [row["frequency"] for row in reference_records if not row["verified"]],
        "zero_reference_policy": "Any point with a zero reference vector makes the full-plane maximum undefined; no epsilon or weak-field mask",
        "zero_on_log_axis": "Zero error values remain in the source CSV but cannot be displayed on logarithmic axes",
        "receiver_count": 3321,
        "job_id": "20466531", "MPI_ranks": 1, "threads_per_rank": 1,
        "cost_columns": [column for column, _ in COSTS],
        "cost_values": {
            method: {column: float(costs[method][column]) for column, _ in COSTS}
            for method, _, _, _, _ in GROUPS
        },
        "curves": curve_metadata, "outputs": outputs,
    }
    provenance = report.parent / "source_metadata.json"
    if provenance.is_file():
        metadata["source_metadata"] = fingerprint(provenance)
    run_provenance = source_provenance(report)
    if run_provenance:
        metadata['run_provenance'] = run_provenance
        metadata['reference'] = 'Independent same-input UFL/MUMPS reference from the selected run'
        metadata['reference_caveat'] = 'Native per-frequency reference diagnostic flags are preserved'
        metadata.pop('job_id', None)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(relative(metadata_path))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input-dir', type=Path)
    source.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = strategy_inputs(args.run_dir, "poles") if args.run_dir else (args.input_dir or DEFAULT_REPORT)
    draw(inputs, args.output_dir)


if __name__ == "__main__":
    main()
