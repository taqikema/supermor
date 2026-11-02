"""Read frozen sphere figure inputs or one explicitly selected numerical run.

New runs are normalized in memory. Source measurements and their provenance
are preserved; incomplete experiments never fall back to the paper data.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .paths import ROOT

METHODS = ("DFOM", "IFOM", "CPC", "SUMOR")
RANKS = (1, 4, 8, 16, 32, 64)
COUNTS = (2, 8, 10, 16, 20, 30, 40)
ALIASES = {"DFOM": "DFOM", "IFOM": "IFOM", "RK": "CPC", "CPC": "CPC",
           "SUPERMOR": "SUMOR", "SUMOR": "SUMOR"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite measurement")
    return value


def _method(name):
    try:
        return ALIASES[name]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown measured method: {name!r}") from None


def _run_rows(run_dir, experiment):
    root = Path(run_dir).expanduser().resolve()
    summary_file = root / "summary.json"
    rows = _json(summary_file).get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Run summary must contain nonempty cases")
    expected = ({(m, p) for m in METHODS for p in RANKS} if experiment == "mpi_scaling"
                else {(m, n) for m in METHODS for n in COUNTS})
    records, sources, seen, stats_all = [], [summary_file], set(), []
    for original in rows:
        method = _method(original.get("method"))
        dimension = int(original["mpi_ranks"] if experiment == "mpi_scaling" else original["n_frequency"])
        key = (method, dimension)
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate or unexpected sphere configuration: {key}")
        seen.add(key)
        if original.get("status") != "converged" or original.get("returncode") != 0:
            raise ValueError(f"Sphere configuration did not converge: {key}")
        ranks = int(original["mpi_ranks"])
        if experiment == "frequency_count" and ranks != 8:
            raise ValueError("Frequency-count measurements require 8 MPI ranks")
        # Coordinator outputs record their result directory. Relative paths are
        # resolved against this run; relocated conventional run trees also work.
        result = Path(original["output"])
        if not result.is_absolute():
            result = root / result
        result = result.resolve()
        if not result.is_relative_to(root) or not result.is_dir():
            label = "p" + str(dimension) if experiment == "mpi_scaling" else "N" + str(dimension)
            result = (root / original["method"] / label / "result").resolve()
        if not result.is_relative_to(root):
            raise ValueError("Worker output must belong to the selected run directory")
        stats_file, resources_file = result / "stats.json", result.parent / "resources.json"
        stats, resources = _json(stats_file), _json(resources_file)
        if stats.get("status") != "converged" or resources.get("returncode") != 0:
            raise ValueError(f"Worker or monitor failed for configuration {key}")
        if _method(stats["method"]) != method or int(stats["mpi_ranks"]) != ranks:
            raise ValueError(f"Coordinator and worker configuration disagree: {key}")
        if experiment == "frequency_count" and int(stats["n_frequency"]) != dimension:
            raise ValueError(f"Coordinator and worker frequency count disagree: {key}")
        seconds = _positive(stats.get("complete_forward_s"), "Worker complete_forward_s")
        if original.get("complete_forward_s") is not None and not math.isclose(
                seconds, float(original["complete_forward_s"]), rel_tol=1e-12):
            raise ValueError("Coordinator and worker forward times disagree")
        pss = resources.get("peak_job_pss_bytes")
        if isinstance(pss, bool) or not isinstance(pss, int) or pss <= 0:
            raise ValueError("Monitor peak_job_pss_bytes must be a positive integer")
        gib = pss / 2**30
        if original.get("peak_job_pss_gib") is not None and not math.isclose(
                gib, float(original["peak_job_pss_gib"]), rel_tol=1e-12):
            raise ValueError("Coordinator and monitor memory measurements disagree")
        row = {**original, "method": method, "complete_forward_s": seconds,
               "peak_job_pss_bytes": pss, "peak_pss_gib": gib,
               "complete_forward_s_origin": "measured", "peak_pss_gib_origin": "measured",
               "data_origin": "measured", "source_method": original["method"],
               "worker_provenance": {k: stats[k] for k in (
                   "kernel_selection", "historical_core_equivalence", "production_commit",
                   "historical_production_commit", "production_source_sha256", "software",
                   "timing_policy", "frequencies_hz", "threads_per_rank") if k in stats},
               "measurement_provenance": {k: resources[k] for k in (
                   "measurement", "profile", "implementation", "historical_monitor_byte_identical",
                   "monitor_excluded", "sample_interval_s", "thread_environment") if k in resources}}
        records.append(row)
        stats_all.append(stats)
        sources.extend([stats_file, resources_file])
    if seen != expected:
        raise ValueError(f"Incomplete sphere {experiment} grid: missing {sorted(expected-seen)}")
    return records, sources, stats_all


def load_mpi_inputs(*, input_dir=None, run_dir=None):
    if input_dir is not None and run_dir is not None:
        raise ValueError("Choose either frozen input_dir or run_dir")
    if run_dir is None:
        folder = Path(input_dir) if input_dir is not None else ROOT / "outputs/case2_layered_sphere/mpi_scaling/paper"
        source = folder / "mpi_performance_measured.json"
        companions = [folder / name for name in ("mpi_performance_measured.csv", "results.json", "old_DFOM1_measured.json")]
        return _json(source), [source, *(p for p in companions if p.is_file())]
    rows, sources, stats = _run_rows(run_dir, "mpi_scaling")
    for key in ("cells", "field_dofs", "frequencies_hz"):
        if any(s.get(key) != stats[0].get(key) for s in stats) or stats[0].get(key) is None:
            raise ValueError(f"MPI worker {key} must agree for all configurations")
    baselines = {row["method"]: row["complete_forward_s"] for row in rows if row["mpi_ranks"] == 1}
    for row in rows:
        row["baseline_1mpi_s"] = baselines[row["method"]]
        row["speedup"] = row["baseline_1mpi_s"] / row["complete_forward_s"]
    summary = {"description": "Explicitly selected new sphere MPI measurements",
               "input_kind": "new_run", "run_directory": str(Path(run_dir).resolve()),
               "cells": stats[0]["cells"], "field_dofs": stats[0]["field_dofs"],
               "frequencies_hz": stats[0]["frequencies_hz"],
               "threads_per_rank": stats[0].get("threads_per_rank"),
               "kernel_selections": sorted({s.get("kernel_selection", "unrecorded") for s in stats}),
               "rows": rows}
    return summary, sources


def load_frequency_inputs(*, input_dir=None, run_dir=None):
    if input_dir is not None and run_dir is not None:
        raise ValueError("Choose either frozen input_dir or run_dir")
    if run_dir is not None:
        rows, files, _ = _run_rows(run_dir, "frequency_count")
        return rows, files
    folder = Path(input_dir) if input_dir is not None else ROOT / "outputs/case2_layered_sphere/frequency_count/paper"
    source = folder / "performance_source.csv"
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    validation = folder / "validation_summary.json"
    return rows, [source, *([validation] if validation.is_file() else [])]
