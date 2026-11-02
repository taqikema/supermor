"""Launch the paper MR3D method/mesh/repeat matrix with portable MPI IO.

Execute this entry on a node with sufficient resources. --dry-run lists the
commands without importing FEM, launching MPI, or writing output files.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "core", ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from runs.case3_mr3d.run_methods_mpi import (
    EXPECTED_CELLS, EXPECTED_REAL_DOF, load_project_inputs, method_name,
)
from runs.common.measurement import run_monitored
from runs.common.result_io import write_json

DISPLAY = {"SUMOR": "SUPERMOR", "CPC": "RK", "DFOM": "DFOM", "IFOM": "IFOM"}


def paper_profile(method, level):
    if method in ("SUMOR", "IFOM") or (method == "CPC" and level == "M1"):
        return "batch_r03"
    if (method, level) in (("DFOM", "M1"), ("CPC", "M2")):
        return "fat09"
    return "later"


def summarize(output, rows):
    """Time median / maximum sampled job PSS, matching the paper aggregation."""
    import statistics

    groups = {}
    for row in rows:
        if row["status"] == "passed":
            groups.setdefault((row["method"], row["level"]), []).append(row)
    summaries = []
    for (method, level), records in groups.items():
        summaries.append(dict(method=DISPLAY[method], level=level,
            cells=EXPECTED_CELLS[level], real_block_dof=EXPECTED_REAL_DOF[level],
            successful_repeats=len(records),
            complete_forward_s=statistics.median(row["complete_forward_s"] for row in records),
            peak_pss_gib=max(row["peak_job_pss_bytes"] for row in records) / 2**30,
            max_independent_residual=max(row["max_independent_residual"] for row in records)))
    write_json(output / "summary.json", dict(runs=rows, groups=summaries,
        time_aggregation="median", memory_aggregation="maximum",
        measurement_implementation="portable_release_v1",
        historical_performance_reproduced=False))
    if summaries:
        with (output / "performance.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", choices=list(EXPECTED_CELLS), default=list(EXPECTED_CELLS))
    parser.add_argument("--methods", nargs="+", type=method_name, default=["SUMOR", "IFOM", "CPC", "DFOM"])
    parser.add_argument("--repeats", type=int, default=3, help="SUPERMOR/IFOM repeats (paper: 3); RK/DFOM always one")
    parser.add_argument("--ranks", type=int, choices=[8], default=8)
    parser.add_argument("--mpi-launcher", default="mpiexec --bind-to core",
                        help="MPI launcher and binding flags; -n 8 is appended")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--material-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--measurement-profile", choices=["auto", "batch_r03", "fat09", "later"], default="auto")
    parser.add_argument("--include-large-dfom", action="store_true",
                        help="Opt in to DFOM M3-M5, which are not paper table results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 3:
        parser.error("--repeats must be 1, 2, or 3")
    args.data_root = args.data_root.expanduser().resolve()
    if args.material_dir:
        args.material_dir = args.material_dir.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    output = (args.output.expanduser().resolve() if args.output else
              ROOT / "outputs/case3_mr3d/mesh_scaling" / stamp)
    input_roots = [args.data_root, ROOT / "data"]
    if args.material_dir:
        input_roots.append(args.material_dir)
    if any(output.is_relative_to(path) for path in input_roots):
        parser.error("Run output must be outside input data directories")
    commands = []
    for level in dict.fromkeys(args.levels):
        for method in dict.fromkeys(args.methods):
            if method == "DFOM" and level not in ("M1", "M2") and not args.include_large_dfom:
                continue
            repeats = range(1, args.repeats + 1) if method in ("SUMOR", "IFOM") else (1,)
            for repeat in repeats:
                profile = paper_profile(method, level) if args.measurement_profile == "auto" else args.measurement_profile
                folder = output / level / DISPLAY[method] / f"run_{repeat:03d}"
                command = shlex.split(args.mpi_launcher) + ["-n", str(args.ranks), args.python, "-B",
                    str(Path(__file__).with_name("run_methods_mpi.py")), "--method", DISPLAY[method],
                    "--level", level, "--repeat", str(repeat), "--data-root", str(args.data_root),
                    "--output", str(folder), "--measurement-profile", profile]
                if args.material_dir:
                    command += ["--material-dir", str(args.material_dir)]
                commands.append(dict(level=level, method=method, repeat=repeat,
                    measurement_profile=profile, output=str(folder), command=command))
    if not commands:
        parser.error("No selected groups: DFOM M3-M5 require --include-large-dfom")
    if args.dry_run:
        print(json.dumps(dict(output=str(output), runs=commands,
                              unique_configurations=len({(r['level'], r['method']) for r in commands})), indent=2))
        return 0
    for level in dict.fromkeys(row["level"] for row in commands):
        load_project_inputs(args.data_root, level, args.material_dir)
    if args.check_inputs:
        print(json.dumps(dict(status="inputs_available", levels=args.levels, runs=len(commands)), indent=2))
        return 0
    if output.exists() and any(output.iterdir()):
        parser.error("Output is not empty; select a new --output")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "campaign.json", dict(runs=commands, data_root=args.data_root,
        material_dir=args.material_dir, mpi_ranks=args.ranks, threads_per_rank=1,
        repeats=args.repeats, historical_performance_reproduced=False))
    rows = []
    for planned in commands:
        print("MR3D_LAUNCH " + shlex.join(planned["command"]), flush=True)
        resources = run_monitored(planned["command"], planned["output"], profile=planned["measurement_profile"])
        path = Path(planned["output"]) / "stats.json"
        stats = json.loads(path.read_text()) if path.is_file() else {}
        passed = (resources["returncode"] == 0 and resources["memory_passed"]
                  and stats.get("status") == "passed" and stats.get("completed_indices") == list(range(19))
                  and stats.get("max_independent_residual", float("inf")) <= 1e-6)
        row = dict(planned, status="passed" if passed else "failed",
                   complete_forward_s=stats.get("end_to_end_wall_s"),
                   peak_job_pss_bytes=resources["peak_job_pss_bytes"],
                   max_independent_residual=stats.get("max_independent_residual"),
                   failure=stats.get("failure"))
        rows.append(row)
        summarize(output, rows)
        if not passed:
            print("MR3D run failed; inspect " + str(planned["output"]), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
