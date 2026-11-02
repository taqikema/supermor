"""Run the paper's four-frequency DFOM/SUPERMOR/emg3d cross-validation.

Numerical workers keep their independent production paths. This coordinator
only resolves inputs, launches them, measures the process tree and merges
accepted frequency shards. Execute this coordinator once, outside mpiexec.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "core"), str(ROOT)]

from runs.common.mesh_inputs import get_case_mesh
from runs.common.result_io import output_directory, write_json
from runs.common.measurement import run_monitored
from runs.common.receiver_metrics import compare_archives, nrmsd

if __package__:
    from .validation_common import merge_responses
else:
    from validation_common import merge_responses


FORMAL_RANKS = {"DFOM": 16, "SUPERMOR": 32, "EMG3D": 1}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT")


def commands(args, output):
    selected = tuple(FORMAL_RANKS) if args.method == "ALL" else (args.method,)
    result = {}
    for method in selected:
        indices = ["--indices", args.indices]
        data = ["--data-root", str(args.data_root.resolve())] if args.data_root else []
        folder = output / method
        common = [*data, *indices, "--output", str(folder)]
        if method == "EMG3D":
            command = [str(args.emg3d_python or sys.executable), str(HERE / "emg3d_worker.py"),
                       "--stage", "solve", *common]
            if args.emg3d_source:
                command += ["--source", str(args.emg3d_source.resolve())]
            if not args.check_inputs:
                command = ["taskset", "-c", str(min(os.sched_getaffinity(0))), *command]
        else:
            stage = "dfom" if method == "DFOM" else "sumor"
            command = [str(args.custem_python or sys.executable), str(HERE / "custem_worker.py"),
                       "--stage", stage, *common]
            if not args.check_inputs:
                command = [*shlex.split(args.mpiexec), "-n", str(FORMAL_RANKS[method]), *command]
        if args.check_inputs:
            command.append("--check-inputs")
        result[method] = command
    return result


def component_nrmsd(actual_path, reference_path):
    """Paper Fig.3/4 and Table3 use Ex/Hy per-component symmetric NRMSD."""
    rows = []
    with np.load(actual_path, allow_pickle=False) as actual, np.load(reference_path, allow_pickle=False) as reference:
        for field, component in (("E", 0), ("H", 1)):
            error = nrmsd(actual[field][..., component], reference[field][..., component])
            for index, frequency in enumerate(actual["frequencies_hz"]):
                rows.append(dict(component="Ex" if field == "E" else "Hy", index=index,
                                 frequency_hz=float(frequency),
                                 p95_percent=float(np.percentile(error[index], 95)),
                                 p99_percent=float(np.percentile(error[index], 99)),
                                 max_percent=float(error[index].max())))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=str.upper, default="ALL",
                        choices=("ALL", *FORMAL_RANKS))
    parser.add_argument("--data-root", type=Path, help="input data root, default checkout/data")
    parser.add_argument("--output", type=Path, help="fresh experiment output directory")
    parser.add_argument("--indices", default="0,1,2,3")
    parser.add_argument("--custem-python", type=Path)
    parser.add_argument("--emg3d-python", type=Path)
    parser.add_argument("--emg3d-source", type=Path)
    parser.add_argument("--mpiexec", default=os.environ.get("MPIEXEC", "mpiexec"))
    parser.add_argument("--measurement-profile", choices=("batch_r03", "fat09", "later"), default="later")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print commands without creating outputs")
    mode.add_argument("--check-inputs", action="store_true", help="read/audit inputs without numerical libraries")
    args = parser.parse_args(argv)
    if __package__:
        from .custem_worker import parse_indices
    else:
        from custem_worker import parse_indices
    parse_indices(args.indices)
    # Resolve the full input contract before launching any large computation.
    selected = tuple(FORMAL_RANKS) if args.method == "ALL" else (args.method,)
    for method in selected:
        get_case_mesh("block_emg3d" if method == "EMG3D" else "block_cross",
                      data_root=args.data_root, require=not args.dry_run)
    preview = (args.output or ROOT / "outputs/case1_block/cross_validation/NEW_RUN").resolve()
    if args.dry_run:
        print(json.dumps({name: command for name, command in commands(args, preview).items()}, indent=2))
        return 0
    environment = dict(os.environ)
    environment.update({key: "1" for key in THREADS})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.check_inputs:
        for command in commands(args, preview).values():
            subprocess.run(command, check=True, env=environment)
        return 0
    data_directory = (args.data_root or ROOT / "data").resolve()
    if preview == data_directory or data_directory in preview.parents:
        raise ValueError("--output must be outside the input data directory")
    output = output_directory("case1_block", "cross_validation", output=args.output)
    summary = {"status": "running", "methods": {}, "formal_ranks": FORMAL_RANKS,
               "measurement_profile": args.measurement_profile, "output": str(output)}
    write_json(output / "summary.json", summary)
    for method, command in commands(args, output).items():
        record = run_monitored(command, output / method, env=environment,
                               profile=args.measurement_profile)
        merged = merge_responses(output / method, parse_indices(args.indices))
        summary["methods"][method] = dict(resources=record, merge=merged)
        write_json(output / "summary.json", summary)
        if record["returncode"] != 0 or not merged["required_complete"]:
            summary["status"] = "failed"
            write_json(output / "summary.json", summary)
            return 1
    reference = output / "DFOM/responses.npz"
    if reference.is_file():
        for method in ("SUPERMOR", "EMG3D"):
            response = output / method / "responses.npz"
            if response.is_file():
                summary["methods"][method]["receiver_metrics_vs_DFOM"] = compare_archives(response, reference)
                summary["methods"][method]["component_nrmsd_vs_DFOM"] = component_nrmsd(response, reference)
    summary["status"] = "complete"
    write_json(output / "summary.json", summary)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
