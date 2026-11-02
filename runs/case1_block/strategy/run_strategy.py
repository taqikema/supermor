"""Launch independent paper block strategies and summarize receiver errors."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "core")]

from runs.common.measurement import run_monitored
from runs.common.result_io import output_directory, write_json
from runs.common.receiver_metrics import compare_archives
from runs.case1_block.strategy.worker import check_inputs, strategy_options

STRATEGIES = ("split_u_pcg", "total_field_pcg", "total_field_lu", "single_pole", "dual_pole")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=(*STRATEGIES, "all", "reference"), default="dual_pole")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path, help="Existing same-input receivers.npz or its directory")
    parser.add_argument("--compute-reference", action="store_true")
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    check_inputs(args.data_root)
    names = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    if args.compute_reference and "reference" not in names:
        names.insert(0, "reference")
    if args.dry_run:
        print(json.dumps({name: strategy_options(name) for name in names}, indent=2))
        return 0
    output = output_directory("case1_block", "strategy", output=args.output)
    reference = args.reference
    if reference and reference.is_dir():
        reference = reference / "receivers.npz"
    summaries = []
    for name in names:
        folder = output / name
        command = [sys.executable, "-B", str(Path(__file__).with_name("worker.py")),
                   "--strategy", name, "--output", str(folder)]
        if args.data_root:
            command += ["--data-root", str(args.data_root.resolve())]
        if args.save_fields:
            command += ["--save-fields"]
        resources = run_monitored(command, folder)
        stats_path = folder / "stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        summary = dict(strategy=name, returncode=resources["returncode"],
            status=stats.get("status", "missing_stats"), dynamic_basis_columns=stats.get("dynamic_basis_columns"),
            complete_forward_s=stats.get("complete_forward_s"),
            peak_job_pss_GiB=resources["peak_job_pss_bytes"] / 2**30)
        if resources["returncode"]:
            summaries.append(summary)
            write_json(output / "summary.json", summaries)
            print("Worker failed; see " + str(folder / "run.log"), file=sys.stderr)
            return resources["returncode"]
        if name == "reference":
            reference = folder / "receivers.npz"
        elif reference:
            rows = compare_archives(folder / "receivers.npz", reference)
            with (folder / "frequency_metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for component in ("E", "H"):
                summary[component + "_max_pointwise_pct"] = max(
                    row["max_pointwise_pct"] for row in rows if row["component"] == component)
            summary["reference"] = str(reference.resolve())
        summaries.append(summary)
        write_json(output / "summary.json", summaries)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
