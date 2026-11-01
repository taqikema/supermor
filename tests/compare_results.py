"""Compare source baselines at fixed MPI size, or physical E/H across MPI."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Baseline .npz path")
    parser.add_argument("candidate", type=Path, help="Candidate .npz path")
    parser.add_argument("--cross-mpi", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rtol, atol = (1e-4 if args.cross_mpi else 1e-10), 1e-12
    results = {}
    with np.load(args.reference) as reference, np.load(args.candidate) as candidate:
        if set(reference.files) != set(candidate.files):
            raise AssertionError("Baseline and candidate array keys differ")
        for key in reference.files:
            if args.cross_mpi and not key.endswith(("_E", "_H")):
                continue
            if not args.cross_mpi and key.endswith("_residuals"):
                # These are verified independently against the equation threshold.
                if np.max(candidate[key]) > 1.01e-6:
                    raise AssertionError(f"{key}: original equation threshold exceeded")
                continue
            expected, actual = reference[key], candidate[key]
            if expected.shape != actual.shape or not np.isfinite(actual).all():
                raise AssertionError(f"{key}: shape mismatch or nonfinite values")
            difference, scale = float(np.linalg.norm(actual - expected)), float(np.linalg.norm(expected))
            accepted = difference <= atol + rtol * scale
            results[key] = dict(difference_l2=difference, reference_l2=scale,
                                relative_l2=difference / scale if scale else 0., accepted=accepted)
            if not accepted:
                raise AssertionError(f"{key}: {results[key]}")
    if not args.cross_mpi:
        old = json.loads(args.reference.with_suffix(".json").read_text())
        new = json.loads(args.candidate.with_suffix(".json").read_text())
        if old["mpi_ranks"] != new["mpi_ranks"] or old["summaries"] != new["summaries"]:
            raise AssertionError("Fixed-MPI algorithm summaries differ")
    report = dict(accepted=True, rtol=rtol, atol=atol, arrays=len(results),
                  maximum_relative_l2=max(r["relative_l2"] for r in results.values()), results=results)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))


if __name__ == "__main__":
    main()
