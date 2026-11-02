"""Receiver-only paper accuracy metrics, evaluated outside forward timing."""
import argparse
import csv
from pathlib import Path

import numpy as np


def relative_metrics(actual, reference):
    """Return one frequency's global L2 and maximum pointwise vector errors.

    A nonzero discrepancy at a zero reference has infinite relative error;
    no epsilon floor or discarded receiver conceals that case.
    """
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[1] != 3:
        raise ValueError("Expected matching (receiver,3) complex arrays")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Receiver arrays must be finite")
    difference = np.linalg.norm(actual - reference, axis=1)
    scale = np.linalg.norm(reference, axis=1)
    pointwise = np.zeros_like(scale)
    np.divide(difference, scale, out=pointwise, where=scale != 0)
    pointwise[(scale == 0) & (difference != 0)] = np.inf
    total_scale, total_difference = np.linalg.norm(reference), np.linalg.norm(actual - reference)
    overall = total_difference / total_scale if total_scale else (0. if not total_difference else np.inf)
    return {"relative_L2_pct": float(100 * overall),
            "max_pointwise_pct": float(100 * pointwise.max(initial=0)),
            "zero_reference_points": int(np.count_nonzero(scale == 0))}


def nrmsd(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError("Mismatched response shapes")
    scale = np.abs(actual) + np.abs(reference)
    result = np.zeros(scale.shape, float)
    np.divide(200 * np.abs(actual - reference), scale, out=result, where=scale != 0)
    return result


def compare_archives(actual_path, reference_path):
    with np.load(actual_path, allow_pickle=False) as a, np.load(reference_path, allow_pickle=False) as b:
        for key in ("frequencies_hz", "rx_xyz", "indices"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError("Receiver archive identity mismatch: " + key)
        if not np.array_equal(a["indices"], np.arange(len(a["frequencies_hz"]))):
            raise ValueError("Comparison requires all ordered frequency indices")
        rows = []
        for component in ("E", "H"):
            shape = (len(a["frequencies_hz"]), len(a["rx_xyz"]), 3)
            if a[component].shape != shape or b[component].shape != shape:
                raise ValueError("Wrong receiver array shape: " + component)
            for index, frequency in enumerate(a["frequencies_hz"]):
                rows.append(dict(component=component, index=index, frequency_hz=float(frequency),
                                 **relative_metrics(a[component][index], b[component][index])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = compare_archives(args.actual, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
