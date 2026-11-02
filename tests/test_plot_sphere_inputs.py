"""Independent measurement fixtures for the sphere plotting adapters."""
import json
from pathlib import Path

import pytest

from figs.common.sphere_inputs import COUNTS, RANKS, load_frequency_inputs, load_mpi_inputs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(root, experiment):
    rows = []
    dimensions = RANKS if experiment == "mpi_scaling" else COUNTS
    for method in ("DFOM", "IFOM", "RK", "SUPERMOR"):
        for dimension in dimensions:
            ranks = dimension if experiment == "mpi_scaling" else 8
            n = 20 if experiment == "mpi_scaling" else dimension
            label = ("p" if experiment == "mpi_scaling" else "N") + str(dimension)
            result = root / method / label / "result"
            result.mkdir(parents=True)
            seconds = 1000. / dimension
            pss = 2**30 + dimension
            rows.append(dict(method=method, mpi_ranks=ranks, n_frequency=n,
                             status="converged", returncode=0, complete_forward_s=seconds,
                             peak_job_pss_gib=pss/2**30, output=str(result)))
            write(result/"stats.json", dict(status="converged", method=method,
                  mpi_ranks=ranks, n_frequency=n, complete_forward_s=seconds,
                  cells=100, field_dofs=200, frequencies_hz=list(range(n)),
                  kernel_selection="current", historical_core_equivalence="new run",
                  threads_per_rank=None, production_commit="fixture-independent"))
            write(result.parent/"resources.json", dict(returncode=0, peak_job_pss_bytes=pss,
                  implementation="fixture-monitor", historical_monitor_byte_identical=False))
    write(root/"summary.json", dict(cases=rows))
    return rows


def test_mpi_grid_aliases_exact_memory_and_speedup(tmp_path):
    make_run(tmp_path, "mpi_scaling")
    summary, files = load_mpi_inputs(run_dir=tmp_path)
    assert len(summary["rows"]) == 24
    assert {r["method"] for r in summary["rows"]} == {"DFOM", "IFOM", "SUMOR", "CPC"}
    for row in summary["rows"]:
        assert row["speedup"] == row["mpi_ranks"]
        assert row["peak_job_pss_bytes"] == 2**30 + row["mpi_ranks"]
        assert row["peak_pss_gib"] == row["peak_job_pss_bytes"] / 2**30
        assert row["worker_provenance"]["kernel_selection"] == "current"
        assert row["measurement_provenance"]["historical_monitor_byte_identical"] is False
    assert summary["input_kind"] == "new_run"
    assert summary["threads_per_rank"] is None
    assert len(files) == 49


def test_frequency_grid_preserves_measured_counts(tmp_path):
    make_run(tmp_path, "frequency_count")
    rows, files = load_frequency_inputs(run_dir=tmp_path)
    assert len(rows) == 28
    assert {(r["method"], r["n_frequency"]) for r in rows} == {
        (m, n) for m in ("DFOM", "IFOM", "CPC", "SUMOR") for n in COUNTS}
    assert all(r["mpi_ranks"] == 8 and r["data_origin"] == "measured" for r in rows)
    assert len(files) == 57


@pytest.mark.parametrize("fault", ["missing", "duplicate", "failed", "bad_memory", "bad_worker"])
def test_invalid_new_run_does_not_fall_back_to_paper(tmp_path, fault):
    rows = make_run(tmp_path, "mpi_scaling")
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(dict(rows[0]))
    elif fault == "failed":
        rows[0]["status"] = "failed"
    elif fault == "bad_memory":
        write(Path(rows[0]["output"]).parent/"resources.json",
              dict(returncode=0, peak_job_pss_bytes=0))
    elif fault == "bad_worker":
        p = Path(rows[0]["output"])/"stats.json"
        stats = json.loads(p.read_text()); stats["method"] = "IFOM"; write(p, stats)
    write(tmp_path/"summary.json", dict(cases=rows))
    with pytest.raises(ValueError):
        load_mpi_inputs(run_dir=tmp_path)


def test_frozen_mpi_input_preserved(tmp_path):
    frozen = {"rows": [], "description": "archived measurements", "source_files": ["original"]}
    write(tmp_path/"mpi_performance_measured.json", frozen)
    result, files = load_mpi_inputs(input_dir=tmp_path)
    assert result == frozen
    assert files == [tmp_path/"mpi_performance_measured.json"]


def test_frozen_frequency_csv_bom(tmp_path):
    (tmp_path/"performance_source.csv").write_text(
        "method,n_frequency,mpi_ranks\nSUMOR,40,8\n", encoding="utf-8-sig")
    rows, _ = load_frequency_inputs(input_dir=tmp_path)
    assert rows == [dict(method="SUMOR", n_frequency="40", mpi_ranks="8")]


@pytest.mark.parametrize("loader", [load_mpi_inputs, load_frequency_inputs])
def test_frozen_and_new_inputs_are_mutually_exclusive(loader, tmp_path):
    with pytest.raises(ValueError, match="either"):
        loader(input_dir=tmp_path, run_dir=tmp_path)
