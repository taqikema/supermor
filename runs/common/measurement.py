"""Sample the launched process tree's simultaneous aggregate PSS.

    The root monitor is excluded. RSS is retained separately. Samples scan
    processes sequentially, so the nominal interval is not a guarantee of
    synchronized OS accounting. Worker complete_forward_s is the paper time;
    wall_s here includes Python startup and shutdown.
"""
from pathlib import Path
import os
import subprocess
import time

import psutil

from .result_io import write_json

THREAD_ENVIRONMENT = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT", "VECLIB_MAXIMUM_THREADS"), "1")


def run_monitored(command, output, env=None, interval=.05, profile="later"):
    if interval <= 0 or profile not in ("batch_r03", "fat09", "later"):
        raise ValueError("Invalid sampling interval or measurement profile")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, **THREAD_ENVIRONMENT)
    environment.pop("CUSTEM_CPC_TRACE_DIR", None)
    if env:
        environment.update(env)
    started = time.monotonic()
    peak_pss = peak_rss = samples = 0
    seen, last = {}, set()
    sample_path = output / "memory_samples.csv"
    with (output / "run.log").open("w") as log, sample_path.open("w") as sample:
        sample.write("elapsed_s,processes,job_pss_bytes,job_rss_bytes\n")
        child = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, env=environment)
        root = psutil.Process(child.pid)
        if profile == "later":
            seen[(root.pid, root.create_time())] = root
        try:
            while True:
                try:
                    candidates = [root] + root.children(recursive=True)
                except psutil.Error:
                    candidates = []
                for process in candidates:
                    try:
                        seen[(process.pid, process.create_time())] = process
                    except psutil.Error:
                        pass
                total_pss = total_rss = count = 0
                last = set()
                for identity, process in tuple(seen.items()):
                    try:
                        if process.create_time() != identity[1] or not process.is_running():
                            continue
                        info = process.memory_full_info()
                        if not hasattr(info, "pss"):
                            raise RuntimeError("PSS is unavailable on this platform")
                        total_pss += info.pss
                        total_rss += info.rss
                        count += 1
                        last.add(identity)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass
                peak_pss, peak_rss = max(peak_pss, total_pss), max(peak_rss, total_rss)
                sample.write(f"{time.monotonic()-started:.9f},{count},{total_pss},{total_rss}\n")
                sample.flush()
                samples += 1
                code = child.poll()
                if code is not None:
                    break
                time.sleep(interval)
        except BaseException:
            for process in reversed(list(seen.values())):
                try:
                    process.terminate()
                except psutil.Error:
                    pass
            child.wait()
            raise
    record = dict(returncode=code, wall_s=time.monotonic()-started,
                  peak_job_pss_bytes=peak_pss, peak_job_rss_bytes=peak_rss,
                  memory_passed=peak_pss > 0, termination_reason=None,
                  measurement="synchronized_process_tree_PSS", profile=profile,
                  implementation="portable_release_v1", historical_monitor_byte_identical=False,
                  monitor_excluded=True, sample_interval_s=interval, samples=samples,
                  command=list(map(str, command)), timing_source="worker complete_forward_s, not wall_s",
                  thread_environment={key:environment.get(key) for key in THREAD_ENVIRONMENT})
    write_json(output / "resources.json", record)
    return record
