"""Launch layered-sphere methods over the MPI rank grid.

SUPERMOR defaults to the retained P1 Kg MR kernel used by figure8.
--kernel current explicitly selects the newer no-Kg variant instead.
Run this coordinator in an HPC allocation with the requested CPUs available.
"""
import argparse
import csv
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'core'), str(ROOT)]

METHODS = ('SUPERMOR', 'IFOM', 'RK', 'DFOM')
MPI_RANKS = (1, 4, 8, 16, 32, 64)
WORKER = Path(__file__).with_name('run_methods_mpi.py')


def execute_cases(args, cases, experiment):
    """Launch isolated workers; monitor memory and keep worker time separate."""
    from runs.common.result_io import output_directory, write_json
    # The selected root is displayed but not created in a dry run.
    if args.output is not None:
        proposed = args.output.expanduser().resolve()
        data_directory = (args.data_root or ROOT/'data').expanduser().resolve()
        if proposed == data_directory or data_directory in proposed.parents:
            raise ValueError('--output must be outside the input data directory')
    if args.dry_run:
        output = (args.output.expanduser().resolve() if args.output else
                  ROOT / 'outputs' / 'case2_layered_sphere' / experiment / '<new_run>')
    else:
        output = output_directory('case2_layered_sphere', experiment, output=args.output)
    plan = []
    for case in cases:
        folder = output / case['method'] / case['label']
        command = [args.mpiexec, *shlex.split(args.launcher_args), '-n', str(case['ranks']),
                   args.python, '-B', str(WORKER), '--method', case['method'],
                   '--experiment', 'test1' if experiment == 'mpi_scaling' else 'test2',
                   '--kernel', args.kernel, '--output', str(folder / 'result')]
        if args.data_root:
            command.extend(['--data-root', str(args.data_root.expanduser().resolve())])
        if 'n_frequency' in case:
            command.extend(['--n-frequency', str(case['n_frequency'])])
        plan.append(dict(**case, output=str(folder), command=command))
    payload = dict(experiment=experiment, kernel=args.kernel,
                   historical_MPI_kernel_selected=(args.kernel == 'paper_mpi') if experiment == 'mpi_scaling' else None,
                   historical_performance_reproduced=False,
                   output=str(output), cases=plan)
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0
    from runs.common.measurement import run_monitored
    from runs.common.receiver_metrics import compare_archives
    write_json(output / 'plan.json', payload)
    summaries, results = [], {}
    for case in plan:
        record = run_monitored(case['command'], case['output'], interval=args.interval)
        result = Path(case['output']) / 'result'
        stats_path = result / 'stats.json'
        stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
        row = dict(method=case['method'], mpi_ranks=case['ranks'],
                   n_frequency=case.get('n_frequency', 20),
                   status=stats.get('status', 'worker_failed'), returncode=record['returncode'],
                   complete_forward_s=stats.get('complete_forward_s'),
                   peak_job_pss_gib=record['peak_job_pss_bytes'] / 2**30,
                   kernel_selection=args.kernel, output=str(result))
        summaries.append(row)
        key = (case['method'], case.get('n_frequency', case['ranks']))
        if record['returncode'] == 0 and stats.get('status') == 'converged':
            results[key] = result / 'receivers.npz'
        write_json(output / 'summary.json', dict(cases=summaries,
                   timing_source='worker complete_forward_s',
                   memory_source='sampled simultaneous job aggregate PSS, monitor excluded'))
        if record['returncode'] != 0 and not args.keep_going:
            break
    if summaries:
        with (output / 'performance.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    # Receiver-only comparisons run after the timed worker processes exit.
    accuracy = []
    for (method, dimension), actual in results.items():
        reference = results.get(('DFOM', 8 if experiment == 'mpi_scaling' else dimension))
        if reference is not None and method != 'DFOM':
            for row in compare_archives(actual, reference):
                accuracy.append(dict(method=method, setting=dimension,
                                     reference=str(reference), **row))
    if accuracy:
        with (output / 'accuracy.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(accuracy[0]))
            writer.writeheader()
            writer.writerows(accuracy)
    print(output)
    return 0 if len(summaries) == len(plan) and all(
        row['returncode'] == 0 and row['status'] == 'converged' for row in summaries) else 1


def add_arguments(parser):
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--mpiexec', default='mpiexec')
    parser.add_argument('--launcher-args', default='', help='Additional MPI launcher arguments (shell-word syntax)')
    parser.add_argument('--interval', type=float, default=.05)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--keep-going', action='store_true')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--ranks', nargs='+', type=int, choices=MPI_RANKS, default=MPI_RANKS)
    parser.add_argument('--kernel', choices=['paper_mpi', 'current'], default='paper_mpi',
                        help='paper_mpi retains the P1 Kg kernel used by figure8; current selects the newer variant')
    args = parser.parse_args(argv)
    cases = [dict(method=method, ranks=ranks, label='p'+str(ranks))
             for ranks in args.ranks for method in args.methods]
    return execute_cases(args, cases, 'mpi_scaling')


if __name__ == '__main__':
    sys.exit(main())
