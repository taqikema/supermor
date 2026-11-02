"""Launch four methods over the paper's seven fixed-band frequency counts."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'core'), str(ROOT)]

from runs.case2_layered_sphere.run_mpi_scaling import add_arguments, execute_cases

FREQUENCY_COUNTS = (2, 8, 10, 16, 20, 30, 40)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--frequency-counts', nargs='+', type=int,
                        choices=FREQUENCY_COUNTS, default=FREQUENCY_COUNTS)
    parser.add_argument('--kernel', choices=['current'], default='current')
    args = parser.parse_args(argv)
    cases = [dict(method=method, ranks=8, n_frequency=count, label='N'+str(count))
             for count in args.frequency_counts for method in args.methods]
    return execute_cases(args, cases, 'frequency_count')


if __name__ == '__main__':
    sys.exit(main())
