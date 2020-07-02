import argparse
import glob
import os.path
import os
from collections import Counter

parser = argparse.ArgumentParser(
    description='Concatenate all benchmark results into a single file')
parser.add_argument(
    'bmdir', type=str,
    help='Directory containing benchmark results')
parser.add_argument(
    'outdir', type=str,
    help='Directory to output results')
parser.add_argument(
    '--basename', type=str,
    help='Basename of benchmark results to merge')

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.bmdir):
        raise ValueError(f'Benchmark directory {args.bmdir} not found')
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.basename:
        pattern = args.basename + '-bm*.csv'
    else:
        pattern = '*-bm*.csv'
    bm_files = glob.glob(os.path.join(args.bmdir, pattern))
    benchmark_results = Counter()
    for f in bm_files:
        name = os.path.splitext(os.path.basename(f))[0]
        name = name.split('-')
        name = '-'.join(name[:-1])
        benchmark_results[name] += 1

    for bm, num in benchmark_results.items():
        print(f'Merging {num} results from {bm}')
        with open(os.path.join(args.outdir, bm + '-combined.csv'), 'w') as out_f:
            for i in range(num):
                with open(os.path.join(args.bmdir, f'{bm}-bm{i}.csv'), 'r') as in_f:
                    out_f.write(in_f.read())

