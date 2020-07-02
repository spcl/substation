import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import functools
import os
from typing import Any, List

parser = argparse.ArgumentParser(
    description='Parse/reassemble tensor contraction results')
parser.add_argument(
    'directory', type=str,
    help='Directory with results and configuration files')

FIELDS = [
    'optype', 'M', 'N', 'K', 'trans_a', 'trans_b', 'lda', 'ldb', 'ldc',
    'stride_a', 'stride_b', 'stride_c', 'batch'
]


def match(df: pd.DataFrame, values: List[Any]):
    for i, val in enumerate(values):
        try:
            values[i] = int(val)
        except ValueError:
            pass
    matchcond = functools.reduce(lambda a, b: a & b,
                                 (df[col] == val
                                  for col, val in zip(FIELDS, values)))
    return matchcond


def parse(name: str, directory: str, file: str, bmfiles: List[str]):
    df = pd.read_csv(directory + '/' + file)
    result = pd.DataFrame()

    # Fill in new column with times
    df['Time'] = -1.0
    df['Implementation'] = -1

    # Fill in NaN columns for non-BMMs
    df['stride_a'] = df['stride_a'].fillna(0)
    df['stride_b'] = df['stride_b'].fillna(0)
    df['stride_c'] = df['stride_c'].fillna(0)
    df['batch'] = df['batch'].fillna(1)

    for bmf in bmfiles:
        print('  ', bmf)
        with open(directory + '/' + bmf, 'r') as fp:
            for line in fp.readlines():
                tokenized = line.split(',')
                if len(tokenized) > 15:
                    time = np.median(
                        np.array(tokenized[14:-1], dtype=np.float64))
                    implementation = tokenized[0]
                    columns = tokenized[1:14]
                    dfm = match(df, columns)
                    df.loc[dfm, 'Time'] = time
                    df.loc[dfm, 'Implementation'] = implementation
                    result = result.append(df[dfm])

    result.to_csv(directory + '/' + name + '-result.csv', index=False)


if __name__ == '__main__':
    args = parser.parse_args()

    files = {}
    bmfiles = defaultdict(list)

    for filename in os.listdir(args.directory):
        if not filename.endswith('.csv'):
            continue
        fname = filename[:-4]
        splitname = fname.split('-')
        if 'pruned-bm' in fname:
            if 'fused' in filename:
                bmfiles[splitname[0] + '-' + splitname[1]].append(filename)
            else:
                bmfiles[splitname[0]].append(filename)
        elif len(splitname) == 1:
            files[splitname[0]] = filename
        elif len(splitname) == 2 and 'fused' in fname:
            files[splitname[0] + '-' + splitname[1]] = filename

    for tensor in files.keys():
        print('Processing %s...' % tensor)
        parse(tensor, args.directory, files[tensor], bmfiles[tensor])
