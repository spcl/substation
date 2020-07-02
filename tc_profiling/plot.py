import ast
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
from typing import Any, Dict, List

# Number of timing repetitions in CSV files
REPS = 96


def get_params(basedir: str, dirname: str) -> Dict[str, Any]:
    fname = os.path.join(dirname, 'size-params.txt')
    nettype = dirname.split('-')[0]
    with open(os.path.join(basedir, fname), 'r') as fp:
        result = {'network': nettype}
        result.update(ast.literal_eval(fp.readlines()[0]))
        return result


def get_tc_name(filename: str) -> str:
    # Remove "-bm*.csv" part
    fileparts = filename.split('-')[:-1]
    if fileparts[-1] == 'extra':  # Remove "-extra"
        fileparts = fileparts[:-1]
    return '-'.join(fileparts)  # Reconstruct remaining parts


def median_of_columns(df: pd.DataFrame, columns: List[str],
                      outcol: str) -> pd.DataFrame:
    other_columns = set(df.columns) - set(columns)
    med = df.groupby(list(other_columns))[columns].apply(np.nanmedian)
    med.name = outcol
    df = df.join(med, on=list(other_columns))
    return df.drop(columns=columns)


def read_file(filepath: str, tc_type: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, names=['Title', 'Operation', 'M',
                                      'N', 'K', 'A_trans',
                                      'B_trans', 'lda', 'ldb',
                                      'ldc', 'sta', 'stb', 'stc',
                                      'batches'] + ['time%d' % i
                                                    for i in
                                                    range(REPS + 1)])
    # Filter out failed cases
    df = df[df['time0'] > 0]

    # Add type and shapes
    df['Type'] = tc_type
    # NOTE: we cannot use expand=True here because of column titles
    title_split = df['Title'].str.split(':')
    df['A_shape'] = title_split.str.get(0)
    df['B_shape'] = title_split.str.get(1)
    df['C_shape'] = title_split.str.get(2)
    df['Implementation'] = title_split.str.get(3)

    # Drop extra columns
    df = df.drop(columns=['Title', 'time%d' % REPS])

    # Compute median time
    return median_of_columns(df, ['time%d' % i for i in range(REPS)], 'Time')


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Type'] != 'Q'].copy()
    df.loc[df['Type'] == 'out', 'Type'] = 'Q / out'
    df.loc[df['Type'] == 'dX1gamma', 'Type'] = 'dX1g'
    df.loc[df['Type'] == 'dX2gamma', 'Type'] = 'dX2g'

    df.loc[df['Type'].str.count('-fused') == 1,
           'Type'] = df['Type'].str[:-6] + '\n(fused)'

    # Split by tensor cores
    df.loc[df['Implementation'].str.count('tc') == 0, 'Tensor Cores'] = False
    df.loc[df['Implementation'].str.count('tc') == 1, 'Tensor Cores'] = True
    return df


def addline(ax: plt.Axes, col: int, y: float, text: str):
    """ Add a line with some text to a violin plot. """
    style = {'PT': 'solid', 'XLA': 'dotted', 'Heur.': 'dashed'}
    ax.hlines(y, col-0.5, col+0.5, colors='k',
              linestyles=style[text], zorder=999)
    if col == 0:
        ax.text(col+0.5, y, text, verticalalignment='center')


def plot_violins(title: str, params: Dict[str, Any], df: pd.DataFrame):
    df = postprocess(df)
    df16 = df[df['Implementation'].str.count('32') == 0]
    df32 = df[df['Implementation'].str.count('32') == 1]

    # FP16 post-processing: Filter out results slower than tensor-core max time
    df16 = df16[df16['Time'] <=
                df16[df16['Tensor Cores'] == True]['Time'].max()]

    # FP16 plot
    plt.cla()
    plt.figure(figsize=(32, 9), dpi=120)
    matplotlib.rcParams.update({'font.size': 18})
    ax: plt.Axes = sns.violinplot(x='Type', y='Time', data=df16,
                                  hue='Tensor Cores', split=True, cut=0)
    ax.set_ylabel('Time [ms]')
    ax.set_title('%s FP16\n(Batch=%d, Heads=%d, Embedding=%d, Seqlen=%d)' % (
        params['network'], params['b'], params['h'], params['i'], params['j']
    ))

    times = df16[df16['Implementation'] == 'tc_default']
    for col, tick in enumerate(ax.get_xaxis().majorTicks):
        mintime = times[times['Type'] == tick.label.get_text()]['Time'].min()
        addline(ax, col, mintime, 'Heur.')

    ax.get_figure().savefig('%s-half.png' % title)

    # FP32 plot
    if len(df32) > 0:
        plt.cla()
        ax = sns.violinplot(x='Type', y='Time', data=df32)
        ax.set_ylabel('Time [ms]')
        ax.set_title('%s FP32\n(Batch=%d, Heads=%d, Embedding=%d, Seqlen=%d)' % (
            params['network'], params['b'], params['h'], params['i'], params['j']
        ))

        times = df32[df32['Implementation'] == '32_default']
        for col, tick in enumerate(ax.get_xaxis().majorTicks):
            mintime = times[times['Type'] ==
                            tick.label.get_text()]['Time'].min()
            addline(ax, col, mintime, 'Heur.')

        ax.get_figure().savefig('%s-float.png' % title)


def process_dir(basedir: str, dirname: str):
    params = get_params(basedir, dirname)
    print('Parameters:', params)
    if os.path.isfile('%s-fullfile.csv' % dirname):
        print('Found cached file')
        results = pd.read_csv('%s-fullfile.csv' % dirname)
    else:
        results = pd.DataFrame(columns=['Type', 'A_shape', 'B_shape', 'C_shape',
                                        'Implementation', 'Operation',
                                        'M', 'N', 'K', 'A_trans',
                                        'B_trans', 'lda', 'ldb', 'ldc', 'sta',
                                        'stb', 'stc', 'batches', 'Time'])
        for filename in os.listdir(os.path.join(basedir, dirname)):
            if filename.endswith('.csv'):
                tc_name = get_tc_name(filename)
                results = results.append(read_file(os.path.join(
                    basedir, dirname, filename), tc_name), ignore_index=True)

        results.to_csv('%s-fullfile.csv' % dirname)

    plot_violins(dirname, params, results)


if __name__ == '__main__':
    dirname = '.' if len(sys.argv) < 2 else sys.argv[1]
    for d in os.listdir(dirname):
        if os.path.isdir(d) and not d.startswith('.'):
            process_dir(dirname, d)
