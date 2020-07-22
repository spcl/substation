import os
import os.path
import functools
import numpy as np
import pandas as pd


_int_kernel_cols = ['h', 'b', 'j', 'k', 'u', 'n', 'p']


def load_kernel(filename, cache=True):
    if cache:
        cache_filename = filename + '.pkl'
        if os.path.exists(cache_filename):
            return pd.read_pickle(cache_filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Determine columns.
    parts = lines[0].split(' ')
    # -1 to skip the final 's'.
    columns = [parts[i].lower() for i in range(0, len(parts) - 1, 3)]
    data = {c: [] for c in columns}
    for line in lines:
        parts = line.split(' ')
        for i in range(0, len(parts) - 1, 3):
            data[parts[i].lower()].append(parts[i+2].lower())
    # Convert to milliseconds.
    data['time'] = [float(t)*1000 for t in data['time']]
    # Try to convert to ints.
    for col in _int_kernel_cols:
        if col in data:
            data[col] = [int(x) for x in data[col]]
    df = pd.DataFrame(data=data)
    def translate_layouts(x):
        if isinstance(x, str):
            x = x.translate({ord('n'): 'i'})
        return x
    df = df.applymap(translate_layouts)
    if cache:
        df.to_pickle(cache_filename)
    return df


def load_tc(filename, cache=True):
    if cache:
        cache_filename = filename + '.pkl'
        if os.path.exists(cache_filename):
            return pd.read_pickle(cache_filename)
    df = pd.read_csv(filename)
    df.rename(columns={'Time': 'time'}, inplace=True)
    if cache:
        df.to_pickle(cache_filename)
    return df
