"""Print a pickled layout."""

import pickle
import argparse

def print_dict(d):
    for k, v in d.items():
        print(f'{k}: {v}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Print a pickled layout file')
    parser.add_argument(
        'file', type=str,
        help='Layout file to print')
    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        layouts = pickle.load(f)

    layout_input = dict((layouts['input'],))
    layout_output = dict((layouts['output'],))
    special_dims = dict(layouts['special_dims'])
    algorithms = dict(layouts['algorithms'])
    params = dict(layouts['params'])
    forward_interm = dict(layouts['forward_interm'])
    backward_interm = dict(layouts['backward_interm'])

    print('Input:')
    print_dict(layout_input)
    print('')
    print('Output:')
    print_dict(layout_output)
    print('')
    print('Parameter layouts:')
    print_dict(params)
    print('')
    print('Forward intermediates:')
    print_dict(forward_interm)
    print('')
    print('Backward intermediates:')
    print_dict(backward_interm)
    print('')
    print('GEMM algorithms:')
    print_dict(algorithms)
    print('')
    print('Special dimensions:')
    print_dict(special_dims)
