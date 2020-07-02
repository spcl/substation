from itertools import chain, product, permutations
from functools import reduce
from string import ascii_letters
import argparse
import os.path
import pandas as pd


def prod(iterable):
    return reduce(lambda x, y: x * y, iterable, 1)


def get_gemm_opts(a_strides, b_strides, c_strides):
    """ 
    Returns GEMM argument order, transposition, and leading dimensions
    based on column-major storage from dace arrays. 
    :param a_strides: List of absolute strides for the first matrix. 
    :param b_strides: List of absolute strides for the second matrix. 
    :param c_strides: List of absolute strides for the output matrix. 
    :return: A dictionary with the following keys: swap (if True, a and b 
             should be swapped); lda, ldb, ldc (leading dimensions); ta, tb
             (whether GEMM should be called with OP_N or OP_T).
    """
    # possible order (C, row based) of dimensions in input array
    # and computed result based on
    # 1. N/T - transpose flag in cublas
    # 2. LR/RL - order in which A and B are passed into cublas
    #     k m, n k -> n m (LR, N, N)
    #     m k, n k -> n m (LR, T, N)
    #     k m, k n -> n m (LR, N, T)
    #     m k, k n -> n m (LR, T, T)
    #     m k, k n -> m n (RL, N, N)
    #     m k, n k -> m n (RL, N, T)
    #     k m, k n -> m n (RL, T, N)
    #     k m, n k -> m n (RL, T, T)
    #       |    |      |
    #     use these 3 to detect correct option

    sAM, sAK = a_strides[-2:]
    sBK, sBN = b_strides[-2:]
    sCM, sCN = c_strides[-2:]

    opts = {
        'mkm': {
            'swap': False,
            'lda': sAK,
            'ldb': sBN,
            'ldc': sCN,
            'ta': 'N',
            'tb': 'N'
        },
        'kkm': {
            'swap': False,
            'lda': sAM,
            'ldb': sBN,
            'ldc': sCN,
            'ta': 'T',
            'tb': 'N'
        },
        'mnm': {
            'swap': False,
            'lda': sAK,
            'ldb': sBK,
            'ldc': sCN,
            'ta': 'N',
            'tb': 'T'
        },
        'knm': {
            'swap': False,
            'lda': sAM,
            'ldb': sBK,
            'ldc': sCN,
            'ta': 'T',
            'tb': 'T'
        },
        'knn': {
            'swap': True,
            'lda': sAM,
            'ldb': sBK,
            'ldc': sCM,
            'ta': 'N',
            'tb': 'N'
        },
        'kkn': {
            'swap': True,
            'lda': sAM,
            'ldb': sBN,
            'ldc': sCM,
            'ta': 'N',
            'tb': 'T'
        },
        'mnn': {
            'swap': True,
            'lda': sAK,
            'ldb': sBK,
            'ldc': sCM,
            'ta': 'T',
            'tb': 'N'
        },
        'mkn': {
            'swap': True,
            'lda': sAK,
            'ldb': sBN,
            'ldc': sCM,
            'ta': 'T',
            'tb': 'T'
        },
    }

    if sAM == 1:
        optA = 'm'
    elif sAK == 1:
        optA = 'k'
    else:
        raise ValueError("sAM or sAK should be 1")

    if sBN == 1:
        optB = 'n'
    elif sBK == 1:
        optB = 'k'
    else:
        raise ValueError("sBK or sBN should be 1")

    if sCM == 1:
        optC = 'm'
    elif sCN == 1:
        optC = 'n'
    else:
        raise ValueError("sCM or sCN should be 1")

    return opts[optA + optB + optC]


def get_batchmm_opts(a_shape, a_strides, b_shape, b_strides, c_shape,
                     c_strides):
    """
    Detects whether a matrix multiplication is a batched matrix multiplication
    and returns its parameters (strides, batch size), or an empty dictionary if
    batched multiplication is not detected.
    :param a: Data descriptor for the first tensor.
    :param b: Data descriptor for the second tensor.
    :param c: Data descriptor for the output tensor (optional).
    :return: A dictionary with the following keys: sa,sb,sc (strides for a, b,
             and c); and b (batch size).
    """
    if len(a_shape) > 3 or len(b_shape) > 3 or (c_shape and len(c_shape) > 3):
        raise ValueError('Tensor dimensions too large for (batched) matrix '
                         'multiplication')
    if len(a_shape) <= 2 and len(b_shape) <= 2:
        return {}

    batch = None
    stride_a, stride_b, stride_c = 0, 0, 0
    if len(a_shape) == 3:
        batch = a_shape[0]
        stride_a = a_strides[0]
    if len(b_shape) == 3:
        if batch and batch != b_shape[0]:
            raise ValueError('Batch size mismatch for matrix multiplication')
        batch = b_shape[0]
        stride_b = b_strides[0]
    if c_shape and len(c_shape) == 3:
        if batch and batch != c_shape[0]:
            raise ValueError('Batch size mismatch for matrix multiplication')
        batch = c_shape[0]
        stride_c = c_strides[0]

    if batch is None:
        return {}

    return {'sa': stride_a, 'sb': stride_b, 'sc': stride_c, 'b': batch}


def _get_codegen_gemm_opts(ashape, astride, bshape, bstride, cshape, cstride):
    """ Get option map for GEMM code generation (with column-major order). """
    opt = get_gemm_opts(astride, bstride, cstride)
    bopt = get_batchmm_opts(ashape, astride, bshape, bstride, cshape, cstride)
    opt['M'] = ashape[-2]
    opt['N'] = bshape[-1]
    opt['K'] = ashape[-1]

    if opt['swap']:
        if bopt:
            bopt['sa'], bopt['sb'] = bopt['sb'], bopt['sa']
        opt['lda'], opt['ldb'] = opt['ldb'], opt['lda']
        opt['ta'], opt['tb'] = opt['tb'], opt['ta']
        opt['M'], opt['N'] = opt['N'], opt['M']

    if bopt:
        opt['stride_a'] = bopt['sa']
        opt['stride_b'] = bopt['sb']
        opt['stride_c'] = bopt['sc']
        opt['BATCH'] = bopt['b']
    else:
        opt['BATCH'] = None

    return opt


class EinsumParser(object):
    """ String parser for einsum. """
    def __init__(self, string):
        inout = string.split('->')
        if len(inout) == 1:
            inputs, output = string, ''
        else:
            inputs, output = inout

        for char in chain(inputs, output):
            if char not in ascii_letters + ',':
                raise ValueError(
                    'Invalid einsum string, subscript must contain'
                    ' letters, commas, and "->".')

        inputs = inputs.split(',')

        # No output given, assumed all "free" subscripts in inputs
        if len(inout) == 1:
            # Find intersection and union of all inputs for the non-outputs
            # and free inputs
            nonfree = set()
            free = set()
            for i, inp in enumerate(inputs):
                for var in set(inp):
                    if (all(var not in set(s) for s in inputs[i + 1:])
                            and var not in nonfree):
                        free.add(var)
                    else:
                        nonfree.add(var)
            output = ''.join(sorted(free))

        self.inputs = inputs
        self.output = output
        if len(inputs) != 2:
            return

        # Special case: contracting two tensors
        a, b = inputs
        c = output
        a_vars = set(a)
        b_vars = set(b)
        ab_vars = a_vars.union(b_vars)
        c_vars = set(c)
        if not ab_vars.issuperset(c_vars):
            raise ValueError('Einsum subscript string includes outputs that do'
                             ' not appear as an input')

        batch_vars = a_vars.intersection(b_vars).intersection(c_vars)
        sum_vars = a_vars.intersection(b_vars) - c_vars
        a_only_vars = a_vars - sum_vars - batch_vars
        b_only_vars = b_vars - sum_vars - batch_vars

        self.a_batch = [i for i, d in enumerate(a) if d in batch_vars]
        self.a_sum = [i for i, d in enumerate(a) if d in sum_vars]
        self.a_only = [i for i, d in enumerate(a) if d in a_only_vars]

        self.b_batch = [i for i, d in enumerate(b) if d in batch_vars]
        self.b_sum = [i for i, d in enumerate(b) if d in sum_vars]
        self.b_only = [i for i, d in enumerate(b) if d in b_only_vars]

        self.c_a_only = [i for i, d in enumerate(c) if d in a_only_vars]
        self.c_b_only = [i for i, d in enumerate(c) if d in b_only_vars]
        self.c_batch = [i for i, d in enumerate(c) if d in batch_vars]

    @staticmethod
    def _is_sequential(index_list):
        if not index_list:
            return True
        index_list = sorted(index_list)
        smallest_elem = index_list[0]
        return index_list == list(
            range(smallest_elem, smallest_elem + len(index_list)))

    def is_bmm(self):
        a, b = self.inputs
        c = self.output

        if len(self.inputs) != 2:
            return False

        # Check that batch dimension ordering is the same in a, b, and c
        if any(a[ad] != b[bd] for ad, bd in zip(self.a_batch, self.b_batch)):
            return False
        if any(a[ad] != c[cd] for ad, cd in zip(self.a_batch, self.c_batch)):
            return False
        if any(a[ad] != b[bd] for ad, bd in zip(self.a_sum, self.b_sum)):
            return False
        if any(a[ad] != c[cd] for ad, cd in zip(self.a_only, self.c_a_only)):
            return False
        if any(b[bd] != c[cd] for bd, cd in zip(self.b_only, self.c_b_only)):
            return False

        for key, val in self.fields().items():
            if not EinsumParser._is_sequential(val):
                return False
        return True

    def get_bmm(self, a_shape, b_shape, c_shape):
        if self.is_bmm():
            # Compute GEMM dimensions and strides
            result = dict(
                BATCH=prod([c_shape[dim] for dim in self.c_batch]),
                M=prod([a_shape[dim] for dim in self.a_only]),
                K=prod([a_shape[dim] for dim in self.a_sum]),
                N=prod([b_shape[dim] for dim in self.b_only]),
                sAM=prod(a_shape[self.a_only[-1] + 1:]) if self.a_only else 1,
                sAK=prod(a_shape[self.a_sum[-1] + 1:]) if self.a_sum else 1,
                sAB=prod(a_shape[self.a_batch[-1] +
                                 1:]) if self.a_batch else 1,
                sBK=prod(b_shape[self.b_sum[-1] + 1:]) if self.b_sum else 1,
                sBN=prod(b_shape[self.b_only[-1] + 1:]) if self.b_only else 1,
                sBB=prod(b_shape[self.b_batch[-1] +
                                 1:]) if self.b_batch else 1,
                sCM=prod(c_shape[self.c_a_only[-1] +
                                 1:]) if self.c_a_only else 1,
                sCN=prod(c_shape[self.c_b_only[-1] +
                                 1:]) if self.c_b_only else 1,
                sCB=prod(c_shape[self.c_batch[-1] +
                                 1:]) if self.c_batch else 1)
            if result['BATCH'] == 1:
                return _get_codegen_gemm_opts([result['M'], result['K']],
                                              [result['sAM'], result['sAK']],
                                              [result['K'], result['N']],
                                              [result['sBK'], result['sBN']],
                                              [result['M'], result['N']],
                                              [result['sCM'], result['sCN']])
            else:
                return _get_codegen_gemm_opts(
                    [result['BATCH'], result['M'], result['K']],
                    [result['sAB'], result['sAM'], result['sAK']],
                    [result['BATCH'], result['K'], result['N']],
                    [result['sBB'], result['sBK'], result['sBN']],
                    [result['BATCH'], result['M'], result['N']],
                    [result['sCB'], result['sCM'], result['sCN']])
        else:
            return None

    def fields(self):
        return {
            fname: fval
            for fname, fval in self.__dict__.items()
            if fname not in ('inputs', 'output')
        }

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)


def run_all(
    sizes={
        'b': 8,
        'h': 16,
        'i': 1024,
        'j': 512,
        'k': 512,
        'p': 64,
        'u': 4096,
        'q': 3,
        'v': 2
    },
    output='.'):
    # Forward
    # Assuming same sequence length, K and V should be the same as Q
    runtest("Q", "phi,ibj->phbj", sizes=sizes, output_dir=output)
    runtest("lin1", "bji,ui->bju", sizes=sizes, output_dir=output)
    runtest("lin2", "bju,iu->bji", sizes=sizes, output_dir=output)
    runtest("out", "phi,phbj->bij", sizes=sizes, output_dir=output)
    runtest("QKT", "phbk,phbj->hbjk", sizes=sizes, output_dir=output)
    runtest("gamma", "phbk,hbjk->phbj", sizes=sizes, output_dir=output)
    runtest("QKV-fused", "qphi,ibj->qphbj", sizes=sizes, output_dir=output)
    runtest("KV-fused", "vphi,ibj->vphbj", sizes=sizes, output_dir=output)

    # Backward
    runtest('dWlin2', 'bji,bju->iu', sizes=sizes, output_dir=output)
    runtest('dXlin2', 'bji,iu->bju', sizes=sizes, output_dir=output)
    runtest('dWlin1', 'bju,bji->ui', sizes=sizes, output_dir=output)
    runtest('dXlin1', 'bju,ui->bji', sizes=sizes, output_dir=output)
    runtest('dWout', 'phbj,bij->phi', sizes=sizes, output_dir=output)
    runtest('dXout', 'phi,bij->phbj', sizes=sizes, output_dir=output)
    runtest('dX2gamma', 'phbj,hbjk->phbk', sizes=sizes,
            output_dir=output)  # dVV
    runtest('dX1gamma', 'phbk,phbj->hbjk', sizes=sizes,
            output_dir=output)  # dAlpha
    runtest('dX2QKT', 'phbj,hbjk->phbk', sizes=sizes, output_dir=output)  # dKK
    runtest('dX1QKT', 'phbk,hbjk->phbj', sizes=sizes, output_dir=output)  # dVV
    runtest('dWQ', 'ibj,phbj->phi', sizes=sizes, output_dir=output)
    runtest('dXQ', 'phi,phbj->ibj', sizes=sizes, output_dir=output)
    runtest('dWQK-fused', 'ibj,vphbj->vphi', sizes=sizes, output_dir=output)
    runtest('dXQK-fused', 'vphi,vphbj->ibj', sizes=sizes, output_dir=output)
    runtest('dWQKV-fused', 'ibj,qphbj->qphi', sizes=sizes, output_dir=output)
    runtest('dXQKV-fused', 'qphi,qphbj->ibj', sizes=sizes, output_dir=output)


def runtest(title, einsum, sizes, output_dir):
    ins, output = einsum.split('->')
    a, b = ins.split(',')
    is_bmm = False

    with open(output_dir + '/' + title + '.csv', 'w') as fp:
        # Print header
        print(
            'A,B,C,optype,trans_a,trans_b,M,N,K,lda,ldb,ldc,stride_a,'
            'stride_b,stride_c,batch',
            file=fp)

        # Print the parameters of each permutation
        allperms = product(permutations(a), permutations(b),
                           permutations(output))
        for in1, in2, out in allperms:
            in1 = ''.join(in1)
            in2 = ''.join(in2)
            out = ''.join(out)
            einsum_perm = (in1 + ',' + in2 + '->' + out)
            e = EinsumParser(einsum_perm)

            ashp = list(map(lambda k: sizes[k], in1))
            bshp = list(map(lambda k: sizes[k], in2))
            cshp = list(map(lambda k: sizes[k], out))
            try:
                params = e.get_bmm(ashp, bshp, cshp)
            except ValueError:
                continue
            if params:
                is_bmm = True
                if params['BATCH']:
                    print(in1,
                          in2,
                          out,
                          'BMM',
                          params['ta'],
                          params['tb'],
                          params['M'],
                          params['N'],
                          params['K'],
                          params['lda'],
                          params['ldb'],
                          params['ldc'],
                          params['stride_a'],
                          params['stride_b'],
                          params['stride_c'],
                          params['BATCH'],
                          sep=',',
                          file=fp)
                else:
                    print(in1,
                          in2,
                          out,
                          'GEMM',
                          params['ta'],
                          params['tb'],
                          params['M'],
                          params['N'],
                          params['K'],
                          params['lda'],
                          params['ldb'],
                          params['ldc'],
                          sep=',',
                          file=fp)

        if not is_bmm:
            print('ERROR:', title, "IS NOT BMM")

    # Generate the pruned version.
    df = pd.read_csv(output_dir + '/' + title + '.csv')
    fdf = df.drop(columns=['A', 'B', 'C'])
    fdf.drop_duplicates(inplace=True)
    fdf.to_csv(output_dir + '/' + title + '-pruned.csv', index=False)


parser = argparse.ArgumentParser(
    description=
    'Generate attention benchmarks (next step: run through makecsv.py)')
parser.add_argument('--output',
                    type=str,
                    required=True,
                    help='Output directory')
parser.add_argument('--b', type=int, required=True, help='Batch size')
parser.add_argument('--h', type=int, required=True, help='Number of heads')
parser.add_argument('--i', type=int, required=True, help='Embedding size')
parser.add_argument('--j', type=int, required=True, help='Sequence length')
parser.add_argument('--k',
                    type=int,
                    default=None,
                    help='Sequence length from encoder to decoder')
parser.add_argument('--p', type=int, default=None, help='Projection size')
parser.add_argument('--u', type=int, default=None, help='Intermediate size')

if __name__ == '__main__':
    # import time
    # times = []
    # for i in range(10):
    #     start = time.time()
    #     runtest('test', 'phbj,hbjk->phbk', None)
    #     times.append((time.time() - start))
    # import numpy as np
    # print('Time:', np.median(np.array(times)) * 1000, 'ms')

    args = parser.parse_args()
    if args.k is None:
        args.k = args.j
    if args.p is None:
        args.p = args.i // args.h
    if args.u is None:
        args.u = 4 * args.i
    sizes = {
        'b': args.b,
        'h': args.h,
        'i': args.i,
        'j': args.j,
        'k': args.k,
        'p': args.p,
        'u': args.u,
        'q': 3,
        'v': 2
    }
    # Record sizes for reference:
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(args.output + '/size-params.txt', 'w') as f:
        f.write(str(sizes))

    run_all(sizes, args.output)
