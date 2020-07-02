import dace
from dace import data
import numpy as np
import itertools
import tqdm
import argparse
import os
import os.path

def test_configuration(einsum_str, adesc, bdesc):
    @dace.program
    def einsumtest(A: adesc, B: bdesc):
        for i in range(100):
            result = np.einsum(einsum_str, A, B)

    print('===')
    print('{"config": "', ','.join(einsum_str.split('->')), '"}')

    # Create GPU SDFG
    from dace.transformation.dataflow import MapTiling
    dace.libraries.blas.MatMul.default_implementation = 'cuBLAS'
    sdfg = einsumtest.to_sdfg()
    #sdfg.apply_transformations(GPUTransformSDFG, {'strict_transform': False})
    #sdfg.apply_gpu_transformations()
    sdfg.expand_library_nodes()

    #for node, _ in sdfg.all_nodes_recursive():
    #    if isinstance(node, dace.nodes.MapEntry):
    #        node.map.flatten = True

    # Mark map or tasklet for instrumentation
    #for node, _ in sdfg.all_nodes_recursive():
    #    if isinstance(node, (dace.nodes.MapEntry, dace.nodes.Tasklet)):
    #        node.instrument = dace.InstrumentationType.CUDA_Events
    #        #print("Instrumenting", node)
    #        break
    #else:
    #    raise RuntimeError('No node was instrumented')

    # Get implementation
    implementation_used = 'pure'
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.NestedSDFG):
            if 'sBB' in node.symbol_mapping:
                print(node.symbol_mapping)

        if isinstance(node, dace.nodes.Tasklet):
            if 'gemm(' in node.code:
                implementation_used = 'GEMM'
            elif 'gemmStridedBatched' in node.code:
                implementation_used = 'BMM'
            break

    #code = sdfg.generate_code()
#    if 'gemm(' in code[0].code:
#        implementation_used = 'GEMM'
#    elif 'gemmStridedBatched' in code[0].code:
#        implementation_used = 'BMM'
#
    #return None, implementation_used
    if implementation_used == 'pure':
        print('{}\n{}')
        return None, None

    '''
    A = np.random.rand(*adesc.shape).astype(np.float16)
    B = np.random.rand(*bdesc.shape).astype(np.float16)

    csdfg = sdfg.compile(optimizer=False)
    csdfg(A=A, B=B)
    return sdfg.get_latest_report(), implementation_used
    '''
    return None, None

def repval(report):
    return list(map(str, v.values()))
    #return list(map(str, next(v for v in report.entries.values())))
    
# b: batch
# h: heads
# i: embedding dimension
# j: sequence length
# k: sequence length (same as j)
# p: projection size
# u: intermediate FF size
# q: always 3 (q, k, v - for fusion)
# v: always 2 (k, v - for fusion)
def runtest(title, einsum_str, output, dtype=dace.float16,
            sizes={'b': 8, 'h': 16, 'i': 1024, 'j': 512, 'k': 512, 'p': 64, 'u': 4096,
                   'q': 3, 'v': 2}):
    print('Generating ' + title)
    import sys
    real_stdout = sys.stdout
    sys.stdout = open(f'{output}/{title}-configs.csv', 'w')

    inputs, output = einsum_str.split('->')
    a, b = inputs.split(',')
    
    allperms = list(itertools.product(itertools.permutations(a),
                                      itertools.permutations(b),
                                      itertools.permutations(output)))
    for in1, in2, out in allperms:
        in1 = ''.join(in1)
        in2 = ''.join(in2)
        out = ''.join(out)
        einsum_perm = (in1 + ',' + in2 + '->' + out)
        adesc = data.Array(dtype, list(map(lambda k: sizes[k], in1)))
        bdesc = data.Array(dtype, list(map(lambda k: sizes[k], in2)))
        try:
            report, implementation = test_configuration(einsum_perm, adesc, bdesc)
        except:
            print('ERROR: Failed "', einsum_perm, '". Skipping')
            continue
        if report is None:
            continue
        with open(title + '.csv', 'a') as fp:
            fp.write(','.join([in1, in2, out, implementation] 
                              + repval(report)
                          ) + '\n')

    sys.stdout = real_stdout


def multihead_attention(sizes, output):
    # Forward
    runtest("Q", "phi,ibj->phbj", output, sizes=sizes) # Assuming same sequence length, K and V should be the same as Q
    runtest("lin1", "bji,ui->bju", output, sizes=sizes)
    runtest("lin2", "bju,iu->bji", output, sizes=sizes)
    runtest("out", "phi,phbj->bij", output, sizes=sizes)
    runtest("QKT", "phbk,phbj->hbjk", output, sizes=sizes)
    runtest("gamma", "phbk,hbjk->phbj", output, sizes=sizes)
    runtest("QKV-fused", "qphi,ibj->qphbj", output, sizes=sizes)
    runtest("KV-fused", "vphi,ibj->vphbj", output, sizes=sizes)

    # Backward
    runtest('dWlin2', 'bji,bju->iu', output, sizes=sizes)
    runtest('dXlin2', 'bji,iu->bju', output, sizes=sizes)
    runtest('dWlin1', 'bju,bji->ui', output, sizes=sizes)
    runtest('dXlin1', 'bju,ui->bji', output, sizes=sizes)
    runtest('dWout', 'phbj,bij->phi', output, sizes=sizes)
    runtest('dXout', 'phi,bij->phbj', output, sizes=sizes)
    runtest('dX2gamma', 'phbj,hbjk->phbk', output, sizes=sizes)  # dVV
    runtest('dX1gamma', 'phbk,phbj->hbjk', output, sizes=sizes)  # dAlpha
    runtest('dX2QKT', 'phbj,hbjk->phbk', output, sizes=sizes)  # dKK
    runtest('dX1QKT', 'phbk,hbjk->phbj', output, sizes=sizes)  # dVV
    runtest('dWQ', 'ibj,phbj->phi', output, sizes=sizes)
    runtest('dXQ', 'phi,phbj->ibj', output, sizes=sizes)
    runtest('dWQK-fused', 'ibj,vphbj->vphi', output, sizes=sizes)
    runtest('dXQK-fused', 'vphi,vphbj->ibj', output, sizes=sizes)
    runtest('dWQKV-fused', 'ibj,qphbj->qphi', output, sizes=sizes)
    runtest('dXQKV-fused', 'qphi,qphbj->ibj', output, sizes=sizes)
    

parser = argparse.ArgumentParser(
    description='Generate attention benchmarks (next step: run through makecsv.py)')
parser.add_argument(
    '--output', type=str, required=True,
    help='Output directory')
parser.add_argument(
    '--b', type=int, required=True,
    help='Batch size')
parser.add_argument(
    '--h', type=int, required=True,
    help='Number of heads')
parser.add_argument(
    '--i', type=int, required=True,
    help='Embedding size')
parser.add_argument(
    '--j', type=int, required=True,
    help='Sequence length')
parser.add_argument(
    '--k', type=int, default=None,
    help='Sequence length from encoder to decoder')
parser.add_argument(
    '--p', type=int, default=None,
    help='Projection size')
parser.add_argument(
    '--u', type=int, default=None,
    help='Intermediate size')

if __name__ == '__main__':
    dace.Config.set('debugprint', value=False)
    dace.Config.set('optimizer', 'automatic_strict_transformations', value=False)

    args = parser.parse_args()
    if args.k is None:
        args.k = args.j
    if args.p is None:
        args.p = args.i // args.h
    if args.u is None:
        args.u = 4 * args.i
    sizes = {'b': args.b, 'h': args.h, 'i': args.i,
             'j': args.j, 'k': args.k, 'p': args.p,
             'u': args.u, 'q': 3, 'v': 2}
    # Record sizes for reference:
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(args.output + '/size-params.txt', 'w') as f:
        f.write(str(sizes))

    multihead_attention(sizes, args.output)

    import ctypes
    _cudart = ctypes.CDLL('libcudart.so')
    _cudart.cudaDeviceReset()
    import os
    os._exit(0)
