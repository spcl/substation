import sys
import ast

with open(sys.argv[1], 'r') as fp:
    lines = fp.readlines()

state = None

print('A,B,C,optype,trans_a,trans_b,M,N,K,lda,ldb,ldc,stride_a,stride_b,stride_c,batch')

for line in lines:
    if 'ERROR' in line: continue
    if '===' in line:
        if state:
            if 'BATCH' in state:
                if state['BATCH'] is None:
                    print(state['config'][1:-1], 'GEMM', state['ta'], state['tb'], 
                          state['M'], state['N'], state['K'],
                          state['lda'], state['ldb'], state['ldc'], sep=',')
                else:
                    print(state['config'][1:-1], 'BMM', state['ta'], state['tb'], 
                          state['M'], state['N'], state['K'],
                          state['lda'], state['ldb'], state['ldc'], 
                          state['stride_a'], state['stride_b'], 
                          state['stride_c'], state['BATCH'], sep=',')
        state = {}
    else:
        cfg = ast.literal_eval(line)
        if 'swap' in cfg or 'config' in cfg:
            state.update(cfg)
        else:
            for elem in ('M', 'N', 'K'): # Indirection
                if elem in state:
                    state[elem] = cfg[state[elem]]


