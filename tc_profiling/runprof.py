import sys
import os
import os.path
import tqdm

def get_local_rank():
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    else:
        raise RuntimeError('Cannot get local rank')


def get_local_size():
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE'])
    elif 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    elif 'SLURM_NTASKS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        raise RuntimeError('Cannot get local comm size')


def get_world_rank():
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    else:
        raise RuntimeError('Cannot get world rank')


def get_world_size():
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    else:
        raise RuntimeError('Cannot get world size')

with open(sys.argv[1], 'r') as fp:
    lines = list(fp.readlines())[1:]
    # Split lines between ranks.
    lines_per_rank = [len(lines) // get_world_size()] * get_world_size()
    remainder = len(lines) % get_world_size()
    for i in range(remainder): lines_per_rank[i] += 1
    lines_prefix_sum = [0]
    lines_sum = 0
    for i in range(get_world_size()):
        lines_sum += lines_per_rank[i]
        lines_prefix_sum.append(lines_sum)
    start_line = lines_prefix_sum[get_world_rank()]
    end_line = lines_prefix_sum[get_world_rank() + 1]
    print(f'{get_world_rank()}: Handling lines {start_line}:{end_line}, {len(lines)} total')
    lines = lines[start_line:end_line]
    # Select the right CUDA device.
    os.putenv('CUDA_VISIBLE_DEVICES', str(get_local_rank()))
    output_file = os.path.abspath(os.path.splitext(sys.argv[1])[0] + f'-bm{get_world_rank()}.csv')
    for line in lines:
        values = list(line[:-1].split(','))
        # Work on the pruned format.
        if values[-4:] == ['', '', '', '']:
            values[-4:] = [0, 0, 0, 1]
        optype,trans_a,trans_b,M,N,K,lda,ldb,ldc,stride_a,stride_b,stride_c,batch = values
        impls = list(range(24)) + [f'tc{i}' for i in range(16)] + [f'32{i}' for i in range(24)] + ['_default', 'tc_default', '32_default']
        for impl in impls:
            #title = A + ':' + B + ':' + C + ':' + str(impl)
            title = str(impl)
            os.system(f'''./gemm{impl} {title} {M} {N} {K} {lda} {ldb} {ldc} {trans_a} {trans_b} {stride_a} {stride_b} {stride_c} {batch} >> {output_file}''')

