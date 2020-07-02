import sys
import os
import os.path
import stat
import argparse

def write_batch(task, size, task_dir, num_nodes, base_dir, time=720, bank='exalearn'):
    script = f"""#!/bin/bash
#BSUB -J {task}
#BSUB -q pbatch
#BSUB -nnodes {num_nodes}
#BSUB -W {time}
#BSUB -cwd {base_dir}
#BSUB -o {task_dir}/out_{task}.log
#BSUB -e {task_dir}/err_{task}.log
#BSUB -G {bank}

jsrun --bind packed:8 --nrs {num_nodes} --rs_per_host 1 --tasks_per_rs 4 --launch_distribution packed --cpu_per_rs ALL_CPUS --gpu_per_rs ALL_GPUS python {base_dir}/benchmark.py --kernel {task} --size "{size}" --out-dir {task_dir}
"""

    script_name = f'{task_dir}/bm_{task}.sh'
    with open(script_name, 'w') as f:
        f.write(script)
    # Make executable.
    st = os.stat(script_name)
    mode = st.st_mode | ((st.st_mode & 0o444) >> 2)
    os.chmod(script_name, mode)

parser = argparse.ArgumentParser(
    description='Generate batch scripts')
parser.add_argument(
    'sizes', type=str,
    help='Problem size')
parser.add_argument(
    'dir', type=str,
    help='Directory to output scripts to')
parser.add_argument(
    '--base_dir', type=str,
    default='/usr/WS1/dryden1/substation/pytorch_module',
    help='Directory with benchmark.py')

if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    kernels = ['softmax', 'bad', 'bdrln', 'blnrd', 'bsb', 'bdrlb', 'bs',
               'ebsb', 'bei', 'aib', 'baib', 'baob']
    nodes = 16
    for kernel in kernels:
        write_batch(kernel, args.sizes, os.path.abspath(args.dir), nodes, args.base_dir)
