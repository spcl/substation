import sys
import os
import os.path
import stat
import argparse

def write_batch(task, task_dir, num_nodes, base_dir, time=600, bank='exalearn'):
    script = f"""#!/bin/bash
#BSUB -J {task}
#BSUB -q pbatch
#BSUB -nnodes {num_nodes}
#BSUB -W {time}
#BSUB -cwd {base_dir}
#BSUB -o {task_dir}/out_{task}.log
#BSUB -e {task_dir}/err_{task}.log
#BSUB -G {bank}

jsrun --bind packed:8 --nrs {num_nodes} --rs_per_host 1 --tasks_per_rs 4 --launch_distribution packed --cpu_per_rs ALL_CPUS --gpu_per_rs ALL_GPUS python {base_dir}/runprof.py {task_dir}/{task}.csv
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
    'dir', type=str,
    help='Directory to output scripts to')
parser.add_argument(
    '--base_dir', type=str,
    default='/usr/WS1/dryden1/substation/tc_profiling',
    help='Directory with runprof.py')

if __name__ == '__main__':
    tasks = ['Q', 'lin1', 'lin2', 'out', 'QKT', 'gamma', 'QKV-fused', 'KV-fused']
    nodes = [1, 1, 1, 1, 4, 4, 4, 4]
    tasks += ['dWlin2', 'dXlin2', 'dWlin1', 'dXlin1', 'dWout', 'dXout',
             'dX2gamma', 'dX1gamma', 'dX2QKT', 'dX1QKT', 'dWQ', 'dXQ',
             'dWQK-fused', 'dXQK-fused', 'dWQKV-fused', 'dXQKV-fused']
    nodes += [1, 1, 1, 1, 2, 2,
             4, 4, 4, 4, 1, 1,
             4, 4, 4, 4]
    args = parser.parse_args()
    for task, n in zip(tasks, nodes):
        write_batch(task, os.path.abspath(args.dir), n, args.base_dir)
