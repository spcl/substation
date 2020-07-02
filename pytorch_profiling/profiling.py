"""Utilities for profiling PyTorch."""

import statistics

import torch
import torch.cuda
import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pycudaprof
import gpu_wait

# Validate that we have CUDA and initialize it on load.
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError('No CUDA support or GPUs available')
torch.cuda.init()

# A device for CUDA.
cuda_device = torch.device('cuda:0')

_DEFAULT_WAIT_TIME = 0.001  # Default wait kernel time, 1 ms.


"""Time CUDA kernels with CUDA events."""
class CudaTimer:
    def __init__(self, n):
        """Support timing n regions."""
        self.events = []
        for _ in range(n + 1):
            self.events.append(torch.cuda.Event(enable_timing=True))
        self.cur = 0

    def start(self):
        """Start timing the first region."""
        self.events[0].record()
        self.cur = 1

    def next(self):
        """Start timing for the next region."""
        if self.cur >= len(self.events):
            raise RuntimeError('Trying to time too many regions')
        self.events[self.cur].record()
        self.cur += 1

    def end(self):
        """Finish timing."""
        if self.cur != len(self.events):
            raise RuntimeError('Called end too early')
        self.events[-1].synchronize()

    def get_times(self):
        """Return measured time for each region."""
        times = []
        for i in range(1, len(self.events)):
            times.append(self.events[i-1].elapsed_time(self.events[i]))
        return times


def time_one(funcs, launch_wait=True, timer=None, name='', func_names=None):
    """Run and time a single iteration of funcs.

    funcs is a list functions that do GPU work.
    
    launch_wait, if True, launches a wait kernel before measuring to hide
    kernel launch latency.

    timer is an already existing instance of CudaTimer. If None, one will be
    created.

    name will be used for NVTX ranges.

    func_names is a list of names to be used for NVTX ranges for each
    function.

    Returns the time, in ms, of each function in funcs.

    """
    if timer is None:
        timer = CudaTimer(len(funcs))
    if launch_wait:
        gpu_wait.wait(_DEFAULT_WAIT_TIME)
    torch.cuda.nvtx.range_push(name + ' Iter')
    timer.start()
    if not func_names:
        func_names = ['func ' + str(i) for i in range(len(funcs))]
    for f, name in zip(funcs, func_names):
        torch.cuda.nvtx.range_push(name)
        f()
        torch.cuda.nvtx.range_pop()
        timer.next()
    timer.end()
    torch.cuda.nvtx.range_pop()
    return timer.get_times()


def time_funcs(funcs, name='', func_names=None, num_iters=100, warmups=5,
               launch_wait=True):
    """Run and time funcs.

    funcs is a list of functions that do GPU work.

    name will be used for NVTX ranges.

    func_names is a list of names to be used for NVTX ranges for each
    function.

    num_iters is the number of iterations to perform.

    warmups is the number of warmup iterations to perform.

    launch_wait, if True, launches a wait kernel before measuring to hide
    kernel launch latency.

    Returns the time, in ms, of each function in funcs on each
    iteration.

    """
    timer = CudaTimer(len(funcs))
    pycudaprof.start_cuda_profiling()
    torch.cuda.nvtx.range_push(name + ' Warmup')
    for _ in range(warmups):
        for f in funcs: f()
    torch.cuda.nvtx.range_pop()
    times = [list() for _ in range(len(funcs))]
    for _ in range(num_iters):
        iter_times = time_one(
            funcs, launch_wait=launch_wait, timer=timer, name=name,
            func_names=func_names)
        for i, t in enumerate(iter_times):
            times[i].append(t)
    pycudaprof.stop_cuda_profiling()
    return times


def print_time_statistics(times, func_names):
    """Print timing statistics.

    times is as produced by time_funcs.

    func_names is a name to use for each function timed.

    """
    headers = ['Name', 'Min', 'Mean', 'Median', 'Stdev', 'Max']
    rows = []
    for name, func_time in zip(func_names, times):
        rows.append(
            [name,
             min(func_time),
             statistics.mean(func_time),
             statistics.median(func_time),
             statistics.stdev(func_time) if len(func_time) > 1 else 0.0,
             max(func_time)])
    print(tabulate.tabulate(
        rows, headers=headers, floatfmt='.4f', tablefmt='github'))

def plot_violins(times, func_names, output_file):
    """Save violin plots to output_file."""
    fig, axes = plt.subplots(1, len(func_names))
    for name, func_time, ax in zip(func_names, times, axes):
        sns.violinplot(y=func_time, ax=ax)
        ax.set_ylabel('Time (ms)')
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(output_file)

def generate_batch(batch_size, seq_len, embed_size):
    """Generate batch data in (sequence, batch, embedding) order."""
    return torch.rand((seq_len, batch_size, embed_size))

def gen_attention_mask(q_seq_len, k_seq_len):
    """Generate a causal attention mask."""
    return torch.triu(
        torch.full((q_seq_len, k_seq_len), float('-inf')),
        diagonal=1)
