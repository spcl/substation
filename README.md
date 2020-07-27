# Substation: Optimized Transformers :zap:

Substation is a project to optimize transformers using data movement analysis.

This code is presently at a research-and-development stage. We are actively working to make it both faster and more usable.

For more background, please see our paper, [_Data Movement Is All You Need: A Case Study on Optimizing Transformers_](https://arxiv.org/abs/2007.00072). If you use our code, please cite the paper:
```
@article{ivanov2020data,
  title={Data Movement Is All You Need: A Case Study on Optimizing Transformers},
  author={Ivanov, Andrei and Dryden, Nikoli and Ben-Nun, Tal and Li, Shigang and Hoefler, Torsten},
  journal={arXiv preprint arXiv:2007.00072},
  year={2020}
}
```

## Current Performance

We presently include configurations for two versions of a single BERT-large encoder layer:
1. Batch size 8 and max sequence length 512.
2. Batch size 96 and max sequence length 128.

These benchmarks were run on the [Lassen supercomputer](https://hpc.llnl.gov/hardware/platforms/lassen). Note that the Nvidia V100s this system uses are the SXM2 variety, with a peak of 125 tflop/s using Tensor Cores. We compare with the same transformer architecture implemented in TensorFlow (with XLA), PyTorch, and DeepSpeed. These results are with the latest version of our code, but see our paper for other details.

All times are in milliseconds (ms).

#### BERT-large, batch size 8, max sequence length 512 runtime
| PyTorch | TensorFlow+XLA | DeepSpeed | Substation
|---------|----------------|-----------|-----------
| 9.14    | 8.4            | 7.6       | 6.71

#### BERT-large, batch size 96, max sequence length 128 runtime
| PyTorch | TensorFlow+XLA | DeepSpeed | Substation
|---------|----------------|-----------|-----------
| 18.43   | n/a            | 16.19     | 15.42

## Usage

_Note: We are actively working to improve the usability for standard deep learning workflows._

Our encoder implementation is available as a PyTorch module in `pytorch_module/encoder.py`. Whenever you create a Substation encoder, you must specify an associated set of layouts and other configurations (see below for generating one yourself). We have provided the configurations used for the two BERT-large versions above as `layouts-bert-b8-l512-h16-e1024.pickle` and `layouts-bert-b96-l128-h16-e1024.pickle`, respectively. These configurations are optimized for the specific configuration and hardware, but should run for other problem sizes and on other hardware. The underlying optimized implementation for the encoder will be generated and compiled the first time you use it.

For performance benchmarking, we provide the `run_encoder.py` script. See its `--help` information for details.

### Generating New Configurations

If you want to get the best performance for your particular problem configuration and/or hardware, you will need to generate a configuration. This involves two phases: benchmarking to gather performance data, then configuration selection.

#### Benchmarking

_Warning: This can take a long time._

This exhaustively benchmarks the possible layouts (and other options) for every operator used in the encoder layer. There are two sets of benchmarks, one for tensor contractions (which uses cuBLAS) and one for our custom fused kernel implementations.

##### Tensor Contractions

These are located in `tc_profiling`.
1. Run `compile.sh` to build cuBLAS benchmarks.
2. Run `einsum_perms.py` (e.g., `einsum_perms.py --b 8 --j 512 --h 16 --i 1024`) to generate the benchmark configurations for each operator.
3. These configurations can be run with `runprof.py <config file>`.

##### Fused Kernels

These are run with the `pytorch_module/benchmark.py` script. You specify the kernel to benchmark with `--kernel name`. By default, this uses the batch size 8, sequence length 512 configuration of BERT-large. You can change the size using the `--size` argument. For example:
```
python benchmark.py --kernel softmax --size "H=16,B=96,J=128,K=128,U=4096,N=1024,P=64"
```
See its `--help` for more arguments.

You will need to run every tensor contraction and kernel benchmark.

#### Configuration Selection

These scripts are located in the `config_selection` directory. First, collect the benchmark data into a directory. You can just copy the kernel benchmark output. Use the `parse_tc_results.py` script to assemble the tensor contraction results and then copy them into the same directory.

Final configuration selection can then be run with `python optimize.py --output_config my_layouts.pickle results-dir`.

##### Advanced

The `optimize.py` script can use several strategies for performing configuration selection, controlled with the `--graph_order` argument. The default, `bp_first`, will optimize the encoder layer's backpropagation pass first, and then its forward pass. `fp_first` will optimize forward propagation first, then backpropagation. `bp_first` typically results in configurations that are faster than `fp_first`. The third option, `combined`, will optimize over forward and backpropagation simultaneously, and typically results in the fastest configurations. However, this approach is somewhat finnicky, and can often fail to find a valid layout. This can be worked around by telling the optimizer to "split" at certain variables using the `--split_vars` argument.

The `layouts-bert-b8-l512-h16-e1024.pickle` configuration was generated using `optimize.py --graph_order combined --split_vars X LN1 LN2 LIN2 DLIN2`. The `layouts-bert-b96-l128-h16-e1024.pickle` configuration was generated using `optimize.py --graph_order combined --split_vars X DROP2 LN1`.

## Contributors

This project is led by the [Scalable Parallel Computing Lab](https://spcl.inf.ethz.ch/) at ETH Zurich.

See also the [list of contributors](https://github.com/spcl/substation/graphs/contributors).
