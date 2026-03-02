# Performance Benchmarking

The performance benchmarking is done through the file `Performance_Benchmarking.py`.
The script was built to be scalable if any kernels were to be added to the project.


## Overview

The benchmark script performs the following tasks:
* Measures per-kernel runtimes across a range of $N$ (from 10,000 to 3,000,000).
* Logs the timing results to a CSV file (`benchmark_results.csv`).
* Produces scaling plots for individual kernels and an $O(N \log N)$ reference overlay for tree-based operations.
* Generates a pie chart displaying the kernel runtime shares at the median $N$ value.

## Dependencies

To run the benchmarking script, ensure you have the project dependencies.

## How to Modify the Benchmark

You can easily adjust the benchmark parameters by editing the `# Benchamrk configuration` section directly within `Performance_Benchmarking.py`. 

Key variables you might want to modify include:
* **`N_VALUES`**: A list of integer particle counts to test (e.g., `[10_000, 15_000, ...]`).
* **`WARMUP`**: The number of initial runs to perform before timing, which allows the JIT compiler and caches to warm up (default: 5).
* **`ITERATIONS`**: The number of timed runs to average for the final result (default: 10).
* **`THETA_LIST`**: The opening angles ($\theta$) used when benchmarking the `compute_forces` kernel (default: `[0.3, 0.5, 0.7, 0.9, 1.1]`).
* **`TPB`**: Threads per block for GPU execution (default: 256).

## How to Run

Execute the script from your terminal:

```bash
python Performance_Benchmarking.py
```
Upon completion, the script will output the timing details to the console, save benchmark_results.csv to the current directory, and save all generated plots into the benchmark_plots/ directory.

## Results & Visualizations
Below are the results for a single RTX 5070, running on N = 250,000 bodies and $\theta = 0.5$

| Kernel | Time (ms) | Share |
| :--- | :--- | :--- |
| compute_forces | 16.881 | 83.6% |
| compute_multipoles | 2.010 | 10.0% |
| integrate | 0.243 | 1.2% |
| render_density | 0.196 | 1.0% |
| reset_tree | 0.160 | 0.8% |
| build_cache | 0.160 | 0.8% |
| compute_codes | 0.158 | 0.8% |
| normalize_com | 0.149 | 0.7% |
| build_tree | 0.146 | 0.7% |
| find_root | 0.079 | 0.4% |
| **TOTAL** | **20.181** | **100.0%** |

### Simulation Complexity

<img src="../assets/benchmarking _examples/forces_vs_N.png" width="600"><br><br>

### kernel runtimes

<img src="../assets/benchmarking _examples/scaling_kernels.png" width="600"><br><br>
<img src="../assets/benchmarking _examples/scaling_nlogn_kernels.png" width="600"><br><br>

### Pie Chart

<img src="../assets/benchmarking _examples/total_scaling.png" width="600"><br><br>