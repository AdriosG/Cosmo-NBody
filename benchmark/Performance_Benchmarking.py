"""
benchmark.py
============
Standalone kernel benchmarking script for the N-Body cosmological simulation.

Measures per-kernel runtimes across a range of N (10^4 → 2×10^6),
logs results to CSV, and produces:
  - Time vs N plots for compute_forces (5 theta values)
  - A pie chart ("camembert") of kernel runtime shares at the median N
  - O(N log N) reference overlay on each timing plot
"""

import os
import sys
import csv
import time
import math
import warnings
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

# ---------------------------------------------------------------------------
# Import simulation kernels  (adjust paths if your project layout differs)
# ---------------------------------------------------------------------------
from src.kernels.morton import compute_codes
from src.kernels.tree_kernels import (
    build_tree,
    compute_multipoles,
    normalize_com,
    find_root,
    reset_tree_buffers,
    build_top_tree_cache,
)
from src.kernels.physics_kernels import compute_forces, integrate
from src.kernels.render_kernels import render_density
from src.utils.visualizer import get_mvp_matrix

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
N_VALUES = [
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
]

WARMUP      = 5
ITERATIONS  = 10
TPB         = 256
CACHE_SIZE  = 256
RES         = 512          # render grid resolution (kept small for speed)
BOX_SIZE    = 1.0
SOFTENING   = 0.001
G           = 1.0
THETA_LIST  = [0.3, 0.5, 0.7, 0.9, 1.1]   # 5 opening angles for forces plot
THETA_PIE   = 0.7                           # theta used in the camembert
OUTPUT_CSV  = "benchmark_results.csv"
OUTPUT_DIR  = "benchmark_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bpg(n: int, tpb: int = TPB) -> int:
    return (n + tpb - 1) // tpb


def cuda_sync():
    cp.cuda.Device().synchronize()


def timed_kernel(fn, warmup: int = WARMUP, iters: int = ITERATIONS) -> float:
    """Run fn() warmup+iters times; return mean wall-time (ms) over iters."""
    for _ in range(warmup):
        fn()
        cuda_sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        times.append((time.perf_counter() - t0) * 1e3)   # → ms

    return float(np.mean(times))

def timed_kernel_GPU(fn, warmup: int = WARMUP, iters: int = ITERATIONS) -> float:
    """
    Time a GPU kernel using CUDA events.
    Returns mean execution time in milliseconds over `iters` runs.
    """
    # Warmup — let the JIT compile and caches warm up
    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()

    times = []
    for _ in range(iters):
        start = cp.cuda.Event()
        end   = cp.cuda.Event()

        start.record()
        fn()
        end.record()
        end.synchronize()   # blocks CPU until the end event is reached on GPU

        times.append(cp.cuda.get_elapsed_time(start, end))  # ms, float

    return float(np.mean(times))

# ---------------------------------------------------------------------------
# Data factory — creates a fully valid tree state for a given N
# ---------------------------------------------------------------------------

class BenchState:
    """Allocates and populates GPU buffers needed by all kernels."""

    def __init__(self, n: int):
        self.n = n
        self.n_nodes = 2 * n - 1

        # --- positions & masses ---
        pos_np = np.random.rand(3, n).astype(np.float32)
        self.pos_x = cp.asarray(pos_np[0])
        self.pos_y = cp.asarray(pos_np[1])
        self.pos_z = cp.asarray(pos_np[2])
        self.mass   = cp.full(n, 1.0 / n, dtype=cp.float32)
        self.vel_x  = cp.zeros(n, dtype=cp.float32)
        self.vel_y  = cp.zeros(n, dtype=cp.float32)
        self.vel_z  = cp.zeros(n, dtype=cp.float32)
        self.force  = cp.zeros((n, 3), dtype=cp.float32)

        # --- morton ---
        self.codes   = cp.zeros(n, dtype=cp.uint64)
        self.indices = cp.zeros(n, dtype=cp.uint32)

        # --- tree buffers ---
        self.child    = cp.full((self.n_nodes, 2), -1, dtype=cp.int32)
        self.parent   = cp.full(self.n_nodes, -1, dtype=cp.int32)
        self.root_idx = cp.zeros(1, dtype=cp.int32)
        self.n_mass   = cp.zeros(self.n_nodes, dtype=cp.float32)
        self.n_com    = cp.zeros((self.n_nodes, 3), dtype=cp.float32)
        self.n_min    = cp.zeros((self.n_nodes, 3), dtype=cp.float32)
        self.n_max    = cp.zeros((self.n_nodes, 3), dtype=cp.float32)
        self.counters = cp.zeros(self.n_nodes, dtype=cp.int32)

        # --- top-tree cache ---
        self.top_nodes    = cp.full(CACHE_SIZE, -1, dtype=cp.int32)
        self.node_to_cache = cp.full(self.n_nodes, -1, dtype=cp.int32)

        # --- render grid ---
        self.grid = cp.zeros((RES, RES), dtype=cp.float32)

        # Build a valid tree once so force / cache benchmarks have real data
        self._build_tree()

    # ------------------------------------------------------------------ #
    def _build_tree(self):
        n, tpb = self.n, TPB

        compute_codes[bpg(n), tpb](
            self.pos_x, self.pos_y, self.pos_z,
            self.codes, self.indices, n
        )
        sort_idx = cp.argsort(self.codes)
        self.pos_x = self.pos_x[sort_idx]
        self.pos_y = self.pos_y[sort_idx]
        self.pos_z = self.pos_z[sort_idx]
        self.mass  = self.mass[sort_idx]
        self.codes = self.codes[sort_idx]

        n_nodes = self.n_nodes
        bpg_nodes = bpg(n_nodes)

        reset_tree_buffers[bpg_nodes, tpb](
            self.child, self.parent, self.counters, self.root_idx
        )
        build_tree[bpg(n), tpb](self.codes, self.child, self.parent)
        compute_multipoles[bpg(n), tpb](
            self.pos_x, self.pos_y, self.pos_z, self.mass,
            self.child, self.parent,
            self.n_mass, self.n_com, self.n_min, self.n_max,
            self.counters
        )
        normalize_com[bpg(n_nodes, 256), 256](self.n_mass, self.n_com, n_nodes)
        find_root[bpg_nodes, tpb](self.parent, self.root_idx)
        self.node_to_cache.fill(-1)
        build_top_tree_cache[1, 1](
            self.child, self.root_idx,
            self.top_nodes, self.node_to_cache, CACHE_SIZE
        )
        cuda_sync()


# ---------------------------------------------------------------------------
# Per-kernel benchmark functions
# ---------------------------------------------------------------------------

def bench_compute_codes(s: BenchState) -> float:
    def fn():
        compute_codes[bpg(s.n), TPB](
            s.pos_x, s.pos_y, s.pos_z, s.codes, s.indices, s.n
        )
    return timed_kernel(fn)


def bench_reset_tree(s: BenchState) -> float:
    bpg_n = bpg(s.n_nodes)
    def fn():
        reset_tree_buffers[bpg_n, TPB](
            s.child, s.parent, s.counters, s.root_idx
        )
    return timed_kernel(fn)


def bench_build_tree(s: BenchState) -> float:
    def fn():
        build_tree[bpg(s.n), TPB](s.codes, s.child, s.parent)
    return timed_kernel(fn)


def bench_compute_multipoles(s: BenchState) -> float:
    def fn():
        compute_multipoles[bpg(s.n), TPB](
            s.pos_x, s.pos_y, s.pos_z, s.mass,
            s.child, s.parent,
            s.n_mass, s.n_com, s.n_min, s.n_max,
            s.counters
        )
    return timed_kernel(fn)


def bench_normalize_com(s: BenchState) -> float:
    bpg_n = bpg(s.n_nodes, 256)
    def fn():
        normalize_com[bpg_n, 256](s.n_mass, s.n_com, s.n_nodes)
    return timed_kernel(fn)


def bench_find_root(s: BenchState) -> float:
    bpg_n = bpg(s.n_nodes)
    def fn():
        find_root[bpg_n, TPB](s.parent, s.root_idx)
    return timed_kernel(fn)


def bench_build_cache(s: BenchState) -> float:
    def fn():
        s.node_to_cache.fill(-1)
        build_top_tree_cache[1, 1](
            s.child, s.root_idx,
            s.top_nodes, s.node_to_cache, CACHE_SIZE
        )
    return timed_kernel(fn)


def bench_compute_forces(s: BenchState, theta: float) -> float:
    BATCH = min(5_000, s.n)
    def fn():
        s.force.fill(0)
        for offset in range(0, s.n, BATCH):
            c = min(BATCH, s.n - offset)
            compute_forces[bpg(c), TPB](
                s.pos_x, s.pos_y, s.pos_z,
                s.child, s.n_mass, s.n_com, s.n_min, s.n_max,
                s.force,
                theta, G, SOFTENING, BOX_SIZE,
                s.root_idx, s.top_nodes, s.node_to_cache,
                offset
            )
    return timed_kernel(fn)


def bench_integrate(s: BenchState) -> float:
    def fn():
        integrate[bpg(s.n), TPB](
            s.pos_x, s.pos_y, s.pos_z,
            s.vel_x, s.vel_y, s.vel_z,
            s.force, 0.5, 67.8, 1e-4
        )
    return timed_kernel(fn)


def bench_render(s: BenchState) -> float:
    # Build a minimal MVP matrix (no config needed — just a dummy 4×4)
    eye    = np.array([1.5, 0.2, 1.5], dtype=np.float32)
    target = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    z = (eye - target); z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = x;  view[0, 3] = -np.dot(x, eye)
    view[1, :3] = y;  view[1, 3] = -np.dot(y, eye)
    view[2, :3] = z;  view[2, 3] = -np.dot(z, eye)
    f = 1.0 / np.tan(np.radians(60) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f; proj[1, 1] = f
    proj[2, 2] = -(100 + 0.1) / (100 - 0.1)
    proj[2, 3] = -2 * 100 * 0.1 / (100 - 0.1)
    proj[3, 2] = -1.0
    mvp = cp.asarray(proj @ view)

    def fn():
        s.grid.fill(0)
        render_density[bpg(s.n), TPB](
            s.pos_x, s.pos_y, s.pos_z, s.grid, mvp, RES
        )
    return timed_kernel(fn)


# ---------------------------------------------------------------------------
# Kernel registry (name → bench function) — forces excluded (handled separately)
# ---------------------------------------------------------------------------

KERNELS = {
    "compute_codes":       bench_compute_codes,
    "reset_tree":          bench_reset_tree,
    "build_tree":          bench_build_tree,
    "compute_multipoles":  bench_compute_multipoles,
    "normalize_com":       bench_normalize_com,
    "find_root":           bench_find_root,
    "build_cache":         bench_build_cache,
    "integrate":           bench_integrate,
    "render_density":      bench_render,
}

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

print("=" * 60)
print("  N-Body Simulation — Kernel Benchmarking")
print(f"  N values : {[f'{n:,}' for n in N_VALUES]}")
print(f"  Warmup / Iters : {WARMUP} / {ITERATIONS}")
print("=" * 60)

# results[n][kernel_name] = ms
results: dict[int, dict[str, float]] = {}

# forces_results[theta][n] = ms
forces_results: dict[float, dict[int, float]] = {t: {} for t in THETA_LIST}

for n in N_VALUES:
    print(f"\n▶  N = {n:,}")
    s = BenchState(n)
    results[n] = {}

    # --- standard kernels ---
    for name, fn in KERNELS.items():
        ms = fn(s)
        results[n][name] = ms
        print(f"   {name:<24s} {ms:8.3f} ms")

    # --- compute_forces for all thetas ---
    for theta in THETA_LIST:
        ms = bench_compute_forces(s, theta)
        forces_results[theta][n] = ms
        print(f"   compute_forces (θ={theta})   {ms:8.3f} ms")

    # Store forces at pie-theta in main results
    results[n]["compute_forces"] = forces_results[THETA_PIE][n]

    del s   # free VRAM between runs
    cp.get_default_memory_pool().free_all_blocks()

print("\n✔  Benchmarks complete.")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

all_kernel_names = list(KERNELS.keys()) + ["compute_forces"]
header = ["N"] + all_kernel_names + [f"compute_forces_theta_{t}" for t in THETA_LIST]

with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for n in N_VALUES:
        row = [n] + [results[n][k] for k in all_kernel_names]
        row += [forces_results[t][n] for t in THETA_LIST]
        w.writerow(row)

print(f"✔  Results saved to {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

plt.style.use("bmh")
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.titlecolor"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
})

N_arr = np.array(N_VALUES, dtype=float)

PALETTE = [
    "#1B3A57",
    "#5C677D",
    "#9E2A2B",
    "#6A994E",
    "#386641",
    "#A7C957",
    "#344E41",
    "#7F5539",
    "#B08968",
    "#6D597A"
]

def nlogn_reference(n_arr, t_arr):
    """Return a scaled N log N curve aligned to the median data point."""
    mid = len(n_arr) // 2
    ref = n_arr * np.log2(n_arr)
    scale = t_arr[mid] / ref[mid]
    return ref * scale


# ---------------------------------------------------------------------------
# Plot 1 — compute_forces: Time vs N for 5 theta values
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("compute_forces — Time vs N  (Barnes-Hut, 5 opening angles)",
             fontsize=13, fontweight="bold", y=1.01)

for idx, theta in enumerate(THETA_LIST):
    t_arr = np.array([forces_results[theta][n] for n in N_VALUES])
    ax.plot(N_arr, t_arr, "o-", color=PALETTE[idx],
            linewidth=2, markersize=5, label=f"θ = {theta}")

# O(N log N) reference using median theta
t_med = np.array([forces_results[THETA_LIST[2]][n] for n in N_VALUES])
ref   = nlogn_reference(N_arr, t_med)
ax.plot(N_arr, ref, "--", color="#00000093", linewidth=1.5,
        alpha=0.7, label="O(N log N) ref")

ax.set_xscale("log")
#ax.set_yscale("log")
ax.set_xlabel("N  (number of bodies)", fontsize=11)
ax.set_ylabel("Time  (ms)", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
ax.legend(fontsize=10)
fig.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "forces_vs_N.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
print(f"✔  Saved {path1}")
plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2a — O(N) kernels (everything except compute_multipoles & compute_forces)
# ---------------------------------------------------------------------------

on_kernels = [k for k in all_kernel_names if k not in ("compute_multipoles", "compute_forces")]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle("Kernel Scaling — O(N) kernels  (log-log)",
             fontsize=14, fontweight="bold")
axes_flat = axes.flatten()


for idx, name in enumerate(on_kernels):
    ax    = axes_flat[idx]
    t_arr = np.array([results[n][name] for n in N_VALUES])

    ax.plot(N_arr, t_arr, "o-", color=PALETTE[idx % len(PALETTE)],
            linewidth=2, markersize=4, label="measured")

    ax.set_title(name, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("N", fontsize=8); ax.set_ylabel("ms", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v))))
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=7)

# hide unused subplots if on_kernels < 8
for idx in range(len(on_kernels), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.tight_layout()
path2a = os.path.join(OUTPUT_DIR, "scaling_on_kernels.png")
fig.savefig(path2a, dpi=150, bbox_inches="tight")
print(f"✔  Saved {path2a}")
plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2b — O(N log N) kernels: compute_multipoles & compute_forces (all thetas)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Kernel Scaling — O(N log N) kernels  (log-log)",
             fontsize=14, fontweight="bold")

# --- compute_multipoles ---
ax    = axes[0]
t_arr = np.array([results[n]["compute_multipoles"] for n in N_VALUES])
ref   = nlogn_reference(N_arr, t_arr)
ax.plot(N_arr, t_arr, "o-", color="#d2a8ff", linewidth=2, markersize=4, label="measured")
ax.plot(N_arr, ref,   "--", color="#8b949e", linewidth=1.2, alpha=0.7, label="O(N log N) ref")
ax.set_title("compute_multipoles", fontsize=11)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("N", fontsize=9); ax.set_ylabel("ms", fontsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v))))
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
ax.legend(fontsize=8)

# --- compute_forces (all 5 thetas) ---
ax = axes[1]
for idx, theta in enumerate(THETA_LIST):
    t_arr = np.array([forces_results[theta][n] for n in N_VALUES])
    ax.plot(N_arr, t_arr, "o-", color=PALETTE[idx],
            linewidth=2, markersize=4, label=f"θ = {theta}")

t_ref = np.array([forces_results[THETA_LIST[2]][n] for n in N_VALUES])
ref   = nlogn_reference(N_arr, t_ref)
ax.plot(N_arr, ref, "--", color="#8b949e", linewidth=1.2, alpha=0.7, label="O(N log N) ref")
ax.set_title("compute_forces", fontsize=11)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("N", fontsize=9); ax.set_ylabel("ms", fontsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v))))
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
ax.legend(fontsize=8)

fig.tight_layout()
path2b = os.path.join(OUTPUT_DIR, "scaling_nlogn_kernels.png")
fig.savefig(path2b, dpi=150, bbox_inches="tight")
print(f"✔  Saved {path2b}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3 — Camembert (pie chart) of kernel runtime shares at median N
# ---------------------------------------------------------------------------

median_n = N_VALUES[len(N_VALUES) // 2]
shares   = {k: results[median_n][k] for k in all_kernel_names}
total    = sum(shares.values())

labels = list(shares.keys())
sizes  = [shares[k] / total * 100 for k in labels]
colors = PALETTE[:len(labels)]

# Sort descending for a cleaner pie
pairs  = sorted(zip(sizes, labels, colors), reverse=True)
sizes, labels, colors = zip(*pairs)

fig, ax = plt.subplots(figsize=(9, 9))

wedge_props = dict(linewidth=1.2, edgecolor="#0d1117")
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
    startangle=140,
    wedgeprops=wedge_props,
    textprops={"color": "#0d1117", "fontsize": 9},
    pctdistance=0.78,
)
for at in autotexts:
    at.set_color("#0d1117")
    at.set_fontsize(8)
    at.set_fontweight("bold")

ax.set_title(
    f"Kernel Runtime Distribution\n(N = {median_n:,},  θ = {THETA_PIE},  total ≈ {total:.1f} ms)",
    fontsize=12, fontweight="bold", pad=20
)

path3 = os.path.join(OUTPUT_DIR, "camembert.png")
fig.savefig(path3, dpi=150, bbox_inches="tight")
print(f"✔  Saved {path3}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print(f"\n{'─'*60}")
print(f"  Runtime summary at N = {median_n:,}")
print(f"{'─'*60}")
print(f"  {'Kernel':<26s} {'Time (ms)':>10s}  {'Share':>7s}")
print(f"{'─'*60}")
for s_pct, lbl, _ in zip(sizes, labels, colors):
    ms = shares[lbl]
    print(f"  {lbl:<26s} {ms:10.3f}  {s_pct:6.1f}%")
print(f"{'─'*60}")
print(f"  {'TOTAL':<26s} {total:10.3f}  100.0%")
print(f"{'─'*60}")
print(f"\nAll plots saved in ./{OUTPUT_DIR}/")