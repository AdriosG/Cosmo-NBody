"""
Energy_Conservation.py
======================
Energy conservation benchmark for the N-body simulation.

This script runs a simulation for a fixed number of steps and
plots the total energy of the system over time. The plot shows
the relative energy error (E(t) - E(0)) / E(0), which is a
measure of how well the simulation conserves energy.
"""

import os
import sys
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.state import SimulationState
from src.core.engine import SimulationEngine
import src.utils.config as Config

# Simulation parameters
N_PARTICLES = 10_000
N_STEPS = 500
THETA = 0.7
G = 1.0

potential_energy_kernel = cp.RawKernel(r'''
extern "C" __global__
void potential_energy_kernel(const float* x, const float* y, const float* z, const float* mass,
                             int n, float* potential_energy, float G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double pe = 0.0;
    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        float dx = x[j] - x[i];
        float dy = y[j] - y[i];
        float dz = z[j] - z[i];
        float dist_sq = dx * dx + dy * dy + dz * dz;
        float dist = sqrt(dist_sq);
        pe -= G * mass[i] * mass[j] / dist;
    }
    atomicAdd(potential_energy, pe / 2.0); // Divide by 2 to correct for double counting
}
''', 'potential_energy_kernel')

def compute_potential_energy_gpu(state: SimulationState, G: float = 1.0) -> float:
    """Computes the total potential energy of the system using a GPU kernel."""
    potential_energy_gpu = cp.zeros(1, dtype=cp.float64)
    tpb = 256
    bpg = (state.n + tpb - 1) // tpb
    
    potential_energy_kernel(
        (bpg,), (tpb,),
        (state.pos[0], state.pos[1], state.pos[2], state.mass,
         state.n, potential_energy_gpu, G)
    )
    return float(potential_energy_gpu.get())

def compute_kinetic_energy(state: SimulationState) -> float:
    """Computes the total kinetic energy of the system."""
    vel_sq = state.vel[0]**2 + state.vel[1]**2 + state.vel[2]**2
    return 0.5 * float(cp.sum(state.mass * vel_sq))

def run_benchmark():
    """
    Runs the energy conservation benchmark.
    """
    print("=" * 60)
    print("  N-Body Simulation — Energy Conservation Benchmark")
    print(f"  N_PARTICLES: {N_PARTICLES}")
    print(f"  N_STEPS: {N_STEPS}")
    print(f"  THETA: {THETA}")
    print("=" * 60)

    # Initial conditions
    pos_init_np = np.random.rand(3, N_PARTICLES).astype(np.float32)
    pos_init = (cp.asarray(pos_init_np[0]), cp.asarray(pos_init_np[1]), cp.asarray(pos_init_np[2]))
    vel_init = (cp.zeros(N_PARTICLES, dtype=cp.float32), 
                cp.zeros(N_PARTICLES, dtype=cp.float32), 
                cp.zeros(N_PARTICLES, dtype=cp.float32))

    # Initialize the simulation state and engine
    config = Config
    state = SimulationState(N_PARTICLES, pos_init, vel_init)
    engine = SimulationEngine(config)

    energies = []
    times = []

    # Initial energy
    kinetic_energy = compute_kinetic_energy(state)
    potential_energy = compute_potential_energy_gpu(state)
    total_energy = kinetic_energy + potential_energy
    energies.append(total_energy)
    times.append(0.0)

    initial_energy = total_energy
    print(f"Initial Energy: {initial_energy:.6e}")

    # Main simulation loop
    for i in range(N_STEPS):
        engine.step(state)
        
        kinetic_energy = compute_kinetic_energy(state)
        potential_energy = compute_potential_energy_gpu(state)
        total_energy = kinetic_energy + potential_energy
        
        energies.append(total_energy)
        # The engine does not seem to track time, so I will just use step number
        times.append(i+1)

        if (i + 1) % 10 == 0:
            print(f"Step {i+1}/{N_STEPS} | "
                  f"Energy: {total_energy:.6e} | "
                  f"Error: {(total_energy - initial_energy) / abs(initial_energy):.6e}")

    # Plotting
    plt.style.use("bmh")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rel_error = (np.array(energies) - initial_energy) / abs(initial_energy)
    
    ax.plot(times, rel_error, lw=2)
    
    ax.set_title("Energy Conservation", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Relative Energy Error (E(t) - E(0)) / |E(0)|", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Scientific notation for y-axis
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    output_path = os.path.join(os.path.dirname(__file__), "energy_conservation.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    run_benchmark()
