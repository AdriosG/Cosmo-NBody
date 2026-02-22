"""
Configuration Module
--------------------

This module contains all the configuration parameters of the simulation.
"""

# --- SIMULATION PARAMETERS ---
N_GRID = 128                # Number of grid points per dimension
N_BODIES = N_GRID**3        # Total number of bodies in the simulation
BOX_SIZE = 1.0              # Size of the simulation box
SOFTENING = 0.001           # Softening length to prevent singularities
N_STEPS = 2500              # Number of simulation steps
A_START, A_END = 0.02, 1.0  # Scale factor range for the simulation

# --- RENDER SETTINGS ---
ROTATION_SPEED = 0.001      # Rotation speed for the camera
CAM_DIST_MULT = 1.8         # Multiplier for camera distance from the center
FOV = 60.0                  # Field of view for rendering
RES = 1024                  # Resolution of the output images
OUTPUT_DIR = "render_output"

# --- CUDA CONFIG ---
TPB = 256                   # Threads per block for CUDA kernels
BPG = (N_BODIES + TPB - 1) // TPB # Blocks per grid for CUDA kernels
BATCH_SIZE = int(5_000)            # Batch size for processing bodies in CUDA kernels
CACHE_SIZE = 256                   # Cache size for force computation shared array

# --- PHYSICS CONSTANTS ---
THETA = 0.7
G = 1.0                     # Gravitational constant
OMEGA_M = 1.0               # Matter density parameter
OMEGA_L = 0.692             # Dark Energy density parameter
H0 = 67.8                   # Hubble constant
STACK_SIZE_CHILD = 128      # Stack size for tree transversal and force computation