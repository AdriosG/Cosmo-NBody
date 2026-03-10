from multiprocessing import util

import cupy as cp
import numpy as np
from tqdm import tqdm
from src.core.octree import OctreeManager
from src.kernels.physics_kernels import compute_forces, integrate
from src.kernels.tree_kernels import build_top_tree_cache
from src.utils.cosmology import get_hubble_param
from src.core.camera import Camera
import time
import threading
import pynvml as nvml
import cProfile
import pstats
"""

Simulation Engine module
-------------------------

This module assembles and manages the physical evolution of the smulation.
It manages the time stepping, the cosmological scaling and the GPU workload distribution.
"""



def profile(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(15)
        return result
    return wrapper

class SimulationEngine:

    def __init__(self, config, state):

        """
        Main simulation engine.
        Contains the step logic and simulation loop.

        - step() to compute a single step (does not generate any frames
        - run() to loop over steps (and load frames into buffer)
        - save() to send frames back to CPU and make video
        """

        self.config = config
        self.state = state
        self.tree_manager = OctreeManager(config.N_BODIES, config.TPB)
        self.camera = Camera(config.N_STEPS, config.RES, config)
        
        self.cache_size = config.CACHE_SIZE
        self.tpb = config.TPB
        self.a = config.A_START
        self.da = (config.A_END - config.A_START) / config.N_STEPS

        self.top_nodes = cp.full(self.cache_size, -1, dtype=cp.int32)
        self.node_to_cache = cp.full(self.tree_manager.n_nodes, -1, dtype=cp.int32)

        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(0)
        
    @profile
    def profile_run(self):
        """
        Loops over every simulation steps and generates frames into VRAM, with profiling.
        """
        with tqdm(range(self.config.N_STEPS)) as pbar:
            for i in pbar:
                if i % 20 == 0:
                    gpu, mem = self.get_gpu_util()
                    pbar.set_postfix(GPU=f"{gpu}%", MEM=f"{mem}%")

                self.step()
                frame = self.camera.capture(i, *self.state.pos)
                self.camera.video_buffer[i] = (frame.get() * 255).astype(np.uint8)

    def run(self):
        """
        Loops over every simulation steps and generates frames into VRAM.
        """
        self.profile_run()

    def save(self):

        """
        Gathers frames from VRAM and save them into a video. 
        Writing frames into a folder named render_output.
        Writing video in the same folder as main.py script.
        """

        self.camera.get_frames()
        self.camera.make_video()
        nvml.nvmlShutdown()
    
    def get_gpu_util(self):
        """
        Get current GPU utilization and memory usage using NVML.
        """
        util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
        return util.gpu, util.memory
        