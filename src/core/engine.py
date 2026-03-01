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
"""

Simulation Engine module
-------------------------

This module assembles and manages the physical evolution of the smulation.
It manages the time stepping, the cosmological scaling and the GPU workload distribution.
"""



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
        

    def step(self):
        
        """
        Compute a single step of simulation (does not handle rendering)
        """

        cp.cuda.Device().synchronize()
        
        H = get_hubble_param(self.a, self.config)
        dt_physics = self.da / (self.a * H) * 0.8
        
        self.state.sort_buffer_by_morton(self.config)

        self.tree_manager.build(
            self.state.pos,
            self.state.mass,
            self.state.codes
        )
        


        
        self.state.force.fill(0)
        
        tree_buffers = self.tree_manager.get_buffers()
        
        self.node_to_cache.fill(-1)

        build_top_tree_cache[1, 1](
            tree_buffers["child"],
            tree_buffers["root"],
            self.top_nodes,
            self.node_to_cache,
            self.cache_size
        )

        for offset in range(0, self.config.N_BODIES, self.config.BATCH_SIZE):

            c = min(self.config.BATCH_SIZE, self.config.N_BODIES - offset)

            compute_forces[(c+self.tpb-1)//self.tpb, self.tpb](
                *self.state.pos,
                tree_buffers["child"],
                tree_buffers["mass"],
                tree_buffers["com"],
                tree_buffers["min"],
                tree_buffers["max"],
                self.state.force,
                self.config.THETA,
                self.config.G,
                self.config.SOFTENING,
                self.config.BOX_SIZE,
                tree_buffers["root"],
                self.top_nodes,
                self.node_to_cache,
                offset
            )


        integrate[self.config.BPG, self.tpb](
            *self.state.pos,
            *self.state.vel,
            self.state.force,
            self.a,
            H,
            dt_physics
        )

        self.a += self.da
    
    def run(self):

        """
        Loops over every simulation steps and generates frames into VRAM
        """
        with tqdm(range(self.config.N_STEPS)) as pbar:
            for i in pbar:

                if i % 20 == 0:
                    gpu, mem = self.get_gpu_util()
                    pbar.set_postfix(GPU=f"{gpu}%", MEM=f"{mem}%")

                self.step()
                frame = self.camera.capture(i, *self.state.pos)
                self.camera.video_buffer[i] = (frame.get() * 255).astype(np.uint8)

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
        