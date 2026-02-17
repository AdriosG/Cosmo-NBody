import cupy as cp
from numba import cuda
from src.kernels.tree_kernels import build_tree, compute_multipoles, normalize_com, find_root, reset_tree_buffers

"""
Octree Manager
----------------- 

This module contains the Octree class which manages the construction and traversal of the binary radix data structure on the GPU.
The binary radix tree is an unfolded version of the octree.
"""

class OctreeManager:
    def __init__(self, n_bodies, tpb=256):
        """
        Initialize GPU memory buffers for the tree structure.
        
        Param:

        :n_bodies: int\\
            Total number of bodies in the simulation.
        :tpb: int\\
            Threads Per Block for CUDA kernels.
        """

        self.n = n_bodies
        self.n_nodes = 2 * self.n - 1
        self.tpb = tpb
        self.bpg = (self.n + tpb - 1) // tpb
        
        # Tree Structure Buffers (Intra-node topology) allocation
        self.child = cp.full((self.n_nodes, 2), -1, dtype=cp.int32)
        self.parent = cp.full(self.n_nodes, -1, dtype=cp.int32)
        self.root_idx = cp.zeros(1, dtype=cp.int32)
        
        # Multipole Buffers (Physical properties) allocation
        self.mass = cp.zeros(self.n_nodes, dtype=cp.float32)
        self.com = cp.zeros((self.n_nodes, 3), dtype=cp.float32)
        self.aabb_min = cp.zeros((self.n_nodes, 3), dtype=cp.float32)
        self.aabb_max = cp.zeros((self.n_nodes, 3), dtype=cp.float32)
        
        #Atomic counter for upward pass synchronization allocation
        self.counters = cp.zeros(self.n_nodes, dtype=cp.int32)

    def build(self, pos_d, mass_d, codes_d):

        """
        Builds the octree on the GPU using the provided bodies data and Morton codes.

        Param:

        :pos_d: tuple of DeviceArrays\\
            (x, y, z) sorted position arrays.
        :mass_d: DeviceArray\\
            Sorted mass array.
        :codes_d: DeviceArray\\
            Sorted Morton codes.
        """

        bpg_nodes = (self.n_nodes + self.tpb - 1) // self.tpb
        reset_tree_buffers[bpg_nodes, self.tpb](
            self.child, self.parent, self.counters, self.root_idx
        )

        build_tree[self.bpg, self.tpb](codes_d, self.child, self.parent)

        px, py, pz = pos_d
        compute_multipoles[self.bpg, self.tpb](
            px, py, pz, mass_d, 
            self.child, self.parent, 
            self.mass, self.com, 
            self.aabb_min, self.aabb_max, 
            self.counters
        )

        normalize_com[(self.n_nodes + 255) // 256, 256](
            self.mass, self.com, self.n_nodes
        )

        
        find_root[bpg_nodes, self.tpb](self.parent, self.root_idx)

        return self.root_idx
    
    def get_buffers(self):
        """Returns the internal buffers for physics kernels."""
        return {
            "child": self.child,
            "mass": self.mass,
            "com": self.com,
            "min": self.aabb_min,
            "max": self.aabb_max,
            "root": self.root_idx
        }