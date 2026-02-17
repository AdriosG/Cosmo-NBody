import cupy as cp
from src.kernels.morton import compute_codes

class SimulationState:
    def __init__(self, n_bodies, pos_init, vel_init):

        self.n = n_bodies

        self.pos = pos_init

        self.vel = vel_init

        self.force = cp.zeros((self.n, 3), dtype=cp.float32)

        self.mass = cp.full(self.n, 1.0 / self.n, dtype=cp.float32)

        self.codes = cp.zeros(self.n, dtype=cp.uint64)
        self.indices = cp.zeros(self.n, dtype=cp.uint32)

    def sort_buffer_by_morton(self, config):

        """
        Computes codes and reoder all buffers to ensure spatial locality.
        (positions, velocities, masses, codes)
        """
        tpb, bpg = config.TPB, config.BPG

        compute_codes[bpg,  tpb](
            *self.pos,
            self.codes,
            self.indices,
            self.n
        )

        sort_idx = cp.argsort(self.codes)

        self.pos = (self.pos[0][sort_idx], self.pos[1][sort_idx], self.pos[2][sort_idx])
        self.vel = (self.vel[0][sort_idx], self.vel[1][sort_idx], self.vel[2][sort_idx])
        self.mass = self.mass[sort_idx]
        self.codes = self.codes[sort_idx]