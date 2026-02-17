import src.utils.config as Config
import cupy as cp
import numpy as np

"""
Cosmology & Initial Conditions (IC) Module
==========================================
This module implements the Zel'dovich Approximation to generate initial 
conditions for cosmological N-body simulations.
"""

def generate_zeldovich_ics(config: Config):

    """
    Generates initial positions and velocities for bodies using the Zel'dovich Approximation.
    
    Param:
    
    :config: Configuration object containing simulation parameters.
    """

    n_grid = config.N_GRID
    box_size = config.BOX_SIZE

    cp.random.seed(42)

    k = cp.fft.fftfreq(n_grid) * n_grid 
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')
    k_mag = cp.sqrt(kx**2 + ky**2 + kz**2)
    k_mag[0,0,0] = 1e-10
    
    P = k_mag ** -2.5
    P[0,0,0] = 0
    
    phase = cp.random.uniform(0, 2*np.pi, (n_grid, n_grid, n_grid))
    amp = cp.sqrt(P)
    delta_k = amp * (cp.cos(phase) + 1j*cp.sin(phase))
    
    def get_disp(k_comp):
        term = 1j * k_comp / (k_mag**2 + 1e-10)
        term[0,0,0] = 0
        return cp.fft.ifftn(term * delta_k).real
    
    dx = get_disp(kx).flatten()
    dy = get_disp(ky).flatten()
    dz = get_disp(kz).flatten()
    
    disp_std = cp.std(dx)
    target_std = (1.0 / n_grid) * 0.3
    scale = target_std / disp_std
    
    dx *= scale
    dy *= scale
    dz *= scale
    
    lin = cp.linspace(0, 1, n_grid, endpoint=False)
    gx, gy, gz = cp.meshgrid(lin, lin, lin, indexing='ij')
    gx, gy, gz = gx.flatten(), gy.flatten(), gz.flatten()
    
    pos_x = (gx + dx) % 1.0
    pos_y = (gy + dy) % 1.0
    pos_z = (gz + dz) % 1.0
    
    H_start = config.H0 * np.sqrt(config.OMEGA_M * config.A_START**-3 + config.OMEGA_L)
    Om_a = config.OMEGA_M * (config.A_START**-3) / (config.OMEGA_M * config.A_START**-3 + config.OMEGA_L)
    f = Om_a ** 0.55
    vel_coupling = H_start * f * config.A_START / 20

    vel_x = dx * vel_coupling 
    vel_y = dy * vel_coupling 
    vel_z = dz * vel_coupling 
    
    return [pos_x, pos_y, pos_z], [vel_x, vel_y, vel_z]


def get_hubble_param(a, config: Config):
    """
    Computes the Hubble parameter at a given scale factor.
    
    Param:

    :a: Scale factor at which to compute the Hubble parameter.
    :config: Configuration object containing cosmological parameters.
    """
    return config.H0 * np.sqrt(config.OMEGA_M * a**-3 + config.OMEGA_L)