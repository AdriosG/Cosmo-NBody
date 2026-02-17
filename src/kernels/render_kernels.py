import numba
from numba import cuda
import cupy as cp

"""
Render Kernels
--------------

Here the render_density is to be pared with the render functions found in /utils/render_utils.py
"""

@cuda.jit
def render_density(pos_x, pos_y, pos_z, grid, mvp, res):

    """
    CUDA kernel to render the density of particles onto a 2D grid using a given model-view-projection (MVP) matrix.
    Each particle's position is transformed by the MVP matrix to determine its screen coordinates, 
    and its contribution to the density grid is calculated based on its distance from the camera.

    Param:
    
    :pos_x, pos_y, pos_z: DeviceArray\\
        Arrays containing the x, y, z positions of the particles.
    :grid: DeviceArray\\
        2D array representing the density grid to be rendered.
    :mvp: DeviceArray\\
        4x4 array representing the model-view-projection matrix used for transforming particle positions to screen coordinates.
    :res: int\\
        Resolution of the output grid
    """

    i = cuda.grid(1)
    if i < pos_x.shape[0]:
        x, y, z = pos_x[i], pos_y[i], pos_z[i]
        
        xc = mvp[0,0]*x + mvp[0,1]*y + mvp[0,2]*z + mvp[0,3]
        yc = mvp[1,0]*x + mvp[1,1]*y + mvp[1,2]*z + mvp[1,3]
        wc = mvp[3,0]*x + mvp[3,1]*y + mvp[3,2]*z + mvp[3,3]
        
        if wc > 0.01:
            inv = 1.0 / wc
            sx = (xc * inv + 1.0) * 0.5 * res
            sy = (1.0 - yc * inv) * 0.5 * res
            
            if 0 <= sx < res and 0 <= sy < res:
                val = 2.5 * inv 
                cuda.atomic.add(grid, (int(sy), int(sx)), val)

