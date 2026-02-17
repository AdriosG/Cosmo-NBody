import numpy as np
from numpy.typing import NDArray
import src.utils.config as Config
from src.kernels.render_kernels import render_density
import cupy as cp



def normalize(v : NDArray[np.float32]) -> NDArray[np.float32]:

    """
    Normalizes the given vector.
    
    Param:
    
    :v: Array
    """

    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm
              
def look_at(eye : NDArray[np.float32], target : NDArray[np.float32], up : NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Creates a View Matrix for projection.

    Param:

    :eye: Array\\
        Position of the camera
    :target: Array\\
        Position of the target
    :up: Array\\
        Upward direction

    """
    z = normalize(eye - target)
    x = normalize(np.cross(up, z))
    y = normalize(np.cross(z, x))

    view = np.identity(4, dtype=np.float32)
    view[0, :3] = x
    view[1, :3] = y
    view[2, :3] = z
    view[0, 3] = -np.dot(x, eye)
    view[1, 3] = -np.dot(y, eye)
    view[2, 3] = -np.dot(z, eye)
    
    return view

def perspective(fov_deg : float, aspect : float, near : float, far : float) -> NDArray[np.float32] :
    """
    Creates a Perspective Projection Matrix.
    It maps the view-space coordinates into clip-space

    Param:

    :fov_deg: float\\
        Field of view in deg
    :aspect: float\\
        Aspect ratio of the viewport (Width / Height ratio)
    :near: float\\
        Near clippig plane distance
    :far: float\\
        Far plane clipping distance


    """
    fov_rad = np.radians(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)
    
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    
    return proj


def get_mvp_matrix(step : float, box_center : tuple, box_size : float, config: Config) -> NDArray[np.float32]:
    """
    Calculates the full MVP matrix for the current time step.
    implementing an orbital logic (Camera turns around the cube)

    Param:

    :step: float\\
        step of the simulation
    :box_center: Array\\
        coordinates of the center of the box
    :box_size: \\
        Size of the simulation box (code unit)
    :config: Configuration object containing simulation parameters.
    """

    angle = step * config.ROTATION_SPEED
    radius = box_size * config.CAM_DIST_MULT
    
    eye_x = box_center[0] + radius * np.cos(angle)
    eye_y = box_center[1] + (box_size * 0.2) 
    eye_z = box_center[2] + radius * np.sin(angle)
    
    eye = np.array([eye_x, eye_y, eye_z], dtype=np.float32)
    target = np.array(box_center, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    
    view = look_at(eye, target, up)
    proj = perspective(config.FOV, 1.0, 0.1, radius * 3.0)
    
    mvp = cp.asarray(proj @ view)
    return mvp


def colorize_frame(grid_d):

    """
    Responsible for the creation of a frame (Image format),
    Compute rendering functions and kernels and colorize the frame.

    Param:

    :grid_d: DeviceArray\\
        rendering grid for each pixels
    :setp: float\\
        Simulation step
    :px_d, py_d, pz_d: DeviceArrays\\
        Position of the bodies in the simulation at current step
    :config: Configuration object containing simulation parameters.
    """


    img = cp.log1p(grid_d * 120.0) 
    img /= cp.max(img) + 1e-9

    val = cp.power(img, 2.2)

    T1 = 0.6        # Blue ends / Green starts
    T2 = 0.72       # Green ends / Yellow starts
    T3 = 0.82       # Yellow ends / Red starts

    r = cp.zeros_like(val)
    g = cp.zeros_like(val)
    b = cp.zeros_like(val)

    mask1 = (val <= T1)
    t = val[mask1] / T1
    b[mask1] = t * 0.8  


    mask2 = (val > T1) & (val <= T2)
    t = (val[mask2] - T1) / (T2 - T1)
    b[mask2] = (1.0 - t) * 0.8  
    g[mask2] = t * 0.9          

    mask3 = (val > T2) & (val <= T3)
    t = (val[mask3] - T2) / (T3 - T2)
    g[mask3] = 0.9 + (t * 0.1)  
    r[mask3] = t

    mask4 = (val > T3)
    t = (val[mask4] - T3) / (1.0 - T3)
    r[mask4] = 1.0
    g[mask4] = 1.0 - t          

    frame = cp.stack([r, g, b], axis=-1)
    
    frame = frame * 1.3
    frame = cp.clip(frame, 0, 1)

    return frame
