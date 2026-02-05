import numpy as np
import cupy as cp
from cupy.fft import fftn, ifftn
import numba
from numba import cuda, uint64
from PIL import Image
from IPython.display import display, clear_output
import math
import os
import subprocess
from tqdm import tqdm
import concurrent.futures
import time

cuda.synchronize()
cp.cuda.Device().synchronize()

# --- SIMULATION PARAMETERS ---
N_GRID = 100                  # Grid size per dimension
N_PARTICLES = N_GRID**3      # Total particles (~262k for 64^3, 1M for 100^3)
BOX_SIZE = 100.0             # Length of the periodic box side
SOFTENING = 0.1              # Softening length to prevent singularities at r=0
G = 1.0                      # Gravitational constant (code units)
H0 = 0.07                     # Hubble parameter (expansion rate)
DT = 0.002                    # Time step
N_STEPS = 2500                   # Total simulation steps
A_START = 0.01               # Starting scale factor
A_END = 1.0                 # Ending scale factor
OMEGA_M = 1.0              # Matter density parameter

# --- BARNES-HUT PARAMETERS ---
THETA = 0.5                  # Opening angle for MAC (0.5 is standard trade-off)
MAX_DEPTH = 18               # Max tree depth (octree)
WARP_SIZE = 32

# --- VISUALIZATION PARAMETERS ---
RES = 1024                   # Render resolution (RES x RES)
CMAP_GAMMA = 0.5             # Gamma correction for log-scale brightness
OUTPUT_DIR = "render_output"
VIDEO_NAME = "nbody_simulation.mp4"

FOV = 60.0
CAM_DIST_MULT = 1.8
ROTATION_SPEED = 0.005 #Radians per frame
os.makedirs(OUTPUT_DIR, exist_ok=True) 

# --- CUDA CONFIGURATION ---
TPB = 256                    # Threads Per Block
BPG = (N_PARTICLES + TPB - 1) // TPB # Blocks Per Grid
BATCH_SIZE = 20000
BATCH_BPG = (BATCH_SIZE + TPB - 1) // TPB
MAX_H_RATE = 1000.0 

# --- TREE CONSTRUCTION ---
num_nodes = 2 * N_PARTICLES

# --- PHYSICS COMPUTATIONS ---
drag_factor = 1.0 / (1.0 + H0 * DT)

def generate_zeldovich_ics(n_grid, box_size, seed=42, spectral_index=2.0, amplitude=1.0):
    """
    Generates Initial Conditions using the Zel'dovich approximation.
    Method:
    1- Generate a uniform 3D grid of particles (Lagrangian coordinates q).
    2- Generate a Gaussian Random Field in Fourier space.
    3- Apply a power spectrum P(k) ~ k^-n.
    4- Compute displacement field via FFT.
    5- Displace particles and assign initial velocities (Hubble flow + peculiar velocity).
    """
    cp.random.seed(seed)

    lins = cp.linspace(0, box_size, n_grid, endpoint=False) # Lagrangian grid

    qx, qy, qz = cp.meshgrid(lins, lins, lins, indexing='ij') # coordinates of q (see above)
    
    # Fourier Space Setup
    k = cp.fft.fftfreq(n_grid, d=box_size/n_grid) * 2 * cp.pi
    kx, ky, kz = cp.meshgrid(k, k, k, indexing='ij')
    k_sq = kx**2 + ky**2 + kz**2 # coordinates in fourier space (squared)
    k_sq[0,0,0] = 1e-10 # Avoid division by zero at DC component

    # The random Gaussian field in Fourier space
    random_field = cp.random.normal(0, 1, (n_grid, n_grid, n_grid)) + \
                   1j * cp.random.normal(0, 1, (n_grid, n_grid, n_grid))
    

    # aplly power spectrum
    power_spectrum = 1.0 / (k_sq ** (spectral_index / 2.0)) # P(k) ~ k^-2
    power_spectrum[0,0,0] = 0
    delta_k = random_field * cp.sqrt(power_spectrum)

    #Compute displacement field
    def get_displacement(k_component_grid):
        term = 1j * k_component_grid / k_sq
        term[0,0,0] = 0
        disp_k = term * delta_k
        return cp.fft.ifftn(disp_k).real * n_grid**3 
    
    dx = get_displacement(kx).flatten()
    dy = get_displacement(ky).flatten()
    dz = get_displacement(kz).flatten()

    dx *= amplitude
    dy *= amplitude
    dz *= amplitude

    
    # x = q + D * dx with perodic wrap
    pos_x = cp.mod(qx.flatten() + dx, box_size).astype(cp.float32)
    pos_y = cp.mod(qy.flatten() + dy, box_size).astype(cp.float32)
    pos_z = cp.mod(qz.flatten() + dz, box_size).astype(cp.float32)

    H_start = H0 * (A_START ** (-1.5))
    vel_factor = H_start * A_START * 0.8 #0.8 is arbitrary growth rate

    #0.5 is arbitrary growth rate
    vel_x = (dx * vel_factor * 0.5).astype(cp.float32)
    vel_y = (dy * vel_factor * 0.5).astype(cp.float32)
    vel_z = (dz * vel_factor * 0.5).astype(cp.float32)

    mass = cp.ones(n_grid**3, dtype=cp.float32) # Uniform mass
    
    return (pos_x, pos_y, pos_z), (vel_x, vel_y, vel_z), mass

@cuda.jit(device=True)
def expand_bits(v):
    """
    Expands a 21-bit integer into 64 bits by inserting 2 zeros after each bit.
    """
    v &= 0x1fffff # avoid overflow
    v = (v | (v << 32)) & 0x1f00000000ffff
    v = (v | (v << 16)) & 0x1f0000ff0000ff
    v = (v | (v << 8))  & 0x100f00f00f00f00f
    v = (v | (v << 4))  & 0x10c30c30c30c30c3
    v = (v | (v << 2))  & 0x1249249249249249
    return v

@cuda.jit(device=True)
def get_morton_code(x, y, z, min_grid, grid_scale):
    """
    Computes the 3D Morton code for a body given its position.
    """

    ix = uint64((x - min_grid) * grid_scale)
    iy = uint64((y - min_grid) * grid_scale)
    iz = uint64((z - min_grid) * grid_scale)

    return expand_bits(ix) | (expand_bits(iy) << 1) | (expand_bits(iz) << 2)

@cuda.jit
def compute_morton_codes(pos_x, pos_y, pos_z, codes, n_bodies, grid_scale, indices):
    """
    CUDA kernel to compute Morton codes for all particles.
    """
    idx = cuda.grid(1)
    if idx < n_bodies:
        codes[idx] = get_morton_code(pos_x[idx], pos_y[idx], pos_z[idx], 0.0, grid_scale)
        indices[idx] = idx # keep track of original indices

def sort_bodies(pos, vel, mass):
    """
    Radix sort all arrays based on their Morton codes.
    """
    n = pos[0].shape[0]
    min_coord = 0.0
    grid_scale = ((2**21 - 2)/ BOX_SIZE) #Boundary issue solved by -2

    codes = cp.zeros(n, dtype=cp.uint64)
    indices = cp.zeros(n, dtype=cp.uint32)

    compute_morton_codes[BPG, TPB](pos[0], pos[1], pos[2], codes, n, grid_scale, indices)

    sort_idx = cp.argsort(codes) #Radix sort on GPU

    pos_sorted = (pos[0][sort_idx], pos[1][sort_idx], pos[2][sort_idx])
    vel_sorted = (vel[0][sort_idx], vel[1][sort_idx], vel[2][sort_idx])
    mass_sorted = mass[sort_idx]
    codes_sorted = codes[sort_idx]

    return pos_sorted, vel_sorted, mass_sorted, codes_sorted

@cuda.jit(device=True)
def delta_fn(codes, i, j, n):
    """
    Computes the length of the longest common prefix between codes[i] and codes[j].
    Handles boundary conditions.
    """
    if j < 0 or j >= n:
        return -1
    
    code_i = codes[i]
    code_j = codes[j]

    if code_i == code_j :
        code_i, code_j = i, j # if duplicate codes
        return 64 + (32 - cuda.clz(i ^ j)) #32 for int32 index collision
    
    return cuda.libdevice.clzll(numba.int64(code_i ^ code_j)) # CLZ shows shared prefix and XOR differing bits (czll because int64)

@cuda.jit
def build_radix_tree_kernel(codes, children, parents):
    """
    Constructs the internal nodes of the binary radix tree.
    Each thread i (0 to N-2) constructs internal node (i + N).
    """

    i = cuda.grid(1)
    n = codes.shape[0]

    if i >= n-1:
        return
    
    idx = i + n #internal node index

    d_prev = delta_fn(codes, i, i-1, n)
    d_next = delta_fn(codes, i, i+1, n)

    direction = 1
    min_delta = d_prev
    if d_next > d_prev:
        direction = 1
        min_delta = d_prev # Lower bound for split
    else:
        direction = -1
        min_delta = d_next
        
    
    l_max = 2
    while delta_fn(codes, i, i + l_max * direction, n) > min_delta: # Determine upper bound of the range (l_max)
        l_max *= 2 
        
    
    l = 0
    t = l_max // 2
    while t >= 1:
        if delta_fn(codes, i, i + (l + t) * direction, n) > min_delta: #Find the other end using binary search
            l += t
        t //= 2
        
    j = i + l * direction
    
    
    delta_node = delta_fn(codes, i, j, n) 
    s = 0
    t = l
    while t > 0: # Find the split position (gamma)
        
        step = (t + 1) // 2 

        check_idx = i + (s + step) * direction
        if delta_fn(codes, i, check_idx, n) > delta_node: # Standard binary search to find split
            s += step
        t -= step
        
    gamma = i + s * direction
    if direction == -1:
        gamma -= 1 # Adjustment for reverse search

    # Left child
    left = gamma
    right = gamma + 1
    
    range_left = min(i, j)
    range_right = max(i, j)
    

    if range_left == gamma: #assign Left
        children[idx, 0] = gamma #leaf
    else:
        children[idx, 0] = gamma + n #internal Node
        

    if range_right == gamma + 1: #assign Right
        children[idx, 1] = gamma + 1 # Leaf
    else:
        children[idx, 1] = gamma + 1 + n # Internal Node

    
    parents[children[idx, 0]] = idx # Set Parents (needed for upward pass)
    parents[children[idx, 1]] = idx

@cuda.jit
def compute_multipoles_kernel(pos, mass, children, parents, node_mass, node_com, 
                              node_min, node_max, counters):
    """
    Computes Mass, Center of Mass, and Bounding Box (AABB) for each node.
    """
    idx = cuda.grid(1)
    n_leaf = pos[0].shape[0]

    pos_x, pos_y, pos_z = pos

    if idx >= n_leaf:
        return

    #Initialize Leaf Nodes
    m = mass[idx]
    px = pos_x[idx]
    py = pos_y[idx]
    pz = pos_z[idx]

    node_mass[idx] = m
    node_com[idx, 0] = px
    node_com[idx, 1] = py
    node_com[idx, 2] = pz
    
    # Initialize AABB (Leaf min = max = pos)
    node_min[idx, 0] = px
    node_min[idx, 1] = py
    node_min[idx, 2] = pz
    node_max[idx, 0] = px
    node_max[idx, 1] = py
    node_max[idx, 2] = pz
    
    curr = idx

    while True:
        parent = parents[curr]
        if parent == -1:
            break 
        
        cuda.threadfence()
        old_val = cuda.atomic.add(counters, parent, 1)

        if old_val == 0: 
            break # Wait for sibling

        elif old_val == 1: # Second child creates the parent
            left = children[parent, 0]
            right = children[parent, 1]
            
            #mass and COM 
            m_l = node_mass[left]
            m_r = node_mass[right]
            sum_m = m_l + m_r
            
            if sum_m > 0:
                inv_m = 1.0 / sum_m
                cx = (node_com[left, 0] * m_l + node_com[right, 0] * m_r) * inv_m
                cy = (node_com[left, 1] * m_l + node_com[right, 1] * m_r) * inv_m
                cz = (node_com[left, 2] * m_l + node_com[right, 2] * m_r) * inv_m
            else:
                cx, cy, cz = 0.0, 0.0, 0.0

            node_mass[parent] = sum_m
            node_com[parent, 0] = cx
            node_com[parent, 1] = cy
            node_com[parent, 2] = cz
            
            # AABB merge
            
            # Min X
            lx, rx = node_min[left, 0], node_min[right, 0]
            node_min[parent, 0] = lx if lx < rx else rx
            # Min Y
            ly, ry = node_min[left, 1], node_min[right, 1]
            node_min[parent, 1] = ly if ly < ry else ry
            # Min Z
            lz, rz = node_min[left, 2], node_min[right, 2]
            node_min[parent, 2] = lz if lz < rz else rz
            
            # Max X
            lx, rx = node_max[left, 0], node_max[right, 0]
            node_max[parent, 0] = lx if lx > rx else rx
            # Max Y
            ly, ry = node_max[left, 1], node_max[right, 1]
            node_max[parent, 1] = ly if ly > ry else ry
            # Max Z
            lz, rz = node_max[left, 2], node_max[right, 2]
            node_max[parent, 2] = lz if lz > rz else rz

            curr = parent

@cuda.jit
def find_root_kernel(parents, root_idx):
    i = cuda.grid(1)
    if i < parents.shape[0]:
        if parents[i] == -1:
            root_idx[0] = i

@cuda.jit(fastmath=True)
def compute_forces_kernel(pos, mass, children, node_mass, node_com, 
                          node_min, node_max, force,
                          theta, G, softening, box_size, root_idx, i_offset):
    tid = cuda.grid(1)

    i = tid + i_offset

    if i >= pos[0].shape[0]:
        return

    p_pos_x = pos[0][i]
    p_pos_y = pos[1][i]
    p_pos_z = pos[2][i]

    acc_x = 0.0
    acc_y = 0.0
    acc_z = 0.0

    # Stack initialization
    stack = cuda.local.array(96 , dtype=numba.int32) 
    stack_top = 0
    stack[stack_top] = root_idx[0]
    stack_top += 1
    
    theta_sq = theta * theta

    while stack_top > 0:
        stack_top -= 1
        node_idx = stack[stack_top]
        
        # Load Mass
        n_mass = node_mass[node_idx]
        
        #Skip empty nodes or extremely light nodes
        if n_mass <= 0:
            continue
            
        # Distance Calculation (Nearest Image)
        dx = node_com[node_idx, 0] - p_pos_x
        dy = node_com[node_idx, 1] - p_pos_y
        dz = node_com[node_idx, 2] - p_pos_z
        
        dx -= box_size * round(dx / box_size)
        dy -= box_size * round(dy / box_size)
        dz -= box_size * round(dz / box_size)
        
        dist_sq = dx*dx + dy*dy + dz*dz + 1e-10
        
        is_leaf = (children[node_idx, 0] == -1)
        apply_force = is_leaf
        
        if not is_leaf:
            # AABB Size Calculation
            sx = node_max[node_idx, 0] - node_min[node_idx, 0]
            sy = node_max[node_idx, 1] - node_min[node_idx, 1]
            sz = node_max[node_idx, 2] - node_min[node_idx, 2]
            
            # Clamp size for periodic boundary spanning nodes
            # If a node spans > 50% of the box, it wraps.
            if sx > box_size * 0.5: sx = box_size
            if sy > box_size * 0.5: sy = box_size
            if sz > box_size * 0.5: sz = box_size
            
            size_sq = sx*sx + sy*sy + sz*sz # Diagonal squared usually safer
            
            # MAC Criterion
            if size_sq < (theta_sq * dist_sq):
                apply_force = True
        
        # 2. Avoid Self-Interaction
        if node_idx == i:
            apply_force = False

        if apply_force:
            dist_soft = dist_sq + softening*softening
            inv_dist = 1.0 / math.sqrt(dist_soft)
            inv_dist_cube = inv_dist * inv_dist * inv_dist
            f = G * n_mass * inv_dist_cube
            
            acc_x += f * dx
            acc_y += f * dy
            acc_z += f * dz
            
        elif not is_leaf: 
            # Stack Safety: Only push if there is room
            if stack_top + 2 < 96:
                stack[stack_top] = children[node_idx, 0]
                stack_top += 1
                stack[stack_top] = children[node_idx, 1]
                stack_top += 1

    force[i, 0] = acc_x
    force[i, 1] = acc_y
    force[i, 2] = acc_z

@cuda.jit
def integrate_kernel(pos, vel, force, dt, box_size, drag_factor):
    """
    Updates positions and velocities.
    Includes 'drag_factor' to simulate Hubble expansion drag (a(t)).
    v_{i+1} = v_i + a * dt
    x_{i+1} = x_i + v_{i+1} * dt
    """
    i = cuda.grid(1)
    if i >= pos[0].shape[0]:
        return

    pos_x, pos_y, pos_z = pos
    vel_x, vel_y, vel_z = vel

    vx = vel_x[i] + force[i, 0] * dt
    vy = vel_y[i] + force[i, 1] * dt
    vz = vel_z[i] + force[i, 2] * dt

    vx *= drag_factor
    vy *= drag_factor
    vz *= drag_factor
    
    max_v = box_size * 0.1 / dt
    v_sq = vx*vx + vy*vy + vz*vz
    if v_sq > max_v*max_v:
        scale = max_v / math.sqrt(v_sq)
        vx *= scale
        vy *= scale
        vz *= scale

    # Update Position
    px = pos_x[i] + vx * dt
    py = pos_y[i] + vy * dt
    pz = pos_z[i] + vz * dt

    # Periodic Wrap
    px = px % box_size
    py = py % box_size
    pz = pz % box_size
    
    if px < 0: px += box_size
    if py < 0: py += box_size
    if pz < 0: pz += box_size

    # Store
    vel_x[i] = vx
    vel_y[i] = vy
    vel_z[i] = vz
    pos_x[i] = px
    pos_y[i] = py
    pos_z[i] = pz

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def look_at(eye, target, up):
    """
    Creates a View Matrix.
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

def perspective(fov_deg, aspect, near, far):
    """
    Creates a Perspective Projection Matrix.
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

def get_mvp_matrix(step, box_center, box_size):
    """
    Calculates the full MVP matrix for the current time step.
    Orbit logic: Circular path on XZ plane, looking at center.
    """
    angle = step * ROTATION_SPEED
    radius = box_size * CAM_DIST_MULT
    
    # Orbiting position
    eye_x = box_center[0] + radius * np.cos(angle)
    eye_y = box_center[1] + (box_size * 0.2) # Slight elevation
    eye_z = box_center[2] + radius * np.sin(angle)
    
    eye = np.array([eye_x, eye_y, eye_z], dtype=np.float32)
    target = np.array(box_center, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    
    view = look_at(eye, target, up)
    proj = perspective(FOV, 1.0, 0.1, radius * 3.0)
    
    # Model matrix is Identity (simulation coordinates are world coordinates)
    mvp = np.dot(proj, view)
    return mvp

@cuda.jit
def render_3d_density_kernel(pos, grid, mvp, width, height):
    """
    Projects 3D world coordinates to 2D screen coordinates using MVP matrix.
    Performs perspective division and atomic accumulation.
    """
    i = cuda.grid(1)
    if i < pos[0].shape[0]:
        # Load Particle Position (x, y, z, 1.0)
        px = pos[0][i]
        py = pos[1][i]
        pz = pos[2][i]
        pw = 1.0
        
        # Matrix Multiplication: v_clip = MVP * v_world
        c_x = mvp[0,0]*px + mvp[0,1]*py + mvp[0,2]*pz + mvp[0,3]*pw
        c_y = mvp[1,0]*px + mvp[1,1]*py + mvp[1,2]*pz + mvp[1,3]*pw
        c_z = mvp[2,0]*px + mvp[2,1]*py + mvp[2,2]*pz + mvp[2,3]*pw
        c_w = mvp[3,0]*px + mvp[3,1]*py + mvp[3,2]*pz + mvp[3,3]*pw
        
        # Frustum Culling (simple w check)
        if c_w <= 0.001:
            return
            

        inv_w = 1.0 / c_w
        ndc_x = c_x * inv_w
        ndc_y = c_y * inv_w
        ndc_z = c_z * inv_w # Depth
        
    
        # NDC is [-1, 1]. Map to [0, width]
        screen_x = (ndc_x + 1.0) * 0.5 * width
        screen_y = (1.0 - ndc_y) * 0.5 * height # Flip Y for image coords
        
        # Bounds Check
        ix = int(screen_x)
        iy = int(screen_y)
        
        if 0 <= ix < width and 0 <= iy < height:
            # Splat density
            # Depth weighting could be added here (1/c_w), but flat 1.0 looks good for "glowing" gas
            cuda.atomic.add(grid, (iy, ix), 1.0)

def generate_3d_frame(pos_d, step):
    """
    Orchestrates the 3D frame generation.
    1. Computes Host MVP matrix.
    2. Sends MVP to Device.
    3. Runs Kernel.
    4. Colors and converts to Image.
    """

    density_grid_d.fill(0)
    

    box_center = (BOX_SIZE/2, BOX_SIZE/2, BOX_SIZE/2)
    mvp_host = get_mvp_matrix(step, box_center, BOX_SIZE)
    mvp_device = cp.asarray(mvp_host) # Transfer to GPU
    

    render_3d_density_kernel[BPG, TPB](pos_d, density_grid_d, mvp_device, RES, RES)
    

    img_d = cp.log1p(density_grid_d)
    v_max = cp.max(img_d) + 1e-5
    img_d /= v_max
    

    r = cp.clip(img_d * 1.5 - 0.5, 0, 1) 
    g = cp.clip(img_d * 0.8, 0, 1)
    b = cp.clip(img_d * 0.8 + 0.2, 0, 1)
    
    mask = (img_d > 0.0)
    r *= mask
    g *= mask
    b *= mask
    
    rgb_d = cp.stack((r, g, b), axis=-1)
    
    return Image.fromarray((rgb_d.get() * 255).astype(np.uint8))

def generate_3d_frame_gpu(pos_d, step, video_buffer_d):
    """
    Renders the frame and stores it directly into the GPU video buffer.
    No CPU transfer happens here.
    """
    density_grid_d.fill(0)
    
    box_center = (BOX_SIZE/2, BOX_SIZE/2, BOX_SIZE/2)
    mvp_host = get_mvp_matrix(step, box_center, BOX_SIZE)
    mvp_device = cp.asarray(mvp_host) 
    
    render_3d_density_kernel[BPG, TPB](pos_d, density_grid_d, mvp_device, RES, RES)
    
    img_d = cp.log1p(density_grid_d)
    v_max = cp.max(img_d) + 1e-5
    img_d /= v_max
    
    r = cp.clip(img_d * 1.5 - 0.5, 0, 1) 
    g = cp.clip(img_d * 0.8, 0, 1)
    b = cp.clip(img_d * 0.8 + 0.2, 0, 1)
    
    mask = (img_d > 0.0)
    r *= mask
    g *= mask
    b *= mask
    
    rgb_d = cp.stack((r, g, b), axis=-1)
    

    video_buffer_d[step] = (rgb_d * 255).astype(cp.uint8)

def compile_video(image_folder, output_file, fps=30):
    """
    Compiles a sequence of PNGs into an MP4 using FFmpeg.
    """
    print("Compiling Video...")
    
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', f'{image_folder}/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video saved successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print("Error compiling video.")
        print(e.stderr.decode())


# --- MEMORY ALLOCATION ---
density_grid_d = cp.zeros((RES, RES), dtype=cp.float32)
children_d = cp.full((num_nodes, 2), -1, dtype=cp.int32)
parents_d = cp.full(num_nodes, -1, dtype=cp.int32)
node_mass_d = cp.zeros(num_nodes, dtype=cp.float32)
node_com_d = cp.zeros((num_nodes, 3), dtype=cp.float32)
counters_d = cp.zeros(num_nodes, dtype=cp.int32)
force_d = cp.zeros((N_PARTICLES, 3), dtype=cp.float32)
root_idx_d = cp.zeros(1, dtype=cp.int32)
node_min_d = cp.zeros((num_nodes, 3), dtype=cp.float32) #Bottleneck compute forces
node_max_d = cp.zeros((num_nodes, 3), dtype=cp.float32)

# --- INITIALIZATION ---
print(f"Initializing Simulation: {N_PARTICLES} particles")
pos_d, vel_d, mass_d = generate_zeldovich_ics(N_GRID, BOX_SIZE, spectral_index=2.2, amplitude=40.0)
a = A_START
delta_a = (A_END - A_START) / N_STEPS

print(f"Allocating VRAM for {N_STEPS} frames... ({N_STEPS * RES**2 * 3 / 1e9:.2f} GB)")
video_buffer_d = cp.zeros((N_STEPS, RES, RES, 3), dtype=cp.uint8)

print(f"Starting Render -> {OUTPUT_DIR}/")


t_start = time.time()
cuda.synchronize()

# Create the progress bar object
with tqdm(range(N_STEPS), desc="Simulating") as pbar:
    
    for step in pbar:
        
        H_current = H0 * (a ** -1.5)
        if H_current > MAX_H_RATE: H_current = MAX_H_RATE
        denom = 1.0 + 2.0 * H_current * DT
        drag_factor = 1.0 / denom 

        if drag_factor < 0.95:
            drag_factor = 0.95

        pos_d, vel_d, mass_d, codes_d = sort_bodies(pos_d, vel_d, mass_d)
        
        children_d.fill(-1)
        parents_d.fill(-1)
        build_radix_tree_kernel[BPG, TPB](codes_d, children_d, parents_d)


        counters_d.fill(0) #reset counters
        node_mass_d.fill(0)
        node_com_d.fill(0)
        node_min_d.fill(0) 
        node_max_d.fill(0)

        compute_multipoles_kernel[BPG, TPB](pos_d, mass_d, children_d, parents_d, 
                                            node_mass_d, node_com_d, 
                                            node_min_d, node_max_d, counters_d)
        find_root_kernel[BPG, TPB](parents_d, root_idx_d)
        
        for i_offset in range(0, N_PARTICLES, BATCH_SIZE): #Batching logic to avoid TDR
            current_batch = min(BATCH_SIZE, N_PARTICLES - i_offset)
            blocks = (current_batch + TPB - 1) // TPB
            compute_forces_kernel[blocks, TPB](pos_d, mass_d, children_d, 
                                               node_mass_d, node_com_d, 
                                               node_min_d, node_max_d, 
                                               force_d, THETA, G, SOFTENING, BOX_SIZE, 
                                               root_idx_d, i_offset)
            
        integrate_kernel[BPG, TPB](pos_d, vel_d, force_d, DT, BOX_SIZE, drag_factor)
    
        generate_3d_frame_gpu(pos_d, step, video_buffer_d)
        a += delta_a

cuda.synchronize()

def save_frame_to_disk(idx, frame_data, output_dir):
    """
    Helper function for the thread pool
    """
    img = Image.fromarray(frame_data)
    img.save(f"{output_dir}/frame_{idx:04d}.png")

print("Transferring video from GPU to CPU...")
video_buffer_host = video_buffer_d.get() # The big transfer (1.8GB)
print("Transfer Complete. Saving frames to disk...")

# Parallel saving (Uses CPU cores efficiently)
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for i in range(N_STEPS):
        futures.append(executor.submit(save_frame_to_disk, i, video_buffer_host[i], OUTPUT_DIR))
    
    # Wait for all to finish
    for _ in tqdm(concurrent.futures.as_completed(futures), total=N_STEPS, desc="Writing PNGs"):
        pass

print("All frames saved.")

# Compile Video
compile_video(OUTPUT_DIR, VIDEO_NAME, fps=60)

cuda.synchronize()
cp.cuda.Device().synchronize()
time.sleep(2)