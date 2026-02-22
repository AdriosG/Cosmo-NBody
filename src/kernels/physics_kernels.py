import numba
from numba import cuda, float32, int32
import math


"""
Physics kernels for computing forces using a Barnes-Hut tree.
-------------------------------------------------------------

This module contains CUDA kernels for computing the physics behind the simulation,

"""

@cuda.jit(fastmath=True)
def compute_forces(pos_x, pos_y, pos_z, children, n_mass, n_com, n_min, n_max, force, theta, G, soft, box, root, cache_size, top_nodes, node_to_cache, off):

    """
    CUDA kernel to compute gravitational forces on particles using a Barnes-Hut tree.

    Param:

    :pos_x, pos_y, pos_z: DeviceArray\\
        Arrays containing the x, y, z positions of the particles.
    :children: DeviceArray\\
        Array containing the indices of the child nodes for each node in the tree.
    :n_mass: DeviceArray\\
        Array containing the total mass of each node in the tree.
    :n_com: DeviceArray\\
        Array containing the center of mass (x, y, z) for each node in the tree.
    :n_min, n_max: DeviceArray\\
        Arrays containing the minimum and maximum coordinates of the bounding box for each node in the tree.
    :force: DeviceArray\\
        Output array to store the computed forces on each particle.
    :theta: float\\
        Opening angle parameter for the Barnes-Hut algorithm.
    :G: float\\
        Gravitational constant.
    :soft: float\\
        Sofening parameter to avoid singularities in force calculations.
    :box: float\\
        Size of the simulation box.
    :root: DeviceArray\\
        Array containing the index of the root node of the tree.
    :off: int\\
        Offset for indexing particles in the global array.
    """

    tid = cuda.grid(1)
    tx = cuda.threadIdx.x
    i = tid + off

    sh_mass = cuda.shared.array(cache_size, dtype=float32)
    sh_com_x = cuda.shared.array(cache_size, dtype=float32)
    sh_com_y = cuda.shared.array(cache_size, dtype=float32)
    sh_com_z = cuda.shared.array(cache_size, dtype=float32)
    sh_min_x = cuda.shared.array(cache_size, dtype=float32)
    sh_max_x = cuda.shared.array(cache_size, dtype=float32)
    sh_child_l = cuda.shared.array(cache_size, dtype=int32)
    sh_child_r = cuda.shared.array(cache_size, dtype=int32)

    if tx < cache_size:
        n_idx = top_nodes[tx]
        if n_idx != -1:
            sh_mass[tx] = n_mass[n_idx]
            sh_com_x[tx] = n_com[n_idx, 0]
            sh_com_y[tx] = n_com[n_idx, 1]
            sh_com_z[tx] = n_com[n_idx, 2]
            sh_min_x[tx] = n_min[n_idx, 0]
            sh_max_x[tx] = n_max[n_idx, 0]
            sh_child_l[tx] = children[n_idx, 0]
            sh_child_r[tx] = children[n_idx, 1]
            
    cuda.syncthreads()

    if i >= pos_x.shape[0]: 
        return
    
    px, py, pz = pos_x[i], pos_y[i], pos_z[i]
    ax, ay, az = 0.0, 0.0, 0.0
    
    stack = cuda.local.array(64, dtype=numba.int32) 
    stack_top = 0
    stack[stack_top] = root[0]
    stack_top += 1
    
    theta_sq = theta * theta
    
    while stack_top > 0:

        stack_top -= 1
        node = stack[stack_top]
        c_idx = node_to_cache[node]
        in_cache = c_idx != -1

        if in_cache:
            nm = sh_mass[c_idx]
            n_cx = sh_com_x[c_idx]
            n_cy = sh_com_y[c_idx]
            n_cz = sh_com_z[c_idx]
            n_mx = sh_min_x[c_idx]
            n_mxx = sh_max_x[c_idx]
            c1 = sh_child_l[c_idx]
            c2 = sh_child_r[c_idx]
        else:
            nm = n_mass[node]
            n_cx = n_com[node, 0]
            n_cy = n_com[node, 1]
            n_cz = n_com[node, 2]
            n_mx = n_min[node, 0]
            n_mxx = n_max[node, 0]
            c1 = children[node, 0]
            c2 = children[node, 1]

        if nm <= 0: 
            continue

        dx = n_cx - px
        dy = n_cy - py
        dz = n_cz - pz
        
        dx -= round(dx)
        dy -= round(dy)
        dz -= round(dz)
        
        d2 = dx*dx + dy*dy + dz*dz
        
        is_leaf = (c1 == -1)
        sx = n_mxx - n_mx
        sx = box if sx > box * 0.5 else sx
        
        mac_pass = (sx * sx) < (theta_sq * d2)
        use_node = is_leaf or mac_pass
        
        if use_node:
            if d2 > 1e-7:
                d_inv = 1.0 / math.sqrt(d2 + soft*soft)
                f = G * nm * (d_inv * d_inv * d_inv)
                ax += f * dx
                ay += f * dy
                az += f * dz
        else:
            if stack_top + 1 < 64:
                if c1 != -1:
                    stack[stack_top] = c2
                    stack_top += 1
                if c2 != -1:
                    stack[stack_top] = c1
                    stack_top += 1
            
    force[i, 0] = ax
    force[i, 1] = ay
    force[i, 2] = az

@cuda.jit
def integrate(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, force, a, H,dt):

    """
    CUDA kernel to integrate the positions and velocities of particles using a simple symplectic integrator.

    Param:
    
    :pos_x, pos_y, pos_z: DeviceArray\\
        Arrays containing the x, y, z positions of the particles.
    :vel_x, vel_y, vel_z: DeviceArray\\
        Arrays containing the x, y, z velocities of the particles.
    :force: DeviceArray\\
        Array containing the x, y, z forces acting on each particle.
    :a: float\\
        Scale factor for the simulation.
    :H: float\\
        Hubble parameter for the simulation. 
    :dt: float \\
        Time step for the integration.
    """

    i = cuda.grid(1)
    if i < pos_x.shape[0]:
        
        drag = 1.0 - (H * dt)
        f_scale = 1.0 / a

        vx = vel_x[i] * drag + (force[i, 0] * f_scale * dt)
        vy = vel_y[i] * drag + (force[i, 1] * f_scale * dt)
        vz = vel_z[i] * drag + (force[i, 2] * f_scale * dt)

        vel_x[i] = vx
        vel_y[i] = vy
        vel_z[i] = vz
        
        drift_factor = 1.0 / (a)

        px = pos_x[i] + vx * dt * drift_factor
        py = pos_y[i] + vy * dt * drift_factor
        pz = pos_z[i] + vz * dt * drift_factor
        
        px = px - math.floor(px)
        py = py - math.floor(py)
        pz = pz - math.floor(pz)
        
        pos_x[i] = px
        pos_y[i] = py
        pos_z[i] = pz
