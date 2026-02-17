import numba
from numba import cuda

"""
Tree Construction Kernels
-------------------------------

Based upon the work of :
Tero Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, 
and k-d Trees", High-Performance Graphics (2012).
DOI: 10.2312/EGGH/HPG12/033-037
URL: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

The idea here is to construct the tree in parallel.
"""

@cuda.jit(device=True)
def clz64(v):

    """
    Count leading zeros kernel but on a 64-bit logic.
    
    :param v: 64 bit integer
    """

    if v == 0: return 64
    n = 0
    if (v & 0xFFFFFFFF00000000) == 0: n += 32; v <<= 32
    if (v & 0xFFFF000000000000) == 0: n += 16; v <<= 16
    if (v & 0xFF00000000000000) == 0: n += 8; v <<= 8
    if (v & 0xF000000000000000) == 0: n += 4; v <<= 4
    if (v & 0xC000000000000000) == 0: n += 2; v <<= 2
    if (v & 0x8000000000000000) == 0: n += 1
    return n

@cuda.jit(device=True)
def delta_fn(codes, i, j, n):

    """
    Computes the number of common leading bits between the Morton codes of bodies i and j.
    This is used to determine the position of the split in the sorted Morton code array during tree construction.

    Params:

    :codes : DeviceArray\\
        Array of Morton codes for all bodies

    :i, j : int\\
        Indices of the bodies to compare
    """

    if j < 0 or j >= n:
        return -1
    
    if codes[i] == codes[j]: 
        return 64 + (63 - clz64( i ^ j))
    
    return clz64(codes[i] ^ codes[j])

@cuda.jit
def build_tree(codes, children, parents):

    """
    Builds a binary radix tree structure from sorted Morton codes.

    Param:

    :codes: DeviceArray\\
        Array of sorted Morton codes for all bodies
    :children: DevicArray\\
        Output array to store the indices of the left and right children for each internal node
    :parents: DeviceArray \\
        Output array to store the index of the parent for each node
    """

    i = cuda.grid(1)
    n = codes.shape[0]
    if i >= n - 1: 
        return
    
    idx = i + n
    
    d_prev = delta_fn(codes, i, i - 1, n)
    d_next = delta_fn(codes, i, i + 1, n)
    direction = 1 if d_next > d_prev else -1
    min_delta = d_prev if d_next > d_prev else d_next
    
    l_max = 2
    while delta_fn(codes, i, i + l_max * direction, n) > min_delta: 
        l_max *= 2
    
    l = 0
    t = l_max // 2
    while t >= 1:
        if delta_fn(codes, i, i + (l + t) * direction, n) > min_delta: 
            l += t
        t //= 2

    j = i + l * direction
    delta_node = delta_fn(codes, i, j, n)
    
    s = 0
    t = l
    while t > 0:
        step = (t + 1) // 2
        if delta_fn(codes, i, i + (s + step) * direction, n) > delta_node: 
            s += step
        t -= step

    gamma = i + s * direction

    if direction == -1: 
        gamma -= 1
    
    left_child = gamma
    right_child = gamma + 1
    
    if min(i, j) == left_child:
        children[idx, 0] = left_child
    else:
        children[idx, 0] = left_child + n

    if max(i, j) == right_child:
        children[idx, 1] = right_child
    else:
        children[idx, 1] = right_child + n

    parents[children[idx, 0]] = idx
    parents[children[idx, 1]] = idx

@cuda.jit
def compute_multipoles(pos_x, pos_y, pos_z, mass, children, parents, n_mass, n_com, n_min, n_max, counts):

    """
    CUDA kernel to compute the multipole moments (mass, center of mass, bounding box) for each node in the tree.
    This kernel traverses from the leaves (bodies) up to the root, accumulating the properties of child nodes into their parents.

    Param:
    
    :pos_x, pos_y, pos_z : DeviceArray\\
        Floating point coordinates of each body on the GPU
    :mass : DeviceArray\\
        Mass of each body
    :children, parents : DeviceArray\\
        Arrays describing the tree structure
    :n_mass, n_com, n_min, n_max : DeviceArray\\
        Arrays to store the multipole moments for each node in the tree
    :counts : DeviceArray\\
        Array to track how many children have been processed for each node
    """

    i = cuda.grid(1)
    if i >= pos_x.shape[0]: 
        return
    
    m = mass[i]
    px, py, pz = pos_x[i], pos_y[i], pos_z[i]
    
    n_mass[i] = m
    n_com[i, 0], n_com[i, 1], n_com[i, 2] = px*m, py*m, pz*m
    n_min[i, 0], n_min[i, 1], n_min[i, 2] = px, py, pz
    n_max[i, 0], n_max[i, 1], n_max[i, 2] = px, py, pz
    
    curr = i
    while True:
        parent = parents[curr]
        if parent == -1: 
            break
        if cuda.atomic.add(counts, parent, 1) == 0: 
            break
        
        l = children[parent, 0]
        r = children[parent, 1]
        
        m_sum = n_mass[l] + n_mass[r]
        n_mass[parent] = m_sum
        n_com[parent, 0] = n_com[l, 0] + n_com[r, 0]
        n_com[parent, 1] = n_com[l, 1] + n_com[r, 1]
        n_com[parent, 2] = n_com[l, 2] + n_com[r, 2]
        
        n_min[parent, 0] = min(n_min[l, 0], n_min[r, 0])
        n_min[parent, 1] = min(n_min[l, 1], n_min[r, 1])
        n_min[parent, 2] = min(n_min[l, 2], n_min[r, 2])
        
        n_max[parent, 0] = max(n_max[l, 0], n_max[r, 0])
        n_max[parent, 1] = max(n_max[l, 1], n_max[r, 1])
        n_max[parent, 2] = max(n_max[l, 2], n_max[r, 2])
        
        cuda.threadfence()
        
        curr = parent

@cuda.jit
def normalize_com(n_mass, n_com, n_nodes):

    """
    CUDA kernel to normalize the center of mass for each node in the tree.
    
    Param:

    :n_mass : DeviceArray\\
        Aray containing the total mass for each node in the tree
    :n_com : DeviceArray\\
        Output array containing the center of mass for each node in the tree
    :n_nodes : int\\
        Total number of nodes in the tree
    """

    i = cuda.grid(1)
    if i < n_nodes and n_mass[i] > 0:
        inv = 1.0 / n_mass[i]
        n_com[i, 0] *= inv
        n_com[i, 1] *= inv
        n_com[i, 2] *= inv

@cuda.jit
def find_root(parents, root_idx):

    """
    CUDA kernel to find the index of the root node in the tree. 
    The root node is identified by having a parent index of -1.

    Param:

    :parents : DeviceArray\\
        Array containing the parent index for each node in the tree
    :root_idx : DeviceArray\\
        Output array to store the index of the root node
    """

    i = cuda.grid(1)
    if i < parents.shape[0]:
        if parents[i] == -1:
            cuda.atomic.exch(root_idx, 0, i)

@cuda.jit
def reset_tree_buffers(child, parent, counters, root_idx):

    """
    CUDA kernel to safely reset buffers, avoiding CuPy and Numba mess up.

    Param:
    
    :child, parent : DeviceArray\\
        Arrays containing child indexes for each nodes
    :parents : DeviceArray\\
        Array containing the parent index for each node in the tree
    :counters : DeviceArray\\
        Array to track how many children have been processed for each node
    :root_idx: DeviceArray\\
        Array to store the index of the root node
    """

    i = cuda.grid(1)
    if i == 0:
        root_idx[0] = 0 
        
    if i < child.shape[0]:
        child[i, 0] = -1
        child[i, 1] = -1
        parent[i] = -1
        counters[i] = 0