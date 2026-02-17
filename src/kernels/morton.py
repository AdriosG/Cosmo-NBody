import numba
from numba import cuda, uint64

"""
Morton Encoding & Z-Order Curve Kernels
-------------------------------

The mapping of the 3D space has to be done within a 1D array for GPU computing.
It has to preserve spacial locality to ensure proper and efficient tree construction and traversal.

"""

@cuda.jit(device=True, inline=True)
def spread_bits(v: uint64) -> uint64:

    """
    Spread the bits of a 21-bit integer by inserting two zeros between each bit.
    
    This is achieved using bitwise masks and shifts to transform:
    binary: 000000000000000000000ABC
    into:   000A00B00C

    Param:
    
    :v : uint64 A 21 bit integer

    Returns:
    
    :uint64 A 63 bit integer with the bits of v spread out with two zeros in between.
    
    """
    
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v <<  8)) & 0x0300F00F
    v = (v | (v <<  4)) & 0x030C30C3
    v = (v | (v <<  2)) & 0x09249249
    return v


@cuda.jit(device=True)
def get_morton_code(x : float, y : float, z : float) -> uint64:

    """
    Computes a 63 bit Morton code for a 3D point within a normalized cube [0, 1].
    This kernel scales the input coordinates with a level of precision 2^21 = 2,097,152 per axis.

    Param:

    :x, y, z : float The normalized coordinates in [0, 1] for each axis.

    Returns:
    
    :uint64 The 63-bit Morton code for the given 3D point.

    """

    scale = 2097152.0 # 2^21 scale factor to fit coordinates into 21 bits
    ix = uint64(x * scale)
    iy = uint64(y * scale)
    iz = uint64(z * scale)
    
    #Interleave the bits of ix, iy, and iz to create the Morton code
    return spread_bits(ix) | (spread_bits(iy) << 1) | (spread_bits(iz) << 2)

@cuda.jit
def compute_codes(pos_x, pos_y, pos_z, codes, indices, n_bodies):

    """
    CUDA Kernel to compute Morton codes for all particles in parallel
    
    Param:

    :pos_x, pos_y, pos_z : DeviceArray\\
        Floating position arrays on GPU
    :codes : DeviceArray\\
        Output uint64 array to store the computed Morton codes
    :indices : DeviceArray\\
        uint64 array to store the original indices of particles
    :n_bodies : int\\ 
        The number of bodies to process
    """

    i = cuda.grid(1)
    if i < n_bodies:
        
        px = pos_x[i]
        py = pos_y[i]
        pz = pos_z[i]
        
        codes[i] = get_morton_code(px, py, pz)
        indices[i] = i