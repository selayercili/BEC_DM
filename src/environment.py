import numpy as np

def neutron_star_potential(r, M_ns=1.4*1.9885e30, R_ns=1e4):
    """
    Neutron star gravitational potential.
    r: radial distance from star center [m]
    M_ns: mass [kg], R_ns: radius [m]
    """
    G = 6.67430e-11
    return -G * M_ns / np.maximum(r, R_ns)

def apply_environment(grid_positions, potential_func, **kwargs):
    """
    Apply potential_func to a grid of positions.
    Returns potential array.
    """
    V = np.array([potential_func(r, **kwargs) for r in grid_positions.flatten()])
    return V.reshape(grid_positions.shape)
