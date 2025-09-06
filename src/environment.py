import numpy as np
from typing import Tuple

def neutron_star_potential(r, M_ns=1.4*1.9885e30, R_ns=1e4):
    """
    Neutron star gravitational potential.
    r: radial distance from star center [m]
    M_ns: mass [kg], R_ns: radius [m]
    """
    G = 6.67430e-11
    return -G * M_ns / np.maximum(r, R_ns)

def create_environment_potential(X: np.ndarray, Y: np.ndarray, potential_func, **kwargs):
    """
    Create a static environment potential function that can be called with grid coordinates and time.
    
    Args:
        X, Y: meshgrid coordinate arrays
        potential_func: function that takes radial distance and returns potential
        **kwargs: additional arguments for potential_func
    
    Returns:
        function V_env(grid_coords, t) -> potential array
    """
    # Pre-calculate the radial distances
    r = np.sqrt(X**2 + Y**2)
    # Pre-calculate the potential (static, so we can do this once)
    V_static = potential_func(r, **kwargs)
    
    def V_env(grid_coords: Tuple[np.ndarray, np.ndarray], t: float):
        """
        Environment potential function.
        
        Args:
            grid_coords: (X, Y) coordinate arrays (not used since potential is pre-calculated)
            t: time in seconds (not used for static potential)
            
        Returns:
            potential array in Joules
        """
        return V_static
    
    return V_env