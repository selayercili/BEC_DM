# src/environment.py
import numpy as np
from typing import Callable, Tuple

# Physical constant
G = 6.67430e-11  # m^3 kg^-1 s^-2

def neutron_star_potential(r: np.ndarray, M_ns: float = 1.4 * 1.9885e30, R_ns: float = 1e4) -> np.ndarray:
    """
    Newtonian approximation: V(r) = -G M / r, but clamp r>=R_ns to avoid singularity.
    Returns potential in SI units (J) for a test mass.
    """
    r_safe = np.maximum(r, R_ns)
    return -G * M_ns / r_safe

def create_environment_potential(X: np.ndarray, Y: np.ndarray,
                                 potential_func: Callable[[np.ndarray], np.ndarray] = neutron_star_potential,
                                 center_offset=(0.0, 0.0),
                                 **kwargs) -> np.ndarray:
    """
    Create a static 2D environment potential array.

    Parameters:
      - X, Y : meshgrid arrays (meters)
      - potential_func: function taking r array and returning potential array
      - center_offset: (x0,y0) position of neutron star center in same units as X,Y
    Returns:
      - V_env : 2D ndarray same shape as X
    """
    x0, y0 = center_offset
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    V_env = potential_func(r, **kwargs)
    return V_env

# Small helper: linearized gravity patch for tiny regions
def linearized_gravity(X: np.ndarray, Y: np.ndarray, g_vector=(0.0, -9.8)):
    """
    Return a linear potential approximation V = - g · r for small regions
    (useful if the NS radius makes the true potential almost constant).
    g_vector in m/s^2 (acceleration).
    """
    gx, gy = g_vector
    return - (gx * X + gy * Y)
