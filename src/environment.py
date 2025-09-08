# src/environment.py
import numpy as np
from typing import Callable, Tuple

# Physical constants used here (SI)
G = 6.67430e-11  # m^3 kg^-1 s^-2

def neutron_star_potential(r: np.ndarray, M_ns: float = 1.4 * 1.9885e30, R_ns: float = 1e4) -> np.ndarray:
    """
    Simple Newtonian gravitational potential for a (spherically symmetric) neutron star.
    Returns V(r) = -G M / max(r, R_ns).  Units: J/kg * kg = J (potential energy per test mass).
    Note: This is a toy Newtonian potential. Inside the star (r < R_ns) we clamp r to R_ns
    to avoid singularity; more detailed interiors can be used if desired.
    """
    r_safe = np.maximum(r, R_ns)
    V = -G * M_ns / r_safe
    return V

def create_environment_potential(X: np.ndarray, Y: np.ndarray,
                                 potential_func: Callable[[np.ndarray], np.ndarray] = neutron_star_potential,
                                 **kwargs) -> np.ndarray:
    """
    Given meshgrid X,Y build a static environment potential array.
    Returns a 2D ndarray V_env with same shape as X and Y.
    """
    r = np.sqrt(X**2 + Y**2)
    V_env = potential_func(r, **kwargs)
    return V_env

# Optional helper for quick plotting or inspection (not required):
def plot_potential(V: np.ndarray, x=None, y=None, show=True, outpath=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(V, origin='lower', interpolation='nearest')
    ax.set_title("Environment potential (grid)")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, label="Potential [J]")
    if outpath:
        fig.savefig(outpath, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
