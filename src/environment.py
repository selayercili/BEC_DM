import numpy as np

def neutron_star_potential(r, M_ns=1.4*1.9885e30, R_ns=1e4):
    """
    Neutron star gravitational potential.

    Args:
        r: radial distance from star center [m]
        M_ns: mass [kg] (default ~1.4 solar masses)
        R_ns: neutron star radius [m] (default 10 km)

    Returns:
        Gravitational potential at distance r [J/kg]
    """
    G = 6.67430e-11
    return -G * M_ns / np.maximum(r, R_ns)


def create_environment_potential(X: np.ndarray, Y: np.ndarray, potential_func=neutron_star_potential, **kwargs):
    """
    Create a static environment potential array from a given potential function.

    Args:
        X, Y: meshgrid coordinate arrays
        potential_func: function that takes radial distance and returns potential
        **kwargs: extra arguments for potential_func

    Returns:
        V_env: 2D numpy array of potential values with shape matching X, Y
    """
    r = np.sqrt(X**2 + Y**2)
    V_env = potential_func(r, **kwargs)
    return V_env
