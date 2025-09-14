# src/dark_matter.py
import numpy as np

def ul_dm_cosine_potential(grid_coords, amplitude_J, m_phi_ev,
                           phase0=0.0, v_dm=220e3, direction=0.0,
                           spatial_modulation=True):
    """
    Build a ULDM cosine potential function with a consistent interface:
        V_dm((X,Y), t) -> ndarray [nx, ny]

    Parameters
    ----------
    grid_coords : (X,Y)
        Meshgrid arrays (same shape).
    amplitude_J : float
        Amplitude of the potential (J).
    m_phi_ev : float
        ULDM particle mass (eV).
    phase0 : float
        Initial phase (rad).
    v_dm : float
        Typical DM velocity (m/s).
    direction : float
        Propagation direction in radians.
    spatial_modulation : bool
        If True, apply a plane-wave factor exp(i k·r).
    """
    X, Y = grid_coords

    # constants
    eV_to_J = 1.602176634e-19
    hbar = 1.054571817e-34

    # angular frequency
    omega = (m_phi_ev * eV_to_J) / hbar  # rad/s

    # wavevector
    k_mag = (m_phi_ev * eV_to_J) / (hbar * 3e8) * v_dm
    kx, ky = k_mag * np.cos(direction), k_mag * np.sin(direction)

    if spatial_modulation:
        spatial_phase = kx * X + ky * Y
    else:
        spatial_phase = 0.0

    def V_dm(coords, t):
        # ignore coords, we already captured X,Y
        return amplitude_J * np.cos(omega * t + spatial_phase + phase0)

    return V_dm
