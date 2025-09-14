# src/dark_matter.py
import numpy as np

def ul_dm_cosine_potential(grid_coords, amplitude_J, m_phi_ev,
                           phase0=0.0, v_dm=220e3, direction=0.0,
                           spatial_modulation=True):
    """
    Return a callable V_dm(X,Y,t) that produces a spatially and temporally
    varying ULDM potential:
        V(x,y,t) = A * cos(ω t + k·r + φ0)

    Parameters
    ----------
    grid_coords : (X,Y)
        Meshgrid arrays from BECSimulation (shape [nx,ny]).
    amplitude_J : float
        Amplitude of the DM potential (J).
    m_phi_ev : float
        Mass of ULDM particle in eV.
    phase0 : float
        Initial phase offset (rad).
    v_dm : float
        Characteristic DM velocity (m/s).
    direction : float
        Propagation direction of DM field (radians).
    spatial_modulation : bool
        If True, include a plane-wave spatial dependence. If False,
        only time oscillations are included.

    Returns
    -------
    function V_dm(X,Y,t) -> ndarray with shape [nx,ny]
    """
    X, Y = grid_coords

    # --- Convert ULDM mass to angular frequency ---
    eV_to_J = 1.602176634e-19
    hbar = 1.054571817e-34
    omega = (m_phi_ev * eV_to_J) / hbar  # rad/s

    # --- DM wavevector magnitude ---
    k_mag = (m_phi_ev * eV_to_J) / (hbar * 3e8) * v_dm

    # Directional wavevector components
    kx = k_mag * np.cos(direction)
    ky = k_mag * np.sin(direction)

    # Spatial phase term
    if spatial_modulation:
        spatial_phase = kx * X + ky * Y
    else:
        spatial_phase = 0.0

    def V_dm(t):
        # Full potential: cos(ω t + k·r + φ0)
        return amplitude_J * np.cos(omega * t + spatial_phase + phase0)

    return V_dm
