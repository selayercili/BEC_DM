# src/dark_matter.py
import numpy as np

# physical constants
eV_to_J = 1.602176634e-19
hbar = 1.054571817e-34
c_light = 299792458.0

def ul_dm_cosine_potential(grid_coords, amplitude_J, m_phi_ev,
                           phase0=0.0, v_dm=220e3, direction=0.0,
                           spatial_modulation=True):
    """
    Build a ULDM cosine potential function with a **consistent interface**:
        V_dm((X,Y), t) -> ndarray [ny, nx]

    Parameters
    ----------
    grid_coords : tuple (X, Y)
        Meshgrid coordinate arrays (units: meters)
    amplitude_J : float
        Amplitude of the DM potential in Joules
    m_phi_ev : float
        ULDM 'mass' in eV (mass-energy)
    phase0 : float
        Initial phase offset (rad)
    v_dm : float
        DM bulk speed [m/s]
    direction : float
        Propagation direction (radians)
    spatial_modulation : bool
        If True, include k·r spatial modulation; otherwise only cos(ω t + φ)
    """
    X, Y = grid_coords

    # angular frequency (rad/s): ω = (m_phi * eV_to_J) / ħ
    omega = (m_phi_ev * eV_to_J) / hbar

    # convert mass-energy to mass (kg): m = (m_phi * eV_to_J) / c^2
    mass_kg = (m_phi_ev * eV_to_J) / (c_light**2)

    # correct de Broglie wavevector magnitude (k = m * v / ħ)
    if spatial_modulation:
        k_mag = (mass_kg * v_dm) / hbar
        kx = k_mag * np.cos(direction)
        ky = k_mag * np.sin(direction)
    else:
        kx = ky = 0.0

    def V_dm(coords, t):
        """
        ULDM potential at given coordinates and time.
        
        Parameters
        ----------
        coords : tuple (X, Y)
            Coordinate arrays (can be different from grid_coords)
        t : float
            Time in seconds
            
        Returns
        -------
        ndarray
            Potential values with same shape as coordinate arrays
        """
        X_eval, Y_eval = coords
        
        if spatial_modulation:
            spatial_phase = kx * X_eval + ky * Y_eval
        else:
            spatial_phase = 0.0
            
        return amplitude_J * np.cos(omega * t + spatial_phase + phase0)

    return V_dm