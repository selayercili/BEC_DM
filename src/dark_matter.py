# src/dark_matter.py
"""
Simple dark-matter (ULDM) potential generator.

We model the DM as a classical oscillating field that creates a
time-dependent potential V_DM(r, t) = A(r) * cos(omega * t + phi0),
with optional spatial phase (k·r) for a moving "wind".
"""

import numpy as np
from typing import Tuple

def ul_dm_cosine_potential(xy: Tuple[np.ndarray, np.ndarray],
                           amplitude_J: float,
                           m_phi_ev: float,
                           phase0: float = 0.0,
                           v_dm: float = 220e3,
                           direction: float = 0.0,
                           spatial_modulation: bool = True):
    """
    Build a closure function V_DM(grid_coords, t) that returns the potential at time t.

    Args:
        xy: (X, Y) meshgrid arrays in meters.
        amplitude_J: amplitude of the potential in Joules (scalar).
        m_phi_ev: ULDM mass in eV -> used to compute angular frequency omega = m c^2 / hbar
        phase0: initial phase of the field (radians)
        v_dm: effective DM wind speed (m/s) - used if spatial_modulation True
        direction: wind direction angle (radians) in the XY plane
        spatial_modulation: if True include k·r term to simulate moving wave

    Returns:
        function V(grid_coords, t) -> potential array (same shape as X)
    """
    # constants
    hbar = 1.054571817e-34
    c = 2.99792458e8  # m/s
    eV_to_J = 1.602176634e-19
    m_phi_J = m_phi_ev * eV_to_J
    omega = m_phi_J * c**2 / hbar  # angular frequency (rad/s)

    X, Y = xy

    if spatial_modulation:
        # wavenumber magnitude k = m_phi * v / ħ  (order-of-magnitude)
        k = (m_phi_J * v_dm) / (hbar * c**2)  # 1/m
        kx = k * np.cos(direction)
        ky = k * np.sin(direction)
        kr = kx * X + ky * Y
    else:
        kr = 0.0

    def V_dm(grid_coords: Tuple[np.ndarray, np.ndarray], t: float):
        """
        Returns potential grid in Joules.
        
        Args:
            grid_coords: (X, Y) coordinate arrays (not used in calculation but kept for interface consistency)
            t: time in seconds
        """
        # returns potential grid in Joules
        return amplitude_J * np.cos(omega * t - kr + phase0)

    # also provide metadata
    V_dm.omega = omega
    V_dm.m_phi_ev = m_phi_ev
    return V_dm