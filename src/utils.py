# src/utils.py
"""
Utility constants and helpers for the BEC simulation project.
"""

from pathlib import Path
import numpy as np

# File locations (relative)
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
TIME_SERIES_DIR = RESULTS_DIR / "time_series"
SPECTRA_DIR = RESULTS_DIR / "spectra"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in (RESULTS_DIR, TIME_SERIES_DIR, SPECTRA_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Physical constants (SI)
hbar = 1.054571817e-34  # J*s
m_p = 1.67262192369e-27  # kg (proton mass) - placeholder for typical particle mass
c = 2.99792458e8  # m/s
G = 6.67430e-11  # gravitational constant

# Small helpers
def grid_coords(nx, ny, dx, dy):
    """
    Return (x, y) coordinate 2D arrays centered at zero.
    """
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    return np.meshgrid(x, y, indexing="xy")
