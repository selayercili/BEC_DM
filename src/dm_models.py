# File: dm_models.py (separate DM scenarios)
"""Dark matter particle models"""
import numpy as np

DM_MODELS = {
    'axion': {
        'mass': 1e-22,          # kg (~10⁻⁶ eV)
        'cross_section': 1e-50  # m²
    },
    'wimp': {
        'mass': 1e-25,          # kg (~100 GeV)
        'cross_section': 1e-46  # m²
    },
    'sterile_neutrino': {
        'mass': 1e-24,          # kg (~1 keV)
        'cross_section': 1e-48  # m²
    }
}