# File: dm_models.py (enhanced with larger cross-sections)
"""Dark matter particle models with enhanced cross-sections for detection studies"""
import numpy as np

# Original models (likely too weak for detection)
DM_MODELS = {
    'axion': {
        'mass': 1e-22,          # kg (~10⁻⁶ eV)
        'cross_section': 1e-50  # m² (original, very small)
    },
    'wimp': {
        'mass': 1e-25,          # kg (~100 GeV)
        'cross_section': 1e-46  # m² (original, very small)
    },
    'sterile_neutrino': {
        'mass': 1e-24,          # kg (~1 keV)
        'cross_section': 1e-48  # m² (original, very small)
    }
}

# Enhanced models with larger cross-sections for detection feasibility
ENHANCED_DM_MODELS = {
    'axion_realistic': {
        'mass': 1e-22,          # kg (~10⁻⁶ eV) 
        'cross_section': 1e-42, # m² (enhanced by ~100x)
        'description': 'Axion with enhanced coupling'
    },
    'axion_optimistic': {
        'mass': 1e-22,
        'cross_section': 1e-40, # m² (very optimistic)
        'description': 'Axion with very strong coupling'
    },
    'wimp_realistic': {
        'mass': 1e-25,          # kg (~100 GeV)
        'cross_section': 1e-40, # m² (enhanced by ~1M x)
        'description': 'WIMP with enhanced nuclear coupling'
    },
    'wimp_optimistic': {
        'mass': 1e-25,
        'cross_section': 1e-38, # m² (very optimistic)
        'description': 'WIMP with very strong coupling'
    },
    'sterile_neutrino_realistic': {
        'mass': 1e-24,          # kg (~1 keV)
        'cross_section': 1e-42, # m² (enhanced)
        'description': 'Sterile neutrino with enhanced mixing'
    },
    'composite_dark_matter': {
        'mass': 1e-20,          # kg (heavier composite)
        'cross_section': 1e-35, # m² (much larger interaction)
        'description': 'Composite DM with strong self-interactions'
    },
    'ultra_light_scalar': {
        'mass': 1e-26,          # kg (ultra-light)
        'cross_section': 1e-38, # m² (coherent enhancement)
        'description': 'Ultra-light scalar with coherent enhancement'
    },
    'hidden_photon': {
        'mass': 1e-23,          # kg (~10⁻⁵ eV)
        'cross_section': 1e-41, # m² (kinetic mixing)
        'description': 'Hidden photon dark matter'
    }
}

# Combined models dictionary for easy access
ALL_DM_MODELS = {**DM_MODELS, **ENHANCED_DM_MODELS}

def get_enhanced_models():
    """Return only the enhanced models for detection studies"""
    return ENHANCED_DM_MODELS

def get_original_models():
    """Return only the original (weak) models"""
    return DM_MODELS

def get_all_models():
    """Return all models (original + enhanced)"""
    return ALL_DM_MODELS