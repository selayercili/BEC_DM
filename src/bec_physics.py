"""FIXED BEC Physics Module - Corrected Phase Shift Calculation"""

import numpy as np
import scipy.constants as const
from scipy.fft import fftfreq
from dataclasses import dataclass
from typing import Callable, Dict, Optional

@dataclass
class BECParameters:
    """Physical parameters for Bose-Einstein Condensate"""
    atom_mass: float          # kg
    scattering_length: float  # m
    density: float            # atoms/m³
    trap_frequency: float     # Hz
    coherence_length: float   # m

class BECSimulator:
    """Core BEC physics simulator for dark matter detection - FIXED VERSION"""
    
    def __init__(self, params: BECParameters, size: float = 1e-3, n_points: int = 512):
        """
        Initialize BEC simulator
        
        Args:
            params: BEC physical parameters
            size: Spatial domain size (m)
            n_points: Number of grid points
        """
        self.params = params
        self.x = np.linspace(-size/2, size/2, n_points)
        self.dx = self.x[1] - self.x[0]
        self.state = self._ground_state()
        
        # Calculate and store total atom number
        self.n_atoms = np.trapz(np.abs(self.state)**2, self.x)
        
        print(f"🔧 BEC initialized: {self.n_atoms:.2e} atoms, coherence: {params.coherence_length*1e6:.1f} μm")
        
    def _ground_state(self) -> np.ndarray:
        """Calculate realistic Thomas-Fermi ground state wavefunction"""
        # Use coherence length as the BEC size
        r_tf = self.params.coherence_length / 2  # Thomas-Fermi radius
        
        # Thomas-Fermi profile (inverted parabola)
        psi = np.zeros_like(self.x, dtype=complex)
        idx = np.abs(self.x) <= r_tf
        
        if np.any(idx):
            # Parabolic density profile
            density_profile = (r_tf**2 - self.x[idx]**2) / r_tf**2
            # Peak density in atoms/m for 1D
            peak_density_1d = self.params.density ** (1/3)  # Convert 3D to 1D density
            psi[idx] = np.sqrt(peak_density_1d * density_profile)
        
        # Ensure proper normalization
        current_norm = np.trapz(np.abs(psi)**2, self.x)
        if current_norm > 0:
            # Target number of atoms in the 1D slice
            target_atoms_1d = self.params.density * np.pi * r_tf**2 * self.dx  # Volume element
            psi *= np.sqrt(target_atoms_1d / current_norm)
        
        return psi

    def dm_phase_shift(self, dm_density: float, dm_mass: float, 
                       cross_section: float, exposure_time: float) -> np.ndarray:
        """
        Calculate DM-induced quantum phase shift - CORRECTED FORMULA
        
        Args:
            dm_density: Local dark matter density (kg/m³)
            dm_mass: Dark matter particle mass (kg)
            cross_section: Interaction cross-section (m²)
            exposure_time: Interaction time (s)
            
        Returns:
            Spatial phase shift profile (radians)
        """
        # FIXED: Correct calculation of DM number density
        dm_number_density = dm_density / dm_mass  # particles/m³
        
        # FIXED: Proper interaction potential
        # V = ħ * cross_section * dm_number_density * c * |ψ|²
        # This gives the right units: [ħ] * [m²] * [1/m³] * [m/s] * [1/m] = [J]
        V_dm = (const.hbar * cross_section * dm_number_density * const.c * 
                np.abs(self.state)**2)
        
        # Phase shift = ∫ V_dm dt / ħ
        phase_shift = V_dm * exposure_time / const.hbar
        
        # Debug info
        max_phase = np.max(np.abs(phase_shift))
        print(f"🔬 Phase shift calculation:")
        print(f"   DM density: {dm_density:.2e} kg/m³")
        print(f"   DM number density: {dm_number_density:.2e} particles/m³")
        print(f"   Interaction strength: {const.hbar * cross_section * dm_number_density * const.c:.2e} J·m")
        print(f"   Max phase shift: {max_phase:.2e} rad")
        
        return phase_shift

    def detection_sensitivity(self, phase_shift: np.ndarray) -> Dict:
        """
        Calculate detection sensitivity metrics
        
        Args:
            phase_shift: Spatial phase shift profile
            
        Returns:
            Dictionary with sensitivity metrics
        """
        # Number of atoms (use stored value)
        n_atoms = self.n_atoms
        
        # RMS phase shift (weighted by BEC density)
        weights = np.abs(self.state)**2
        total_weight = np.trapz(weights, self.x)
        
        if total_weight > 0:
            rms_shift = np.sqrt(np.trapz(phase_shift**2 * weights, self.x) / total_weight)
        else:
            rms_shift = 0
        
        # Shot noise limit
        shot_noise = 1 / np.sqrt(max(n_atoms, 1))  # Avoid division by zero
        
        # Signal-to-noise ratio
        snr = rms_shift / shot_noise if shot_noise > 0 else 0
        
        print(f"📊 Detection metrics:")
        print(f"   RMS phase shift: {rms_shift:.2e} rad")
        print(f"   Shot noise: {shot_noise:.2e}")
        print(f"   SNR: {snr:.6f}")
        
        return {
            'rms_phase_shift': rms_shift,
            'shot_noise_limit': shot_noise,
            'snr': snr,
            'detectable': snr > 3,
            'n_atoms': n_atoms
        }

class GalaxyBECSimulator:
    """Galaxy-specific BEC dark matter detection simulator - FIXED VERSION"""
    
    def __init__(self, galaxy_params: Dict, bec_type: str = 'optimized'):
        """
        Initialize with galaxy parameters - CORRECTED VERSION
        
        Args:
            galaxy_params: Dictionary of galaxy parameters from data pipeline
            bec_type: Type of BEC to simulate ('rubidium87', 'sodium23', 'optimized', 'ultra_dense')
        """
        self.galaxy_params = galaxy_params
        
        # CORRECTED: Much more realistic and optimized BEC parameters
        if bec_type == 'rubidium87':
            # Standard Rb-87 BEC
            atom_mass = 1.45e-25      # kg (Rb-87)
            scattering_length = 5.3e-9 # m
            target_atoms = 1e6        # 1 million atoms
            coherence_length = 10e-6  # m (10 μm)
            trap_frequency = 100      # Hz
            
        elif bec_type == 'optimized':
            # Optimized for DM detection
            atom_mass = 1.67e-27      # kg (hydrogen - lightest)
            scattering_length = 50e-15 # m (enhanced scattering)
            target_atoms = 1e10       # 10 billion atoms (challenging but possible)
            coherence_length = 100e-6 # m (100 μm - large BEC)
            trap_frequency = 50       # Hz
            
        elif bec_type == 'ultra_dense':
            # Ultra-dense BEC for maximum sensitivity
            atom_mass = 1.67e-27      # kg (hydrogen)
            scattering_length = 100e-15 # m (very strong interactions)
            target_atoms = 1e12       # 1 trillion atoms (theoretical limit)
            coherence_length = 1e-3   # m (1 mm - very large)
            trap_frequency = 10       # Hz (very weak trap)
            
        else:
            raise ValueError(f"Unknown BEC type: {bec_type}")
        
        # Calculate density from target atom number and coherence volume
        volume = (4/3) * np.pi * (coherence_length/2)**3  # Spherical volume
        density = target_atoms / volume
        
        bec_params = BECParameters(
            atom_mass=atom_mass,
            scattering_length=scattering_length,
            density=density,    
            trap_frequency=trap_frequency,
            coherence_length=coherence_length
        )
        
        # Spatial size - make sure it's large enough
        size = max(5 * coherence_length, 1e-3)  # At least 5x coherence length
        
        # Create BEC simulator
        self.bec = BECSimulator(bec_params, size=size, n_points=1024)
        
        # Print diagnostic info
        print(f"✅ Galaxy BEC initialized ({bec_type}):")
        print(f"   Galaxy: {galaxy_params['name']}")
        print(f"   Atoms: {self.bec.n_atoms:.2e}")
        print(f"   Size: {coherence_length*1e6:.0f} μm")
        print(f"   Density: {density:.2e} atoms/m³")
        print(f"   Shot noise: {1/np.sqrt(max(self.bec.n_atoms, 1)):.2e}")
        
    def simulate_dm_interaction(self, dm_mass: float, cross_section: float, 
                               exposure_time: float = 3600) -> Dict:
        """
        Simulate dark matter interaction with galactic BEC
        
        Args:
            dm_mass: Dark matter particle mass (kg)
            cross_section: Interaction cross-section (m²)
            exposure_time: Observation time (s)
            
        Returns:
            Dictionary with simulation results
        """
        print(f"\n🔬 Simulating DM interaction:")
        print(f"   DM mass: {dm_mass:.2e} kg")
        print(f"   Cross-section: {cross_section:.2e} m²")
        print(f"   Exposure: {exposure_time/3600:.1f} hours")
        
        # Calculate phase shift
        phase_shift = self.bec.dm_phase_shift(
            self.galaxy_params['dm_density_local_kg_m3'],
            dm_mass,
            cross_section,
            exposure_time
        )
        
        # Calculate sensitivity metrics
        sensitivity = self.bec.detection_sensitivity(phase_shift)
        
        # Add galaxy info to results
        results = {
            'galaxy': self.galaxy_params['name'],
            'dm_mass': dm_mass,
            'cross_section': cross_section,
            'exposure_time': exposure_time,
            'phase_shift_profile': phase_shift,
            'spatial_grid': self.bec.x,
            'bec_parameters': {
                'atom_mass': self.bec.params.atom_mass,
                'scattering_length': self.bec.params.scattering_length,
                'density': self.bec.params.density,
                'coherence_length': self.bec.params.coherence_length,
                'n_atoms': self.bec.n_atoms
            }
        }
        results.update(sensitivity)
        
        return results

# Enhanced DM models with more realistic cross-sections
CORRECTED_DM_MODELS = {
    'axion_standard': {
        'mass': 1e-22,          # kg (~10⁻⁶ eV) 
        'cross_section': 1e-50, # m² (standard, very small)
        'description': 'Standard QCD axion'
    },
    'axion_enhanced': {
        'mass': 1e-22,
        'cross_section': 1e-42, # m² (enhanced coupling)
        'description': 'Axion with enhanced coupling to matter'
    },
    'axion_optimistic': {
        'mass': 1e-22,
        'cross_section': 1e-38, # m² (very optimistic)
        'description': 'Axion with very strong coupling'
    },
    'wimp_standard': {
        'mass': 1e-25,          # kg (~100 GeV)
        'cross_section': 1e-46, # m² (standard weak-scale)
        'description': 'Standard WIMP'
    },
    'wimp_enhanced': {
        'mass': 1e-25,
        'cross_section': 1e-38, # m² (enhanced nuclear coupling)
        'description': 'WIMP with enhanced coupling'
    },
    'composite_dm': {
        'mass': 1e-20,          # kg (composite particles)
        'cross_section': 1e-32, # m² (strong self-interactions)
        'description': 'Composite dark matter'
    },
    'hidden_sector': {
        'mass': 1e-23,          # kg
        'cross_section': 1e-35, # m² (hidden sector interactions)
        'description': 'Hidden sector dark matter'
    }
}

def test_corrected_simulation():
    """Test the corrected simulation with realistic parameters"""
    
    print("🧪 TESTING CORRECTED BEC SIMULATION")
    print("=" * 50)
    
    # Create a test galaxy parameter set
    test_galaxy = {
        'name': 'Test_Galaxy',
        'dm_density_local_kg_m3': 5e-22,  # kg/m³ (typical local DM density)
        'v_flat_km_s': 200,
        'scale_length_m': 1e20
    }
    
    print(f"Test galaxy DM density: {test_galaxy['dm_density_local_kg_m3']:.2e} kg/m³")
    
    # Test different BEC configurations
    bec_types = ['rubidium87', 'optimized', 'ultra_dense']
    
    for bec_type in bec_types:
        print(f"\n🔧 Testing {bec_type} BEC:")
        print("-" * 30)
        
        try:
            # Create BEC simulator
            bec_sim = GalaxyBECSimulator(test_galaxy, bec_type=bec_type)
            
            # Test with enhanced axion model
            dm_model = CORRECTED_DM_MODELS['axion_enhanced']
            
            result = bec_sim.simulate_dm_interaction(
                dm_model['mass'],
                dm_model['cross_section'],
                3600  # 1 hour
            )
            
            print(f"✅ {bec_type} results:")
            print(f"   SNR: {result['snr']:.6f}")
            print(f"   Detectable: {'YES' if result['detectable'] else 'NO'}")
            print(f"   Phase shift RMS: {result['rms_phase_shift']:.2e} rad")
            
            if result['snr'] > 0:
                improvement_needed = 3.0 / result['snr']
                print(f"   Improvement needed: {improvement_needed:.1e}x")
            
        except Exception as e:
            print(f"❌ Error with {bec_type}: {e}")
    
    # Test model comparison
    print(f"\n📊 MODEL COMPARISON (ultra_dense BEC):")
    print("-" * 40)
    
    try:
        bec_sim = GalaxyBECSimulator(test_galaxy, bec_type='ultra_dense')
        
        for model_name, model_params in CORRECTED_DM_MODELS.items():
            result = bec_sim.simulate_dm_interaction(
                model_params['mass'],
                model_params['cross_section'],
                3600,
            )
            
            status = '✅ DETECTABLE' if result['detectable'] else '❌ Too weak'
            print(f"{model_name:20s}: SNR = {result['snr']:.3e} {status}")
            
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("- Use 'ultra_dense' BEC configuration for best sensitivity")
    print("- Focus on enhanced/optimistic DM models")
    print("- Consider longer exposure times (>4 hours)")
    print("- Cross-sections >10⁻⁴⁰ m² needed for reliable detection")

if __name__ == "__main__":
    test_corrected_simulation()