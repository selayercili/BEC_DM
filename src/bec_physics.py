"""CORRECTED BEC Physics and Galaxy Simulation Module for Dark Matter Detection"""

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
    """Core BEC physics simulator for dark matter detection"""
    
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
        
    def _ground_state(self) -> np.ndarray:
        """Calculate Thomas-Fermi ground state wavefunction"""
        # Interaction strength
        g = 4 * np.pi * const.hbar**2 * self.params.scattering_length / self.params.atom_mass
        
        # Chemical potential and Thomas-Fermi radius
        omega = 2 * np.pi * self.params.trap_frequency
        
        # For a realistic BEC, use the coherence length directly
        r_tf = self.params.coherence_length / 2  # Thomas-Fermi radius
        
        # Thomas-Fermi profile (inverted parabola)
        psi = np.zeros_like(self.x, dtype=complex)
        idx = np.abs(self.x) <= r_tf
        
        if np.any(idx):
            # Parabolic density profile
            density_profile = (r_tf**2 - self.x[idx]**2) / r_tf**2
            psi[idx] = np.sqrt(self.params.density * density_profile)
        
        # Normalize to get correct total atom number
        current_norm = np.trapz(np.abs(psi)**2, self.x)
        if current_norm > 0:
            # Scale to get target atom number
            target_atoms = self.params.density * (4/3) * np.pi * r_tf**3  # Approximate volume
            psi *= np.sqrt(target_atoms / current_norm)
        
        return psi

    def evolve(self, dt: float, steps: int, 
              potential: Optional[Callable] = None) -> np.ndarray:
        """
        Evolve BEC state using split-step Fourier method
        
        Args:
            dt: Time step (s)
            steps: Number of time steps
            potential: External potential function V(x)
            
        Returns:
            Final wavefunction
        """
        # Kinetic operator in Fourier space
        k = 2 * np.pi * fftfreq(len(self.x), self.dx)
        kinetic_op = np.exp(-0.5j * const.hbar * k**2 * dt / self.params.atom_mass)
        
        # Trap potential
        V_trap = 0.5 * self.params.atom_mass * (2 * np.pi * self.params.trap_frequency)**2 * self.x**2
        
        # Nonlinear coefficient
        g = 4 * np.pi * const.hbar**2 * self.params.scattering_length / self.params.atom_mass
        
        for _ in range(steps):
            # Half-step: Potential energy
            V = V_trap
            if potential:
                V += potential(self.x)
            self.state *= np.exp(-0.5j * V * dt / const.hbar)
            
            # Full-step: Kinetic energy
            psi_k = np.fft.fft(self.state)
            psi_k *= kinetic_op
            self.state = np.fft.ifft(psi_k)
            
            # Nonlinear term
            self.state *= np.exp(-1j * g * np.abs(self.state)**2 * dt / const.hbar)
            
            # Final potential half-step
            self.state *= np.exp(-0.5j * V * dt / const.hbar)
            
        return self.state

    def dm_phase_shift(self, dm_density: float, dm_mass: float, 
                       cross_section: float, exposure_time: float) -> np.ndarray:
        """
        Calculate DM-induced quantum phase shift
        
        Args:
            dm_density: Local dark matter density (kg/m³)
            dm_mass: Dark matter particle mass (kg)
            cross_section: Interaction cross-section (m²)
            exposure_time: Interaction time (s)
            
        Returns:
            Spatial phase shift profile (radians)
        """
        # Interaction strength (dimensionless coupling)
        λ = cross_section * dm_mass / (const.m_p * const.c**2)
        
        # Potential energy from DM interaction
        # V = λ ρ_dm c² |ψ|² (effective field theory approach)
        V_dm = λ * dm_density * const.c**2 * np.abs(self.state)**2
        
        # Phase shift = ∫ V_dm dt / ħ
        return V_dm * exposure_time / const.hbar

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
        rms_shift = np.sqrt(np.trapz(phase_shift**2 * np.abs(self.state)**2, self.x))
        
        # Shot noise limit
        shot_noise = 1 / np.sqrt(max(n_atoms, 1))  # Avoid division by zero
        
        # Signal-to-noise ratio
        snr = rms_shift / shot_noise if shot_noise > 0 else 0
        
        return {
            'rms_phase_shift': rms_shift,
            'shot_noise_limit': shot_noise,
            'snr': snr,
            'detectable': snr > 3,
            'n_atoms': n_atoms
        }

class GalaxyBECSimulator:
    """Galaxy-specific BEC dark matter detection simulator - CORRECTED VERSION"""
    
    def __init__(self, galaxy_params: Dict, bec_type: str = 'rubidium87'):
        """
        Initialize with galaxy parameters - FIXED VERSION
        
        Args:
            galaxy_params: Dictionary of galaxy parameters from data pipeline
            bec_type: Type of BEC to simulate ('rubidium87', 'sodium23', 'optimized')
        """
        self.galaxy_params = galaxy_params
        
        # FIXED: Use realistic BEC parameters based on experimental values
        if bec_type == 'rubidium87':
            # Standard Rb-87 BEC (most common in experiments)
            atom_mass = 1.45e-25      # kg (Rb-87)
            scattering_length = 5.3e-9 # m (Rb-87 s-wave scattering length)
            target_atoms = 1e6        # 1 million atoms (typical)
            coherence_length = 1e-5   # m (10 μm BEC size)
            trap_frequency = 100      # Hz (typical magnetic trap)
            
        elif bec_type == 'sodium23':
            # Na-23 BEC (alternative common choice)
            atom_mass = 3.82e-26      # kg (Na-23)
            scattering_length = 2.75e-9 # m (Na-23 scattering length)
            target_atoms = 5e6        # 5 million atoms
            coherence_length = 1.5e-5 # m (15 μm)
            trap_frequency = 150      # Hz
            
        elif bec_type == 'optimized':
            # Optimized parameters for DM detection
            atom_mass = 1.67e-27      # kg (hydrogen - lightest)
            scattering_length = 1e-8  # m (enhanced scattering)
            target_atoms = 1e9        # 1 billion atoms (ambitious but possible)
            coherence_length = 1e-4   # m (100 μm - large BEC)
            trap_frequency = 50       # Hz (weaker trap for larger size)
            
        else:
            raise ValueError(f"Unknown BEC type: {bec_type}")
        
        # Calculate density from target atom number and coherence volume
        volume = (4/3) * np.pi * (coherence_length/2)**3  # Spherical volume
        density = target_atoms / volume
        
        bec_params = BECParameters(
            atom_mass=atom_mass,
            scattering_length=scattering_length,
            density=density,  # FIXED: Not using DM density!    
            trap_frequency=trap_frequency,
            coherence_length=coherence_length
        )
        
        # Spatial size based on coherence length
        size = 3 * coherence_length  # 3x larger than BEC for numerical stability
        
        # Create BEC simulator
        self.bec = BECSimulator(bec_params, size=size, n_points=1024)
        
        # Print diagnostic info
        print(f"✅ BEC initialized ({bec_type}):")
        print(f"   Atoms: {self.bec.n_atoms:.2e}")
        print(f"   Size: {coherence_length*1e6:.1f} μm")
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

    def parameter_scan(self, dm_model_params: Dict, 
                      exposure_times: list = None,
                      cross_section_multipliers: list = None) -> Dict:
        """
        Perform parameter scan for optimization
        
        Args:
            dm_model_params: Base DM model parameters
            exposure_times: List of exposure times to test (s)
            cross_section_multipliers: Multipliers for base cross-section
            
        Returns:
            Scan results dictionary
        """
        if exposure_times is None:
            exposure_times = [1800, 3600, 7200, 14400, 28800]  # 0.5h to 8h
        
        if cross_section_multipliers is None:
            cross_section_multipliers = [1, 10, 100, 1000, 10000]  # 1x to 10^4x
        
        scan_results = {
            'exposure_times': exposure_times,
            'cross_section_multipliers': cross_section_multipliers,
            'snr_matrix': np.zeros((len(exposure_times), len(cross_section_multipliers))),
            'detectable_matrix': np.zeros((len(exposure_times), len(cross_section_multipliers)), dtype=bool)
        }
        
        base_cross_section = dm_model_params['cross_section']
        dm_mass = dm_model_params['mass']
        
        for i, exp_time in enumerate(exposure_times):
            for j, multiplier in enumerate(cross_section_multipliers):
                cross_section = base_cross_section * multiplier
                
                result = self.simulate_dm_interaction(dm_mass, cross_section, exp_time)
                
                scan_results['snr_matrix'][i, j] = result['snr']
                scan_results['detectable_matrix'][i, j] = result['detectable']
        
        return scan_results

# Enhanced DM models with larger cross-sections for detection studies
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
        'cross_section': 1e-40, # m² (enhanced)
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

def create_enhanced_simulator(galaxy_params: Dict, bec_type: str = 'optimized') -> GalaxyBECSimulator:
    """
    Create BEC simulator optimized for DM detection
    
    Args:
        galaxy_params: Galaxy parameters
        bec_type: BEC configuration ('rubidium87', 'sodium23', 'optimized')
        
    Returns:
        Configured BEC simulator
    """
    return GalaxyBECSimulator(galaxy_params, bec_type=bec_type)

def quick_sensitivity_test(galaxy_params: Dict, dm_models: Dict = None) -> Dict:
    """
    Quick test of detection sensitivity across DM models
    
    Args:
        galaxy_params: Galaxy parameters
        dm_models: Dictionary of DM models to test (default: enhanced models)
        
    Returns:
        Test results
    """
    if dm_models is None:
        dm_models = ENHANCED_DM_MODELS
    
    # Create optimized BEC simulator
    bec_sim = create_enhanced_simulator(galaxy_params, bec_type='optimized')
    
    results = {
        'galaxy': galaxy_params['name'],
        'bec_info': {
            'n_atoms': bec_sim.bec.n_atoms,
            'shot_noise': 1/np.sqrt(max(bec_sim.bec.n_atoms, 1)),
            'coherence_length': bec_sim.bec.params.coherence_length
        },
        'model_results': {}
    }
    
    print(f"\n🧪 Quick sensitivity test for {galaxy_params['name']}")
    print(f"BEC: {bec_sim.bec.n_atoms:.1e} atoms, shot noise: {results['bec_info']['shot_noise']:.2e}")
    print("-" * 60)
    
    for model_name, model_params in dm_models.items():
        sim_result = bec_sim.simulate_dm_interaction(
            model_params['mass'],
            model_params['cross_section'],
            3600  # 1 hour exposure
        )
        
        results['model_results'][model_name] = {
            'snr': sim_result['snr'],
            'detectable': sim_result['detectable'],
            'phase_shift_rms': sim_result['rms_phase_shift']
        }
        
        status = '✅ DETECTABLE' if sim_result['detectable'] else '❌ Too weak'
        print(f"{model_name:25s}: SNR = {sim_result['snr']:.3f} {status}")
    
    # Summary statistics
    detectable_count = sum(1 for r in results['model_results'].values() if r['detectable'])
    total_count = len(results['model_results'])
    
    results['summary'] = {
        'detectable_fraction': detectable_count / total_count,
        'best_snr': max(r['snr'] for r in results['model_results'].values()),
        'detection_rate': f"{detectable_count}/{total_count}"
    }
    
    print("-" * 60)
    print(f"Detection rate: {results['summary']['detection_rate']} ({results['summary']['detectable_fraction']:.1%})")
    print(f"Best SNR: {results['summary']['best_snr']:.3f}")
    
    return results