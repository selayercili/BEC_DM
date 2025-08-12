"""Unified BEC Physics and Galaxy Simulation Module for Dark Matter Detection"""

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
        
    def _ground_state(self) -> np.ndarray:
        """Calculate Thomas-Fermi ground state wavefunction"""
        # Interaction strength
        g = 4 * np.pi * const.hbar**2 * self.params.scattering_length / self.params.atom_mass
        
        # Chemical potential
        omega = 2 * np.pi * self.params.trap_frequency
        r_tf = np.sqrt(2 * const.hbar * np.sqrt(4 * np.pi * self.params.density * self.params.scattering_length) 
                      / (self.params.atom_mass * omega))
        
        # Thomas-Fermi profile
        psi = np.zeros_like(self.x, dtype=complex)
        idx = np.abs(self.x) <= r_tf
        density = (self.params.atom_mass * omega**2 / (2 * g)) * (r_tf**2 - self.x[idx]**2)
        psi[idx] = np.sqrt(density)
        
        # Normalize
        norm = np.sqrt(np.trapz(np.abs(psi)**2, self.x))
        return psi / norm

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
        # Interaction strength (dimensionless)
        λ = cross_section * dm_mass / (const.m_p * const.c**2)
        
        # Potential energy from DM interaction
        # V = λ ρ_dm c² |ψ|² (following effective field theory)
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
        # Number of atoms
        n_atoms = np.trapz(np.abs(self.state)**2, self.x)
        
        # RMS phase shift
        rms_shift = np.sqrt(np.trapz(phase_shift**2 * np.abs(self.state)**2, self.x))
        
        # Shot noise limit
        shot_noise = 1 / np.sqrt(n_atoms)
        
        # Signal-to-noise ratio
        snr = rms_shift / shot_noise
        
        return {
            'rms_phase_shift': rms_shift,
            'shot_noise_limit': shot_noise,
            'snr': snr,
            'detectable': snr > 3
        }

class GalaxyBECSimulator:
    """Galaxy-specific BEC dark matter detection simulator"""
    
    def __init__(self, galaxy_params: Dict):
        """
        Initialize with galaxy parameters
        
        Args:
            galaxy_params: Dictionary of galaxy parameters from data pipeline
        """
        self.galaxy_params = galaxy_params
        
        # Create BEC parameters from galaxy properties
        bec_params = BECParameters(
            atom_mass=1.67e-27,  # Proton mass (hydrogen)
            scattering_length=5e-15,  # Typical s-wave scattering length
            density=galaxy_params['dm_density_local_kg_m3'] / 1.67e-27,
            trap_frequency=galaxy_params['virial_velocity_m_s'] / galaxy_params['scale_length_m'],
            coherence_length=galaxy_params.get('coherence_length_estimate_m', 1e-6)
        )
        
        # Spatial size based on coherence length
        size = min(10 * bec_params.coherence_length, 1e-3)  # Max 1mm
        
        self.bec = BECSimulator(bec_params, size=size)
        
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
            'spatial_grid': self.bec.x
        }
        results.update(sensitivity)
        
        return results
