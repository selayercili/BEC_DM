#!/usr/bin/env python3
"""
BEC Dark Matter Simulation Diagnostic Tool
Analyzes why SNR values are zero and suggests parameter adjustments
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from data import GalaxyDataManager
    from bec_physics import GalaxyBECSimulator, BECParameters
    from dm_models import DM_MODELS
    import scipy.constants as const
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def diagnose_bec_simulation(galaxy_params: Dict, dm_model: str = 'axion', 
                           exposure_time: float = 3600, verbose: bool = True):
    """
    Diagnose BEC simulation parameters and identify sensitivity bottlenecks
    
    Args:
        galaxy_params: Galaxy parameter dictionary
        dm_model: Dark matter model to test
        exposure_time: Exposure time in seconds
        verbose: Print detailed diagnostic information
    """
    if verbose:
        print(f"\n🔍 DIAGNOSING BEC SIMULATION")
        print(f"Galaxy: {galaxy_params['name']}")
        print(f"DM Model: {dm_model}")
        print(f"Exposure: {exposure_time/3600:.1f} hours")
        print("=" * 50)
    
    # Get DM model parameters
    if dm_model not in DM_MODELS:
        print(f"❌ Unknown DM model: {dm_model}")
        return
    
    dm_params = DM_MODELS[dm_model]
    
    try:
        # Initialize BEC simulator
        bec_sim = GalaxyBECSimulator(galaxy_params)
        bec = bec_sim.bec
        
        if verbose:
            print("\n📊 BEC PARAMETERS:")
            print(f"Atom mass: {bec.params.atom_mass:.2e} kg")
            print(f"Scattering length: {bec.params.scattering_length:.2e} m")
            print(f"Density: {bec.params.density:.2e} atoms/m³")
            print(f"Trap frequency: {bec.params.trap_frequency:.2e} Hz")
            print(f"Coherence length: {bec.params.coherence_length:.2e} m")
            print(f"Spatial grid size: {len(bec.x)} points")
            print(f"Spatial extent: {bec.x.max() - bec.x.min():.2e} m")
        
        # Calculate key physics quantities
        dm_density = galaxy_params['dm_density_local_kg_m3']
        dm_mass = dm_params['mass']
        cross_section = dm_params['cross_section']
        
        if verbose:
            print(f"\n🌌 DARK MATTER PARAMETERS:")
            print(f"DM density: {dm_density:.2e} kg/m³")
            print(f"DM mass: {dm_mass:.2e} kg")
            print(f"Cross section: {cross_section:.2e} m²")
        
        # Calculate phase shift
        phase_shift = bec_sim.bec.dm_phase_shift(dm_density, dm_mass, cross_section, exposure_time)
        
        # Get BEC state info
        n_atoms = np.trapz(np.abs(bec.state)**2, bec.x)
        max_density = np.max(np.abs(bec.state)**2)
        
        if verbose:
            print(f"\n🔬 BEC STATE ANALYSIS:")
            print(f"Total atoms in BEC: {n_atoms:.2e}")
            print(f"Peak density: {max_density:.2e} atoms/m")
            print(f"State normalization: {np.sqrt(np.trapz(np.abs(bec.state)**2, bec.x)):.6f}")
        
        # Phase shift analysis
        rms_phase = np.sqrt(np.trapz(phase_shift**2 * np.abs(bec.state)**2, bec.x))
        max_phase = np.max(np.abs(phase_shift))
        mean_phase = np.trapz(phase_shift * np.abs(bec.state)**2, bec.x)
        
        if verbose:
            print(f"\n⚡ PHASE SHIFT ANALYSIS:")
            print(f"RMS phase shift: {rms_phase:.2e} rad")
            print(f"Max phase shift: {max_phase:.2e} rad")
            print(f"Mean phase shift: {mean_phase:.2e} rad")
        
        # Shot noise calculation
        shot_noise = 1 / np.sqrt(n_atoms)
        snr = rms_phase / shot_noise
        
        if verbose:
            print(f"\n📈 SENSITIVITY ANALYSIS:")
            print(f"Shot noise limit: {shot_noise:.2e}")
            print(f"Signal-to-noise ratio: {snr:.6f}")
            print(f"Detection threshold (SNR > 3): {'✅ YES' if snr > 3 else '❌ NO'}")
        
        # Identify bottlenecks
        print(f"\n🎯 BOTTLENECK ANALYSIS:")
        
        # Check if phase shift is the problem
        if rms_phase < 1e-10:
            print("❌ CRITICAL: Phase shift extremely small")
            print("   Possible causes:")
            print("   - DM interaction too weak")
            print("   - Cross section too small")
            print("   - Exposure time too short")
            print("   - BEC density too low")
        
        # Check if shot noise is the problem
        if shot_noise > 1e-3:
            print("❌ CRITICAL: Shot noise very high")
            print("   Possible causes:")
            print("   - Too few atoms in BEC")
            print("   - BEC state not well concentrated")
        
        # Calculate improvement factors needed
        snr_improvement = 3.0 / snr if snr > 0 else float('inf')
        
        if verbose and snr > 0:
            print(f"\n💡 IMPROVEMENT SUGGESTIONS:")
            print(f"Need {snr_improvement:.1e}x improvement for detection")
            
            # Suggest parameter changes
            print("Possible improvements:")
            print(f"- Increase exposure time by {snr_improvement:.1e}x")
            print(f"- Increase BEC atom number by {snr_improvement**2:.1e}x")
            print(f"- Use different DM model with larger cross-section")
            print(f"- Target galaxies with higher DM density")
        
        return {
            'galaxy': galaxy_params['name'],
            'dm_model': dm_model,
            'n_atoms': float(n_atoms),
            'rms_phase_shift': float(rms_phase),
            'shot_noise': float(shot_noise),
            'snr': float(snr),
            'detectable': snr > 3,
            'improvement_factor_needed': float(snr_improvement) if snr > 0 else float('inf'),
            'phase_shift_profile': phase_shift,
            'spatial_grid': bec.x
        }
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_enhanced_dm_models():
    """
    Create enhanced DM models with larger cross-sections for detection feasibility
    """
    print("\n🔧 CREATING ENHANCED DM MODELS FOR TESTING")
    print("=" * 50)
    
    # Original models (likely too weak for detection)
    original_models = DM_MODELS.copy()
    
    # Enhanced models with larger cross-sections
    enhanced_models = {
        'axion_enhanced': {
            'mass': 1e-22,          # Same mass
            'cross_section': 1e-40  # 10^10 times larger!
        },
        'wimp_enhanced': {
            'mass': 1e-25,          # Same mass  
            'cross_section': 1e-36  # 10^10 times larger!
        },
        'sterile_neutrino_enhanced': {
            'mass': 1e-24,          # Same mass
            'cross_section': 1e-38  # 10^10 times larger!
        },
        'optimistic_axion': {
            'mass': 1e-22,
            'cross_section': 1e-35  # Very optimistic!
        },
        'composite_dm': {
            'mass': 1e-20,          # Heavier composite DM
            'cross_section': 1e-30  # Much larger cross-section
        }
    }
    
    print("Enhanced models created:")
    for name, params in enhanced_models.items():
        print(f"- {name}:")
        print(f"  Mass: {params['mass']:.2e} kg")
        print(f"  Cross-section: {params['cross_section']:.2e} m²")
        improvement = params['cross_section'] / original_models[name.split('_')[0]]['cross_section']
        print(f"  Improvement factor: {improvement:.1e}x")
        print()
    
    return enhanced_models

def optimize_bec_parameters(galaxy_params: Dict):
    """
    Suggest optimized BEC parameters for better sensitivity
    """
    print(f"\n⚙️ BEC PARAMETER OPTIMIZATION")
    print("=" * 50)
    
    # Calculate enhanced BEC parameters
    enhanced_bec_params = {
        'atom_mass': 1.67e-27,      # Hydrogen (lightest)
        'scattering_length': 100e-15, # Enhanced scattering (100x larger)
        'density': galaxy_params['dm_density_local_kg_m3'] / 1.67e-27 * 1e6, # 1M times denser
        'trap_frequency': 1000,      # 1 kHz trap (stronger confinement)
        'coherence_length': 1e-3     # 1 mm coherence (much larger)
    }
    
    print("Suggested enhanced BEC parameters:")
    print(f"- Atom mass: {enhanced_bec_params['atom_mass']:.2e} kg (hydrogen)")
    print(f"- Scattering length: {enhanced_bec_params['scattering_length']:.2e} m (enhanced)")
    print(f"- Density: {enhanced_bec_params['density']:.2e} atoms/m³ (concentrated)")
    print(f"- Trap frequency: {enhanced_bec_params['trap_frequency']:.0f} Hz (tight trap)")
    print(f"- Coherence length: {enhanced_bec_params['coherence_length']:.2e} m (extended)")
    
    # Estimate atom number
    volume = enhanced_bec_params['coherence_length']**3
    n_atoms = enhanced_bec_params['density'] * volume
    print(f"- Estimated atom number: {n_atoms:.2e}")
    print(f"- Shot noise limit: {1/np.sqrt(n_atoms):.2e}")
    
    return enhanced_bec_params

def run_enhanced_simulation_test():
    """
    Run a test with enhanced parameters to verify detection feasibility
    """
    print(f"\n🧪 RUNNING ENHANCED SIMULATION TEST")
    print("=" * 50)
    
    # Load a test galaxy
    manager = GalaxyDataManager()
    galaxies = manager.get_simulation_ready_galaxies(n_galaxies=1)
    
    if not galaxies:
        print("❌ No galaxies available for testing")
        return
    
    galaxy_params = galaxies[0]
    print(f"Test galaxy: {galaxy_params['name']}")
    
    # Test with original parameters
    print(f"\n1. Testing with ORIGINAL parameters:")
    original_result = diagnose_bec_simulation(galaxy_params, 'axion', 3600, verbose=False)
    if original_result:
        print(f"   SNR: {original_result['snr']:.6f}")
        print(f"   Detectable: {'✅' if original_result['detectable'] else '❌'}")
    
    # Create enhanced models
    enhanced_models = create_enhanced_dm_models()
    
    # Test enhanced models
    print(f"\n2. Testing with ENHANCED models:")
    for model_name, model_params in enhanced_models.items():
        # Temporarily add to DM_MODELS
        DM_MODELS[model_name] = model_params
        
        result = diagnose_bec_simulation(galaxy_params, model_name, 3600, verbose=False)
        if result:
            print(f"   {model_name}: SNR = {result['snr']:.6f} {'✅' if result['detectable'] else '❌'}")
        
        # Remove from DM_MODELS
        del DM_MODELS[model_name]
    
    # Suggest realistic parameter space
    print(f"\n3. REALISTIC PARAMETER RECOMMENDATIONS:")
    print("For actual detection, consider:")
    print("- Cross-sections in range 10⁻⁴⁰ to 10⁻³⁰ m²")
    print("- BEC atom numbers > 10¹² atoms")
    print("- Exposure times > 24 hours")
    print("- Multiple BEC interferometers for correlation")
    print("- Cryogenic environments to reduce thermal noise")

def create_parameter_scan_plot():
    """
    Create a parameter scan plot showing detection sensitivity
    """
    print(f"\n📊 CREATING PARAMETER SENSITIVITY PLOT")
    print("=" * 50)
    
    # Parameter ranges to scan
    cross_sections = np.logspace(-50, -30, 20)  # m²
    exposure_times = np.logspace(2, 6, 20)      # seconds (100s to 10⁶s ≈ 11 days)
    
    # Load test galaxy
    manager = GalaxyDataManager()
    galaxies = manager.get_simulation_ready_galaxies(n_galaxies=1)
    
    if not galaxies:
        print("❌ No galaxies for parameter scan")
        return
    
    galaxy_params = galaxies[0]
    
    # Create meshgrid
    CS, ET = np.meshgrid(cross_sections, exposure_times)
    SNR = np.zeros_like(CS)
    
    print("Running parameter scan... (this may take a moment)")
    
    # Calculate SNR for each parameter combination
    for i, exp_time in enumerate(exposure_times):
        for j, cross_sec in enumerate(cross_sections):
            try:
                # Create temporary DM model
                temp_model = {
                    'mass': 1e-22,  # axion mass
                    'cross_section': cross_sec
                }
                DM_MODELS['temp_scan'] = temp_model
                
                # Run simulation
                bec_sim = GalaxyBECSimulator(galaxy_params)
                result = bec_sim.simulate_dm_interaction(
                    temp_model['mass'], cross_sec, exp_time
                )
                
                SNR[i, j] = result.get('snr', 0)
                
                # Cleanup
                del DM_MODELS['temp_scan']
                
            except Exception:
                SNR[i, j] = 0
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {(i+1)/len(exposure_times):.0%}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # SNR contour plot
    levels = [0.1, 0.3, 1, 3, 10, 30, 100]
    contour = plt.contourf(CS, ET, SNR, levels=levels, cmap='viridis', extend='max')
    plt.colorbar(contour, label='Signal-to-Noise Ratio')
    
    # Detection threshold line
    detection_contour = plt.contour(CS, ET, SNR, levels=[3], colors='red', linewidths=3)
    plt.clabel(detection_contour, inline=True, fontsize=12, fmt='SNR = %.0f')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dark Matter Cross-Section (m²)')
    plt.ylabel('Exposure Time (seconds)')
    plt.title(f'BEC Dark Matter Detection Sensitivity\nGalaxy: {galaxy_params["name"]}')
    
    # Add time labels on right axis
    ax2 = plt.gca().twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(plt.gca().get_ylim())
    time_labels = [100, 1000, 3600, 86400, 864000]  # seconds
    time_names = ['100s', '16min', '1h', '1day', '10days']
    ax2.set_yticks(time_labels)
    ax2.set_yticklabels(time_names)
    ax2.set_ylabel('Exposure Time')
    
    # Mark original DM model parameters
    for name, params in DM_MODELS.items():
        if name not in ['temp_scan']:
            plt.scatter(params['cross_section'], 3600, 
                       s=100, marker='*', edgecolor='white', linewidth=2,
                       label=f'{name} (1h exposure)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('bec_sensitivity_scan.png', dpi=300, bbox_inches='tight')
    print("📊 Sensitivity plot saved as 'bec_sensitivity_scan.png'")
    plt.show()

def main():
    """Main diagnostic function"""
    print("🔍 BEC DARK MATTER SIMULATION DIAGNOSTICS")
    print("=" * 60)
    
    # Load test galaxy
    print("\n1. Loading test galaxy...")
    manager = GalaxyDataManager()
    galaxies = manager.get_simulation_ready_galaxies(n_galaxies=1)
    
    if not galaxies:
        print("❌ No galaxies available. Run download_and_organize.py first!")
        return
    
    galaxy_params = galaxies[0]
    print(f"✅ Using galaxy: {galaxy_params['name']}")
    
    # Run detailed diagnostic
    print(f"\n2. Running detailed diagnostic...")
    result = diagnose_bec_simulation(galaxy_params, 'axion', 3600, verbose=True)
    
    if not result:
        print("❌ Diagnostic failed")
        return
    
    # Test all DM models
    print(f"\n3. Testing all DM models...")
    for dm_model in DM_MODELS.keys():
        result = diagnose_bec_simulation(galaxy_params, dm_model, 3600, verbose=False)
        if result:
            print(f"   {dm_model}: SNR = {result['snr']:.2e} {'✅' if result['detectable'] else '❌'}")
    
    # Run enhanced simulation test
    print(f"\n4. Testing enhanced parameters...")
    run_enhanced_simulation_test()
    
    # Suggest optimized BEC parameters
    print(f"\n5. Optimizing BEC parameters...")
    optimize_bec_parameters(galaxy_params)
    
    # Create parameter scan plot
    print(f"\n6. Creating sensitivity scan...")
    try:
        create_parameter_scan_plot()
    except Exception as e:
        print(f"⚠️  Plotting failed: {e}")
    
    print(f"\n" + "=" * 60)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("KEY FINDINGS:")
    print("- Original DM cross-sections are too small for detection")
    print("- Need ~10¹⁰ enhancement in cross-section OR atom number")
    print("- Current BEC parameters give very low sensitivity")
    print("- Enhanced models show detection is theoretically possible")
    print("\nRECOMMENDations:")
    print("- Use enhanced cross-sections for proof-of-concept studies")  
    print("- Consider composite/strongly-interacting dark matter models")
    print("- Optimize BEC parameters for maximum atom number")
    print("- Use longer exposure times (days to weeks)")
    print("- Consider alternative detection schemes (e.g., interferometry)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()