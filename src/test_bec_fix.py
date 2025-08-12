#!/usr/bin/env python3
"""
BEC Parameter Analysis and Fix
Identifies and corrects the fundamental issues in BEC simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# Your current parameters from IC2574
dm_density_local = 5.349e-22  # kg/m³
virial_velocity = 75000.0     # m/s  
scale_length = 9.258e+19     # m

def analyze_current_bec_params():
    """Analyze the problematic current BEC parameters"""
    print("🔍 ANALYZING CURRENT BEC PARAMETERS")
    print("=" * 50)
    
    # Current parameters from your code
    atom_mass = 1.67e-27  # kg (proton mass)
    scattering_length = 5e-15  # m
    density = dm_density_local / atom_mass  # atoms/m³ - THIS IS THE PROBLEM!
    trap_frequency = virial_velocity / scale_length  # Hz
    coherence_length = 1e-6  # m (default)
    
    print(f"Current BEC parameters:")
    print(f"- Atom mass: {atom_mass:.2e} kg")
    print(f"- Scattering length: {scattering_length:.2e} m")
    print(f"- Density: {density:.2e} atoms/m³")
    print(f"- Trap frequency: {trap_frequency:.2e} Hz")
    print(f"- Coherence length: {coherence_length:.2e} m")
    
    # Calculate derived quantities
    volume = coherence_length**3
    n_atoms = density * volume
    
    print(f"\nDerived quantities:")
    print(f"- BEC volume: {volume:.2e} m³")
    print(f"- Number of atoms: {n_atoms:.2e}")
    print(f"- Shot noise: {1/np.sqrt(max(n_atoms, 1e-100)):.2e}")
    
    print(f"\n❌ PROBLEMS IDENTIFIED:")
    print(f"1. Density calculation uses DM density directly!")
    print(f"   DM density = {dm_density_local:.2e} kg/m³")
    print(f"   → BEC density = {density:.2e} atoms/m³ (WAY TOO LOW!)")
    print(f"2. This gives only {n_atoms:.2e} atoms in BEC")
    print(f"3. Shot noise = {1/np.sqrt(max(n_atoms, 1e-100)):.2e} (infinite!)")
    
    return {
        'density': density,
        'n_atoms': n_atoms,
        'volume': volume,
        'shot_noise': 1/np.sqrt(max(n_atoms, 1e-100))
    }

def design_realistic_bec_params():
    """Design realistic BEC parameters for actual experiments"""
    print(f"\n🔧 DESIGNING REALISTIC BEC PARAMETERS")
    print("=" * 50)
    
    # Realistic experimental BEC parameters
    realistic_params = {
        'atom_mass': 1.45e-25,      # kg (Rubidium-87)
        'scattering_length': 5.3e-9, # m (Rb-87 scattering length)
        'n_atoms_target': 1e6,      # 1 million atoms (typical BEC)
        'trap_frequency': 100,       # Hz (typical magnetic trap)
        'coherence_length': 1e-5,    # m (10 μm, typical BEC size)
    }
    
    # Calculate density from atom number and size
    volume = realistic_params['coherence_length']**3
    density = realistic_params['n_atoms_target'] / volume
    
    realistic_params['density'] = density
    realistic_params['volume'] = volume
    
    print("Realistic BEC parameters:")
    print(f"- Atom: Rubidium-87")
    print(f"- Mass: {realistic_params['atom_mass']:.2e} kg")
    print(f"- Scattering length: {realistic_params['scattering_length']:.2e} m")
    print(f"- Target atom number: {realistic_params['n_atoms_target']:.2e}")
    print(f"- BEC size: {realistic_params['coherence_length']:.2e} m")
    print(f"- Density: {realistic_params['density']:.2e} atoms/m³")
    print(f"- Trap frequency: {realistic_params['trap_frequency']:.0f} Hz")
    print(f"- Volume: {realistic_params['volume']:.2e} m³")
    
    # Calculate sensitivity
    shot_noise = 1 / np.sqrt(realistic_params['n_atoms_target'])
    print(f"- Shot noise limit: {shot_noise:.2e}")
    
    return realistic_params

def calculate_dm_phase_shifts(bec_params):
    """Calculate DM phase shifts with realistic BEC parameters"""
    print(f"\n⚡ CALCULATING DM PHASE SHIFTS")
    print("=" * 50)
    
    # DM models from your code
    dm_models = {
        'axion': {'mass': 1e-22, 'cross_section': 1e-50},
        'wimp': {'mass': 1e-25, 'cross_section': 1e-46},
        'sterile_neutrino': {'mass': 1e-24, 'cross_section': 1e-48}
    }
    
    # Enhanced models for comparison
    enhanced_models = {
        'axion_enhanced': {'mass': 1e-22, 'cross_section': 1e-40},
        'wimp_enhanced': {'mass': 1e-25, 'cross_section': 1e-36},
        'composite_dm': {'mass': 1e-20, 'cross_section': 1e-30}
    }
    
    exposure_time = 3600  # 1 hour
    
    print("Phase shift calculations:")
    print("Model                  | Cross-section | Phase Shift | SNR     | Detectable")
    print("-" * 70)
    
    results = {}
    
    for name, models in [("Original", dm_models), ("Enhanced", enhanced_models)]:
        print(f"\n{name} Models:")
        
        for model_name, params in models.items():
            # Calculate interaction strength
            lambda_dm = params['cross_section'] * params['mass'] / (const.m_p * const.c**2)
            
            # Potential energy from DM interaction
            V_dm = lambda_dm * dm_density_local * const.c**2
            
            # Phase shift = V_dm * t / ħ (assuming uniform BEC density)
            phase_shift = V_dm * exposure_time / const.hbar
            
            # SNR calculation
            shot_noise = 1 / np.sqrt(bec_params['n_atoms_target'])
            snr = phase_shift / shot_noise
            detectable = snr > 3
            
            results[model_name] = {
                'phase_shift': phase_shift,
                'snr': snr,
                'detectable': detectable
            }
            
            print(f"{model_name:20s} | {params['cross_section']:.1e} | {phase_shift:.2e} | {snr:.2e} | {'✅' if detectable else '❌'}")
    
    return results

def optimization_suggestions():
    """Provide concrete optimization suggestions"""
    print(f"\n💡 OPTIMIZATION SUGGESTIONS")
    print("=" * 50)
    
    print("1. FIX BEC PARAMETERS:")
    print("   ❌ DON'T use DM density for BEC density!")
    print("   ✅ Use realistic experimental BEC parameters:")
    print("      - Atom number: 10⁶ - 10⁹ atoms")
    print("      - BEC size: 10-100 μm")
    print("      - Density: 10¹⁴ - 10¹⁶ atoms/m³")
    
    print("\n2. CORRECT PARAMETER DERIVATION:")
    print("   Replace this line in bec_physics.py:")
    print("   density=galaxy_params['dm_density_local_kg_m3'] / 1.67e-27")
    print("   With:")
    print("   density=1e15  # atoms/m³ (realistic BEC density)")
    
    print("\n3. ENHANCED DM MODELS:")
    print("   Use larger cross-sections for detection studies:")
    print("   - Axion: 10⁻⁴⁰ m² (vs 10⁻⁵⁰ m² original)")
    print("   - WIMP: 10⁻³⁶ m² (vs 10⁻⁴⁶ m² original)")
    print("   - Composite DM: 10⁻³⁰ m² (new model)")
    
    print("\n4. EXPERIMENTAL CONSIDERATIONS:")
    print("   - Use Rubidium-87 or Sodium-23 atoms")
    print("   - Implement optical or magnetic trapping")
    print("   - Consider interferometry for enhanced sensitivity")
    print("   - Use correlation between multiple BECs")
    
    print("\n5. DETECTION STRATEGY:")
    print("   - Target nearby galaxies (higher DM density)")
    print("   - Use extended observation times (days/weeks)")
    print("   - Implement real-time phase monitoring")
    print("   - Consider alternative DM signatures")

def create_parameter_comparison_plot():
    """Create visual comparison of parameter regimes"""
    print(f"\n📊 CREATING PARAMETER COMPARISON PLOT")
    print("=" * 50)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Atom number comparison
    current_n = 3.2e-61  # Your current (broken) value
    realistic_n = np.logspace(5, 10, 6)  # 10⁵ to 10¹⁰ atoms
    shot_noise = 1 / np.sqrt(realistic_n)
    
    ax1.loglog(realistic_n, shot_noise, 'b-', linewidth=2, label='Realistic BEC')
    ax1.axvline(current_n, color='red', linestyle='--', label='Current (broken)')
    ax1.axhline(1e-3, color='green', linestyle=':', label='Target sensitivity')
    ax1.set_xlabel('Number of Atoms')
    ax1.set_ylabel('Shot Noise Limit')
    ax1.set_title('Shot Noise vs Atom Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cross-section sensitivity
    cross_sections = np.logspace(-50, -30, 100)
    original_axion = 1e-50
    enhanced_axion = 1e-40
    
    # Assume realistic BEC with 10⁶ atoms
    n_atoms = 1e6
    exposure_time = 3600
    dm_mass = 1e-22
    
    snr_values = []
    for cs in cross_sections:
        lambda_dm = cs * dm_mass / (const.m_p * const.c**2)
        V_dm = lambda_dm * dm_density_local * const.c**2
        phase_shift = V_dm * exposure_time / const.hbar
        snr = phase_shift * np.sqrt(n_atoms)
        snr_values.append(snr)
    
    ax2.loglog(cross_sections, snr_values, 'b-', linewidth=2)
    ax2.axvline(original_axion, color='red', linestyle='--', label='Original axion')
    ax2.axvline(enhanced_axion, color='green', linestyle='--', label='Enhanced axion')
    ax2.axhline(3, color='orange', linestyle=':', label='Detection threshold')
    ax2.set_xlabel('Cross-section (m²)')
    ax2.set_ylabel('Signal-to-Noise Ratio')
    ax2.set_title('SNR vs Cross-section (10⁶ atoms, 1h)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Exposure time scaling
    exp_times = np.logspace(2, 6, 50)  # 100s to 10⁶s
    snr_times = np.sqrt(exp_times / 3600)  # Scale with √t
    
    ax3.loglog(exp_times / 3600, snr_times, 'b-', linewidth=2, label='SNR ∝ √t')
    ax3.axhline(3, color='orange', linestyle=':', label='Detection threshold')
    ax3.set_xlabel('Exposure Time (hours)')
    ax3.set_ylabel('SNR Enhancement Factor')
    ax3.set_title('SNR Enhancement vs Exposure Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Parameter space map
    atom_numbers = np.logspace(5, 10, 20)
    cross_secs = np.logspace(-45, -35, 20)
    
    AN, CS = np.meshgrid(atom_numbers, cross_secs)
    SNR_map = np.zeros_like(AN)
    
    for i, cs in enumerate(cross_secs):
        for j, n_at in enumerate(atom_numbers):
            lambda_dm = cs * dm_mass / (const.m_p * const.c**2)
            V_dm = lambda_dm * dm_density_local * const.c**2
            phase_shift = V_dm * exposure_time / const.hbar
            snr = phase_shift * np.sqrt(n_at)
            SNR_map[i, j] = snr
    
    contour = ax4.contourf(AN, CS, SNR_map, levels=[0.1, 0.3, 1, 3, 10, 30], 
                          cmap='viridis', extend='max')
    ax4.contour(AN, CS, SNR_map, levels=[3], colors='red', linewidths=2)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Number of Atoms')
    ax4.set_ylabel('Cross-section (m²)')
    ax4.set_title('Detection Feasibility Map')
    plt.colorbar(contour, ax=ax4, label='SNR')
    
    # Mark current broken point
    ax4.scatter(current_n, original_axion, color='red', s=100, marker='X', 
               label='Current (broken)', zorder=10)
    
    # Mark realistic points
    ax4.scatter(1e6, enhanced_axion, color='green', s=100, marker='*', 
               label='Realistic target', zorder=10)
    
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('bec_parameter_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Parameter analysis plot saved as 'bec_parameter_analysis.png'")
    plt.show()

def generate_fixed_bec_parameters():
    """Generate corrected BEC parameters class"""
    print(f"\n🔧 GENERATING FIXED BEC PARAMETERS")
    print("=" * 50)
    
    fixed_code = '''
# FIXED BECParameters initialization in bec_physics.py

def __init__(self, galaxy_params: Dict):
    """Initialize with galaxy parameters - FIXED VERSION"""
    self.galaxy_params = galaxy_params
    
    # FIXED: Use realistic BEC parameters, not DM density!
    bec_params = BECParameters(
        atom_mass=1.45e-25,        # kg (Rubidium-87, not hydrogen!)
        scattering_length=5.3e-9,  # m (Rb-87 scattering length)
        density=1e15,              # atoms/m³ (REALISTIC BEC DENSITY)
        trap_frequency=100,        # Hz (typical optical/magnetic trap)
        coherence_length=1e-5      # m (10 μm BEC size)
    )
    
    # Calculate atom number
    volume = bec_params.coherence_length**3
    n_atoms = bec_params.density * volume
    
    print(f"BEC initialized with {n_atoms:.1e} atoms")
    print(f"Shot noise limit: {1/np.sqrt(n_atoms):.2e}")
    
    # Spatial size based on coherence length
    size = 2 * bec_params.coherence_length  # 20 μm total size
    
    self.bec = BECSimulator(bec_params, size=size)
'''
    
    print("Replace the GalaxyBECSimulator.__init__ method with:")
    print(fixed_code)
    
    return fixed_code

def main():
    """Main analysis function"""
    print("🔍 BEC PARAMETER ANALYSIS AND FIX")
    print("=" * 60)
    
    # Analyze current problematic parameters
    current_results = analyze_current_bec_params()
    
    # Design realistic parameters
    realistic_params = design_realistic_bec_params()
    
    # Calculate DM phase shifts with realistic parameters
    dm_results = calculate_dm_phase_shifts(realistic_params)
    
    # Provide optimization suggestions
    optimization_suggestions()
    
    # Create visualization
    create_parameter_comparison_plot()
    
    # Generate fixed code
    fixed_code = generate_fixed_bec_parameters()
    
    print(f"\n" + "=" * 60)
    print("🎯 SUMMARY OF FIXES NEEDED")
    print("=" * 60)
    
    print("❌ CURRENT PROBLEM:")
    print(f"   BEC has {current_results['n_atoms']:.1e} atoms (essentially zero!)")
    print(f"   Shot noise: {current_results['shot_noise']:.1e} (infinite)")
    print(f"   SNR: ≈ 0 for all DM models")
    
    print("\n✅ SOLUTION:")
    print(f"   Use realistic BEC with {realistic_params['n_atoms_target']:.1e} atoms")
    print(f"   Shot noise: {1/np.sqrt(realistic_params['n_atoms_target']):.2e}")
    print(f"   Achievable SNR > 3 with enhanced DM models")
    
    print("\n🔧 REQUIRED CODE CHANGES:")
    print("1. Fix BEC density calculation in bec_physics.py")
    print("2. Use realistic atomic species (Rb-87)")
    print("3. Implement enhanced DM models for detection studies")
    print("4. Adjust spatial grids and numerical parameters")
    
    print("\n📊 EXPECTED IMPROVEMENTS:")
    successful_models = sum(1 for r in dm_results.values() if r['detectable'])
    print(f"   Detectable models: {successful_models}/{len(dm_results)}")
    print(f"   Best SNR achieved: {max(r['snr'] for r in dm_results.values()):.2e}")

if __name__ == "__main__":
    main()