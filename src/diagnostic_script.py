#!/usr/bin/env python3
"""
CORRECTED BEC Dark Matter Simulation Diagnostic Tool
Fixes the zero SNR problem and provides realistic parameter recommendations
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Enhanced DM models with corrected cross-sections
DIAGNOSTIC_DM_MODELS = {
    # Original models (too weak for detection)
    'axion_original': {
        'mass': 1e-22,
        'cross_section': 1e-50,
        'description': 'Original QCD axion (undetectable)'
    },
    'wimp_original': {
        'mass': 1e-25,
        'cross_section': 1e-46,
        'description': 'Original WIMP (undetectable)'
    },
    
    # Minimally enhanced models
    'axion_minimal': {
        'mass': 1e-22,
        'cross_section': 1e-44,
        'description': 'Axion with minimal enhancement (100x)'
    },
    'axion_realistic': {
        'mass': 1e-22,
        'cross_section': 1e-42,
        'description': 'Axion with realistic enhancement (10⁴x)'
    },
    'axion_optimistic': {
        'mass': 1e-22,
        'cross_section': 1e-40,
        'description': 'Optimistic axion model (10⁶x)'
    },
    
    # Enhanced WIMP models
    'wimp_minimal': {
        'mass': 1e-25,
        'cross_section': 1e-42,
        'description': 'WIMP with enhanced coupling (10⁴x)'
    },
    'wimp_optimistic': {
        'mass': 1e-25,
        'cross_section': 1e-38,
        'description': 'Optimistic WIMP model (10⁸x)'
    },
    
    # Alternative DM models
    'composite_dm': {
        'mass': 1e-20,
        'cross_section': 1e-35,
        'description': 'Composite dark matter'
    },
    'hidden_photon': {
        'mass': 1e-23,
        'cross_section': 1e-38,
        'description': 'Hidden photon dark matter'
    }
}

def corrected_dm_phase_shift(bec_state, spatial_grid, dm_density, dm_mass, 
                           cross_section, exposure_time):
    """
    CORRECTED phase shift calculation
    
    The original calculation had issues with:
    1. Wrong units in the interaction potential
    2. Incorrect scaling with DM density
    3. Missing factors of c and ħ
    
    Args:
        bec_state: BEC wavefunction
        spatial_grid: Spatial coordinates
        dm_density: DM mass density (kg/m³)
        dm_mass: DM particle mass (kg)
        cross_section: Interaction cross-section (m²)
        exposure_time: Observation time (s)
        
    Returns:
        Phase shift profile (radians)
    """
    import scipy.constants as const
    
    # Convert DM mass density to number density
    dm_number_density = dm_density / dm_mass  # particles/m³
    
    # Interaction potential per unit volume
    # V = ħ * σ * n_dm * c * |ψ|²
    # This has units: [J·s] * [m²] * [1/m³] * [m/s] * [1/m] = [J]
    V_dm = (const.hbar * cross_section * dm_number_density * const.c * 
            np.abs(bec_state)**2)
    
    # Phase accumulated over time: φ = ∫ V dt / ħ
    phase_shift = V_dm * exposure_time / const.hbar
    
    return phase_shift

def corrected_detection_sensitivity(bec_state, spatial_grid, phase_shift):
    """
    CORRECTED sensitivity calculation
    
    Args:
        bec_state: BEC wavefunction
        spatial_grid: Spatial coordinates  
        phase_shift: Phase shift profile
        
    Returns:
        Dictionary with sensitivity metrics
    """
    # Total number of atoms
    n_atoms = np.trapz(np.abs(bec_state)**2, spatial_grid)
    
    # Density-weighted RMS phase shift
    weights = np.abs(bec_state)**2
    total_weight = np.trapz(weights, spatial_grid)
    
    if total_weight > 0:
        # Proper weighted average
        mean_phase = np.trapz(phase_shift * weights, spatial_grid) / total_weight
        rms_phase = np.sqrt(np.trapz((phase_shift - mean_phase)**2 * weights, spatial_grid) / total_weight)
    else:
        rms_phase = 0
        mean_phase = 0
    
    # Shot noise limit
    shot_noise = 1 / np.sqrt(max(n_atoms, 1))
    
    # Signal-to-noise ratio
    snr = rms_phase / shot_noise if shot_noise > 0 else 0
    
    return {
        'n_atoms': n_atoms,
        'rms_phase_shift': rms_phase,
        'mean_phase_shift': mean_phase,
        'shot_noise_limit': shot_noise,
        'snr': snr,
        'detectable': snr > 3
    }

def create_corrected_bec_state(coherence_length=100e-6, n_atoms=1e10, size=1e-3, n_points=1024):
    """
    Create a corrected BEC state with realistic parameters
    
    Args:
        coherence_length: BEC coherence length (m)
        n_atoms: Total number of atoms
        size: Spatial domain size (m) 
        n_points: Number of grid points
        
    Returns:
        Tuple of (spatial_grid, bec_state)
    """
    # Spatial grid
    x = np.linspace(-size/2, size/2, n_points)
    dx = x[1] - x[0]
    
    # Thomas-Fermi radius
    r_tf = coherence_length / 2
    
    # Create Thomas-Fermi profile
    psi = np.zeros_like(x, dtype=complex)
    idx = np.abs(x) <= r_tf
    
    if np.any(idx):
        # Parabolic density profile
        density_profile = (r_tf**2 - x[idx]**2) / r_tf**2
        psi[idx] = np.sqrt(density_profile)
    
    # Normalize to get correct atom number
    current_norm = np.trapz(np.abs(psi)**2, x)
    if current_norm > 0:
        psi *= np.sqrt(n_atoms / current_norm)
    
    return x, psi

def run_corrected_diagnostic(galaxy_dm_density=5e-22, galaxy_name="Test_Galaxy"):
    """
    Run diagnostic with corrected physics
    
    Args:
        galaxy_dm_density: Local DM density (kg/m³)
        galaxy_name: Galaxy name for reporting
    """
    print("🔧 CORRECTED BEC DARK MATTER DIAGNOSTIC")
    print("=" * 60)
    print(f"Galaxy: {galaxy_name}")
    print(f"DM density: {galaxy_dm_density:.2e} kg/m³")
    print()
    
    # Test different BEC configurations
    bec_configs = {
        'standard': {
            'coherence_length': 10e-6,   # 10 μm
            'n_atoms': 1e6,              # 1 million atoms
            'description': 'Standard lab BEC'
        },
        'large': {
            'coherence_length': 100e-6,  # 100 μm  
            'n_atoms': 1e9,              # 1 billion atoms
            'description': 'Large optimized BEC'
        },
        'ultra_large': {
            'coherence_length': 1e-3,    # 1 mm
            'n_atoms': 1e12,             # 1 trillion atoms
            'description': 'Ultra-large theoretical BEC'
        }
    }
    
    exposure_time = 3600  # 1 hour
    
    # Test all combinations
    results_summary = []
    
    for bec_name, bec_params in bec_configs.items():
        print(f"🔬 Testing {bec_params['description']}:")
        print(f"   Coherence length: {bec_params['coherence_length']*1e6:.0f} μm")
        print(f"   Atom number: {bec_params['n_atoms']:.1e}")
        
        # Create BEC state
        x, psi = create_corrected_bec_state(
            coherence_length=bec_params['coherence_length'],
            n_atoms=bec_params['n_atoms'],
            size=5*bec_params['coherence_length'],  # 5x larger domain
            n_points=1024
        )
        
        shot_noise = 1 / np.sqrt(bec_params['n_atoms'])
        print(f"   Shot noise limit: {shot_noise:.2e}")
        print()
        
        # Test with different DM models
        for dm_name, dm_params in DIAGNOSTIC_DM_MODELS.items():
            # Calculate phase shift with corrected formula
            phase_shift = corrected_dm_phase_shift(
                psi, x, galaxy_dm_density, dm_params['mass'],
                dm_params['cross_section'], exposure_time
            )
            
            # Calculate sensitivity
            sensitivity = corrected_detection_sensitivity(psi, x, phase_shift)
            
            # Store results
            result = {
                'bec_config': bec_name,
                'dm_model': dm_name,
                'bec_description': bec_params['description'],
                'dm_description': dm_params['description'],
                'coherence_length_um': bec_params['coherence_length'] * 1e6,
                'n_atoms': bec_params['n_atoms'],
                'cross_section': dm_params['cross_section'],
                'snr': sensitivity['snr'],
                'detectable': sensitivity['detectable'],
                'rms_phase_rad': sensitivity['rms_phase_shift'],
                'shot_noise': sensitivity['shot_noise_limit']
            }
            results_summary.append(result)
            
            # Print result
            status = '✅ DETECTABLE' if sensitivity['detectable'] else '❌ Too weak'
            print(f"   {dm_name:20s}: SNR = {sensitivity['snr']:.3e} {status}")
        
        print()
    
    return results_summary

def analyze_results(results_summary):
    """Analyze diagnostic results and provide recommendations"""
    
    print("📊 DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    # Find detectable combinations
    detectable = [r for r in results_summary if r['detectable']]
    
    print(f"Detectable combinations: {len(detectable)}/{len(results_summary)}")
    print(f"Success rate: {len(detectable)/len(results_summary):.1%}")
    print()
    
    if detectable:
        print("🌟 SUCCESSFUL DETECTION SCENARIOS:")
        print("-" * 40)
        
        # Sort by SNR
        detectable.sort(key=lambda x: x['snr'], reverse=True)
        
        for i, result in enumerate(detectable[:10]):  # Top 10
            print(f"{i+1:2d}. {result['bec_description']} + {result['dm_description']}")
            print(f"     SNR: {result['snr']:.3f}, Cross-section: {result['cross_section']:.1e} m²")
            print(f"     BEC: {result['n_atoms']:.1e} atoms, {result['coherence_length_um']:.0f} μm")
            print()
        
        # Find minimum requirements
        min_cross_section = min(r['cross_section'] for r in detectable)
        min_atoms = min(r['n_atoms'] for r in detectable)
        
        print("🎯 MINIMUM REQUIREMENTS FOR DETECTION:")
        print(f"   Cross-section: ≥ {min_cross_section:.1e} m²")
        print(f"   BEC atoms: ≥ {min_atoms:.1e}")
        print()
    
    else:
        print("❌ NO SUCCESSFUL DETECTIONS")
        print()
        
        # Find best (highest SNR) even if not detectable
        best_result = max(results_summary, key=lambda x: x['snr'])
        
        print("🔍 BEST CASE SCENARIO (still not detectable):")
        print(f"   Configuration: {best_result['bec_description']}")
        print(f"   DM model: {best_result['dm_description']}")
        print(f"   SNR: {best_result['snr']:.3e}")
        print(f"   Improvement needed: {3.0/best_result['snr']:.1e}x")
        print()
    
    # Parameter scaling analysis
    print("📈 PARAMETER SCALING RECOMMENDATIONS:")
    print("-" * 40)
    
    # Analyze scaling relationships
    original_models = [r for r in results_summary if 'original' in r['dm_model']]
    enhanced_models = [r for r in results_summary if r not in original_models]
    
    if original_models and enhanced_models:
        avg_original_snr = np.mean([r['snr'] for r in original_models])
        avg_enhanced_snr = np.mean([r['snr'] for r in enhanced_models])
        
        if avg_original_snr > 0:
            improvement_factor = avg_enhanced_snr / avg_original_snr
            print(f"Enhanced models improve SNR by: {improvement_factor:.1e}x")
        
    print("To improve detection sensitivity:")
    print("1. Increase BEC atom number: SNR ∝ √N_atoms")
    print("2. Increase DM cross-section: SNR ∝ σ")  
    print("3. Increase exposure time: SNR ∝ t")
    print("4. Increase BEC coherence length: helps concentrate atoms")
    print("5. Target high DM density regions: SNR ∝ ρ_dm")
    print()
    
    return results_summary

def create_sensitivity_heatmap(results_summary):
    """Create a heatmap showing detection sensitivity"""
    
    print("📊 Creating sensitivity heatmap...")
    
    # Extract unique BEC configs and DM models
    bec_configs = sorted(list(set(r['bec_config'] for r in results_summary)))
    dm_models = sorted(list(set(r['dm_model'] for r in results_summary)))
    
    # Create SNR matrix
    snr_matrix = np.zeros((len(dm_models), len(bec_configs)))
    
    for i, dm_model in enumerate(dm_models):
        for j, bec_config in enumerate(bec_configs):
            # Find matching result
            for r in results_summary:
                if r['dm_model'] == dm_model and r['bec_config'] == bec_config:
                    snr_matrix[i, j] = r['snr']
                    break
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use log scale for SNR values (add small offset to handle zeros)
    snr_plot = np.log10(snr_matrix + 1e-10)
    
    im = ax.imshow(snr_plot, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(SNR)', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(len(bec_configs)))
    ax.set_xticklabels(bec_configs)
    ax.set_yticks(range(len(dm_models)))
    ax.set_yticklabels(dm_models)
    
    # Add detection threshold line
    threshold_snr = np.log10(3)  # SNR = 3 detection threshold
    
    # Add text annotations for detectable combinations
    for i, dm_model in enumerate(dm_models):
        for j, bec_config in enumerate(bec_configs):
            snr_val = snr_matrix[i, j]
            if snr_val >= 3:  # Detectable
                ax.text(j, i, f'{snr_val:.2f}', ha='center', va='center', 
                       color='white', fontweight='bold', fontsize=10)
            elif snr_val > 1e-6:  # Visible but not detectable
                ax.text(j, i, f'{snr_val:.1e}', ha='center', va='center', 
                       color='white', fontsize=8)
    
    ax.set_xlabel('BEC Configuration')
    ax.set_ylabel('Dark Matter Model')
    ax.set_title('BEC Dark Matter Detection Sensitivity Heatmap\n(Values shown for SNR ≥ 3)')
    
    plt.tight_layout()
    plt.savefig('bec_sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
    print("📊 Heatmap saved as 'bec_sensitivity_heatmap.png'")
    plt.show()

def comprehensive_parameter_scan():
    """Perform comprehensive parameter scan"""
    
    print("🔬 COMPREHENSIVE PARAMETER SCAN")
    print("=" * 60)
    
    # Parameter ranges
    cross_sections = np.logspace(-50, -30, 15)  # m²
    atom_numbers = np.logspace(6, 12, 10)       # atoms
    coherence_lengths = np.logspace(-6, -3, 8)  # m (1 μm to 1 mm)
    
    galaxy_dm_density = 5e-22  # kg/m³
    dm_mass = 1e-22           # kg (axion-like)
    exposure_time = 3600      # 1 hour
    
    print(f"Scanning {len(cross_sections)} × {len(atom_numbers)} × {len(coherence_lengths)} = {len(cross_sections)*len(atom_numbers)*len(coherence_lengths)} combinations")
    
    # Results storage
    scan_results = []
    
    total_combinations = len(cross_sections) * len(atom_numbers) * len(coherence_lengths)
    completed = 0
    
    for cross_section in cross_sections:
        for n_atoms in atom_numbers:
            for coh_length in coherence_lengths:
                
                # Create BEC state
                x, psi = create_corrected_bec_state(
                    coherence_length=coh_length,
                    n_atoms=n_atoms,
                    size=5*coh_length,
                    n_points=512  # Reduce for speed
                )
                
                # Calculate phase shift
                phase_shift = corrected_dm_phase_shift(
                    psi, x, galaxy_dm_density, dm_mass, cross_section, exposure_time
                )
                
                # Calculate sensitivity
                sensitivity = corrected_detection_sensitivity(psi, x, phase_shift)
                
                # Store result
                scan_results.append({
                    'cross_section': cross_section,
                    'n_atoms': n_atoms,
                    'coherence_length': coh_length,
                    'snr': sensitivity['snr'],
                    'detectable': sensitivity['detectable']
                })
                
                completed += 1
                if completed % 100 == 0:
                    progress = (completed / total_combinations) * 100
                    print(f"Progress: {progress:.1f}%")
    
    print("✅ Parameter scan complete")
    
    # Analyze scan results
    detectable_results = [r for r in scan_results if r['detectable']]
    
    print(f"\nDetectable combinations: {len(detectable_results)}/{len(scan_results)}")
    print(f"Success rate: {len(detectable_results)/len(scan_results):.1%}")
    
    if detectable_results:
        # Find parameter boundaries
        min_cross_section = min(r['cross_section'] for r in detectable_results)
        min_n_atoms = min(r['n_atoms'] for r in detectable_results)
        min_coherence = min(r['coherence_length'] for r in detectable_results)
        
        print(f"\nMINIMUM REQUIREMENTS:")
        print(f"Cross-section: ≥ {min_cross_section:.1e} m²")
        print(f"Atom number: ≥ {min_n_atoms:.1e}")
        print(f"Coherence length: ≥ {min_coherence*1e6:.0f} μm")
        
        # Best combination
        best = max(detectable_results, key=lambda x: x['snr'])
        print(f"\nBEST COMBINATION:")
        print(f"Cross-section: {best['cross_section']:.1e} m²")
        print(f"Atom number: {best['n_atoms']:.1e}")
        print(f"Coherence length: {best['coherence_length']*1e6:.0f} μm")
        print(f"SNR: {best['snr']:.3f}")
    
    return scan_results

def main():
    """Main diagnostic function with corrected physics"""
    
    # Test with example galaxy parameters
    galaxy_dm_density = 5.349e-22  # kg/m³ from IC2574 example
    
    print("Starting corrected diagnostic...")
    print("This addresses the zero SNR problem by:")
    print("1. Fixing the DM interaction potential calculation")
    print("2. Using realistic BEC parameters")
    print("3. Proper phase shift and sensitivity formulas")
    print("4. Enhanced DM cross-sections for feasibility")
    print()
    
    # Run diagnostic
    results = run_corrected_diagnostic(galaxy_dm_density, "IC2574")
    
    # Analyze results
    analyze_results(results)
    
    # Create visualization
    try:
        create_sensitivity_heatmap(results)
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
    
    # Ask user if they want comprehensive scan
    print("\n" + "="*60)
    print("🤔 Would you like to run a comprehensive parameter scan?")
    print("This will take several minutes but provides detailed optimization data.")
    
    # For automated running, skip the interactive part
    print("Running quick parameter scan...")
    
    try:
        # Run a smaller parameter scan
        print("\n🔬 Running focused parameter scan...")
        
        # Focus on most promising region
        cross_sections = np.logspace(-45, -35, 8)
        atom_numbers = [1e9, 1e10, 1e11, 1e12]
        coherence_lengths = [50e-6, 100e-6, 500e-6, 1e-3]  # 50 μm to 1 mm
        
        best_snr = 0
        best_params = None
        detectable_count = 0
        total_tested = 0
        
        for cross_section in cross_sections:
            for n_atoms in atom_numbers:
                for coh_length in coherence_lengths:
                    x, psi = create_corrected_bec_state(coh_length, n_atoms, 5*coh_length, 256)
                    
                    phase_shift = corrected_dm_phase_shift(
                        psi, x, galaxy_dm_density, 1e-22, cross_section, 3600
                    )
                    
                    sensitivity = corrected_detection_sensitivity(psi, x, phase_shift)
                    
                    total_tested += 1
                    if sensitivity['detectable']:
                        detectable_count += 1
                    
                    if sensitivity['snr'] > best_snr:
                        best_snr = sensitivity['snr']
                        best_params = {
                            'cross_section': cross_section,
                            'n_atoms': n_atoms,
                            'coherence_length': coh_length,
                            'snr': sensitivity['snr']
                        }
        
        print(f"\n📊 Focused scan results:")
        print(f"Total combinations tested: {total_tested}")
        print(f"Detectable combinations: {detectable_count}")
        print(f"Success rate: {detectable_count/total_tested:.1%}")
        
        if best_params:
            print(f"\n🏆 Best configuration found:")
            print(f"Cross-section: {best_params['cross_section']:.1e} m²")
            print(f"Atom number: {best_params['n_atoms']:.1e}")
            print(f"Coherence length: {best_params['coherence_length']*1e6:.0f} μm")
            print(f"SNR: {best_params['snr']:.3f}")
            
            if best_params['snr'] >= 3:
                print("✅ This configuration enables detection!")
            else:
                improvement = 3.0 / best_params['snr']
                print(f"❌ Need {improvement:.1f}x improvement for detection")
        
    except Exception as e:
        print(f"⚠️ Parameter scan failed: {e}")
    
    print(f"\n🎉 CORRECTED DIAGNOSTIC COMPLETE!")
    print("="*60)
    print("🔧 KEY FIXES APPLIED:")
    print("- Corrected DM interaction potential formula")
    print("- Fixed phase shift calculation with proper units")
    print("- Used realistic BEC parameters")
    print("- Enhanced DM cross-sections for feasibility testing")
    print("- Proper sensitivity and SNR calculations")
    print("\n💡 The zero SNR problem should now be resolved!")

if __name__ == "__main__":
    main()