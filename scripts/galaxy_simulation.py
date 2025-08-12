#!/usr/bin/env python3
"""
Enhanced Galaxy BEC Dark Matter Simulation Script
Includes realistic parameter ranges and enhanced sensitivity options
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional
import json
from datetime import datetime
import warnings

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from data import GalaxyDataManager
    from bec_physics import GalaxyBECSimulator
    from dm_models import DM_MODELS
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you have the required source files in src/")
    sys.exit(1)

# Enhanced DM models with larger cross-sections for detection studies
ENHANCED_DM_MODELS = {
    'axion_realistic': {
        'mass': 1e-22,          # kg (~10⁻⁶ eV) 
        'cross_section': 1e-42  # m² (enhanced by ~100x)
    },
    'axion_optimistic': {
        'mass': 1e-22,
        'cross_section': 1e-40  # m² (very optimistic)
    },
    'wimp_realistic': {
        'mass': 1e-25,          # kg (~100 GeV)
        'cross_section': 1e-40  # m² (enhanced by ~1M x)
    },
    'wimp_optimistic': {
        'mass': 1e-25,
        'cross_section': 1e-38  # m² (very optimistic)  
    },
    'sterile_neutrino_realistic': {
        'mass': 1e-24,          # kg (~1 keV)
        'cross_section': 1e-42  # m² (enhanced)
    },
    'composite_dark_matter': {
        'mass': 1e-20,          # kg (heavier composite)
        'cross_section': 1e-35  # m² (larger interaction)
    },
    'ultra_light_scalar': {
        'mass': 1e-26,          # kg (ultra-light)
        'cross_section': 1e-38  # m² (coherent enhancement)
    }
}

class EnhancedGalaxySimulationRunner:
    """Enhanced simulation runner with realistic parameter options"""
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results", 
                 use_enhanced_models: bool = True):
        """
        Initialize enhanced simulation runner
        
        Args:
            data_dir: Directory containing galaxy data
            results_dir: Directory to save simulation results  
            use_enhanced_models: Use enhanced DM models for detection feasibility
        """
        self.data_manager = GalaxyDataManager(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "simulations").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "diagnostics").mkdir(exist_ok=True)
        
        # Select DM models to use
        if use_enhanced_models:
            self.dm_models = {**DM_MODELS, **ENHANCED_DM_MODELS}
            print("🔧 Using enhanced DM models for improved detection feasibility")
        else:
            self.dm_models = DM_MODELS
            print("⚠️  Using original DM models (may have very low detection rates)")
        
        self.simulation_results = []
        
    def diagnose_single_simulation(self, galaxy_params: Dict, dm_model: str = 'axion',
                                 exposure_time: float = 3600) -> Dict:
        """
        Run diagnostic analysis on a single simulation
        
        Args:
            galaxy_params: Galaxy parameter dictionary
            dm_model: DM model name
            exposure_time: Exposure time in seconds
            
        Returns:
            Detailed diagnostic results
        """
        if dm_model not in self.dm_models:
            return {'error': f'Unknown DM model: {dm_model}'}
        
        try:
            # Initialize BEC simulator
            bec_sim = GalaxyBECSimulator(galaxy_params)
            dm_params = self.dm_models[dm_model]
            
            # Run simulation
            result = bec_sim.simulate_dm_interaction(
                dm_params['mass'],
                dm_params['cross_section'], 
                exposure_time
            )
            
            # Add diagnostic information
            bec = bec_sim.bec
            n_atoms = np.trapz(np.abs(bec.state)**2, bec.x)
            
            diagnostic_info = {
                'n_atoms': float(n_atoms),
                'bec_density_peak': float(np.max(np.abs(bec.state)**2)),
                'spatial_extent': float(bec.x.max() - bec.x.min()),
                'dm_density': galaxy_params['dm_density_local_kg_m3'],
                'dm_mass': dm_params['mass'],
                'cross_section': dm_params['cross_section'],
                'phase_shift_max': float(np.max(np.abs(result['phase_shift_profile']))),
                'improvement_factor_needed': 3.0 / result['snr'] if result['snr'] > 0 else float('inf')
            }
            
            result.update(diagnostic_info)
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_parameter_study(self, galaxy_params: Dict, 
                          cross_section_range: tuple = (1e-50, 1e-30),
                          exposure_time_range: tuple = (600, 86400)) -> Dict:
        """
        Run parameter study across cross-section and exposure time ranges
        
        Args:
            galaxy_params: Galaxy parameters
            cross_section_range: (min, max) cross-sections in m²
            exposure_time_range: (min, max) exposure times in seconds
            
        Returns:
            Parameter study results
        """
        print(f"🔬 Running parameter study for {galaxy_params['name']}...")
        
        # Create parameter grids
        n_points = 15  # Reduced for speed
        cross_sections = np.logspace(np.log10(cross_section_range[0]), 
                                   np.log10(cross_section_range[1]), n_points)
        exposure_times = np.logspace(np.log10(exposure_time_range[0]),
                                   np.log10(exposure_time_range[1]), n_points)
        
        CS, ET = np.meshgrid(cross_sections, exposure_times)
        SNR = np.zeros_like(CS)
        
        # Fixed DM mass (axion-like)
        dm_mass = 1e-22  # kg
        
        print(f"  Testing {n_points}×{n_points} = {n_points**2} parameter combinations...")
        
        total_combinations = len(exposure_times) * len(cross_sections)
        completed = 0
        
        for i, exp_time in enumerate(exposure_times):
            for j, cross_sec in enumerate(cross_sections):
                try:
                    bec_sim = GalaxyBECSimulator(galaxy_params)
                    result = bec_sim.simulate_dm_interaction(dm_mass, cross_sec, exp_time)
                    SNR[i, j] = result.get('snr', 0)
                except Exception:
                    SNR[i, j] = 0
                
                completed += 1
                if completed % 50 == 0:
                    progress = (completed / total_combinations) * 100
                    print(f"    Progress: {progress:.1f}%")
        
        return {
            'galaxy': galaxy_params['name'],
            'cross_sections': cross_sections,
            'exposure_times': exposure_times, 
            'snr_grid': SNR,
            'detectable_fraction': np.sum(SNR > 3) / SNR.size
        }
    
    def create_sensitivity_plots(self, param_study_results: List[Dict]):
        """
        Create comprehensive sensitivity plots
        
        Args:
            param_study_results: List of parameter study results for different galaxies
        """
        print("📊 Creating sensitivity plots...")
        
        n_galaxies = len(param_study_results)
        fig, axes = plt.subplots(2, max(2, (n_galaxies + 1) // 2), 
                                figsize=(5 * max(2, (n_galaxies + 1) // 2), 10))
        axes = axes.flatten()
        
        for i, result in enumerate(param_study_results):
            ax = axes[i]
            
            CS, ET = np.meshgrid(result['cross_sections'], result['exposure_times'])
            SNR = result['snr_grid']
            
            # Create contour plot
            levels = [0.1, 0.3, 1, 3, 10, 30, 100]
            contour = ax.contourf(CS, ET, SNR, levels=levels, cmap='viridis', extend='max')
            
            # Detection threshold line
            detection_contour = ax.contour(CS, ET, SNR, levels=[3], colors='red', linewidths=2)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Cross-Section (m²)')
            ax.set_ylabel('Exposure Time (s)')
            ax.set_title(f'{result["galaxy"]}\n'
                        f'Detectable: {result["detectable_fraction"]:.1%}')
            ax.grid(True, alpha=0.3)
            
            # Mark original DM models
            for name, params in DM_MODELS.items():
                ax.scatter(params['cross_section'], 3600, 
                          s=30, marker='o', color='white', edgecolor='black')
        
        # Add colorbar
        plt.colorbar(contour, ax=axes, label='Signal-to-Noise Ratio', shrink=0.8)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / "plots" / f"sensitivity_study_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Sensitivity plots saved to: {plot_path}")
        plt.show()
    
    def run_quick_feasibility_study(self, n_galaxies: int = 3) -> Dict:
        """
        Run a quick feasibility study to identify best parameters
        
        Args:
            n_galaxies: Number of galaxies to test
            
        Returns:
            Feasibility study results
        """
        print("🚀 RUNNING QUICK FEASIBILITY STUDY")
        print("=" * 50)
        
        # Load galaxies
        galaxies = self.data_manager.get_simulation_ready_galaxies(n_galaxies)
        if not galaxies:
            print("❌ No galaxies available")
            return {}
        
        feasibility_results = {
            'galaxies_tested': len(galaxies),
            'dm_models_tested': list(self.dm_models.keys()),
            'galaxy_results': {},
            'best_combinations': [],
            'summary': {}
        }
        
        # Test each galaxy with each DM model (quick test)
        exposure_time = 3600  # 1 hour
        
        for galaxy_params in galaxies:
            galaxy_name = galaxy_params['name']
            print(f"\n🌌 Testing galaxy: {galaxy_name}")
            
            galaxy_results = {}
            
            for dm_model in self.dm_models.keys():
                print(f"   🔬 {dm_model}...", end=" ")
                
                result = self.diagnose_single_simulation(galaxy_params, dm_model, exposure_time)
                
                if 'error' in result:
                    print("❌")
                    galaxy_results[dm_model] = {'error': result['error']}
                    continue
                
                snr = result.get('snr', 0)
                detectable = result.get('detectable', False)
                
                print(f"SNR: {snr:.2e} {'✅' if detectable else '❌'}")
                
                galaxy_results[dm_model] = {
                    'snr': snr,
                    'detectable': detectable,
                    'improvement_needed': result.get('improvement_factor_needed', float('inf'))
                }
                
                # Track best combinations
                if detectable:
                    feasibility_results['best_combinations'].append({
                        'galaxy': galaxy_name,
                        'dm_model': dm_model,
                        'snr': snr,
                        'cross_section': self.dm_models[dm_model]['cross_section'],
                        'exposure_time': exposure_time
                    })
            
            feasibility_results['galaxy_results'][galaxy_name] = galaxy_results
        
        # Analyze results
        total_tests = len(galaxies) * len(self.dm_models)
        successful_detections = len(feasibility_results['best_combinations'])
        
        feasibility_results['summary'] = {
            'total_tests': total_tests,
            'successful_detections': successful_detections,
            'success_rate': successful_detections / total_tests if total_tests > 0 else 0,
            'best_dm_models': self._find_best_dm_models(feasibility_results['galaxy_results']),
            'best_galaxies': self._find_best_galaxies(feasibility_results['galaxy_results'])
        }
        
        # Print summary
        print(f"\n📊 FEASIBILITY STUDY SUMMARY:")
        print(f"Success rate: {feasibility_results['summary']['success_rate']:.1%} "
              f"({successful_detections}/{total_tests})")
        
        if feasibility_results['summary']['best_dm_models']:
            print(f"Best DM models: {', '.join(feasibility_results['summary']['best_dm_models'])}")
        
        if feasibility_results['summary']['best_galaxies']:
            print(f"Best galaxies: {', '.join(feasibility_results['summary']['best_galaxies'])}")
        
        return feasibility_results
    
    def _find_best_dm_models(self, galaxy_results: Dict) -> List[str]:
        """Find DM models with highest detection rates"""
        model_scores = {}
        
        for galaxy_name, galaxy_data in galaxy_results.items():
            for dm_model, result in galaxy_data.items():
                if 'error' not in result:
                    if dm_model not in model_scores:
                        model_scores[dm_model] = {'detections': 0, 'total': 0}
                    
                    model_scores[dm_model]['total'] += 1
                    if result.get('detectable', False):
                        model_scores[dm_model]['detections'] += 1
        
        # Sort by detection rate
        sorted_models = sorted(model_scores.keys(), 
                             key=lambda x: model_scores[x]['detections'] / max(model_scores[x]['total'], 1),
                             reverse=True)
        
        # Return models with >0 detections
        return [m for m in sorted_models if model_scores[m]['detections'] > 0]
    
    def _find_best_galaxies(self, galaxy_results: Dict) -> List[str]:
        """Find galaxies with highest detection rates"""
        galaxy_scores = {}
        
        for galaxy_name, galaxy_data in galaxy_results.items():
            detections = sum(1 for result in galaxy_data.values() 
                           if result.get('detectable', False))
            total = len([r for r in galaxy_data.values() if 'error' not in r])
            
            if total > 0:
                galaxy_scores[galaxy_name] = detections / total
        
        # Sort by detection rate
        sorted_galaxies = sorted(galaxy_scores.keys(),
                               key=lambda x: galaxy_scores[x],
                               reverse=True)
        
        # Return galaxies with >0 detection rate
        return [g for g in sorted_galaxies if galaxy_scores[g] > 0]
    
    def run_targeted_simulation(self, best_combinations: List[Dict], 
                              extended_exposure_times: List[float] = None) -> List[Dict]:
        """
        Run detailed simulation on the best parameter combinations
        
        Args:
            best_combinations: List of promising galaxy/DM model combinations
            extended_exposure_times: List of longer exposure times to test
            
        Returns:
            Detailed simulation results
        """
        if extended_exposure_times is None:
            extended_exposure_times = [3600, 7200, 14400, 28800, 86400]  # 1h to 24h
        
        print(f"🎯 RUNNING TARGETED SIMULATIONS")
        print(f"Testing {len(best_combinations)} combinations with extended exposure times")
        print("=" * 60)
        
        targeted_results = []
        
        for combo in best_combinations:
            galaxy_name = combo['galaxy']
            dm_model = combo['dm_model']
            
            print(f"\n🌟 {galaxy_name} + {dm_model}")
            
            # Load galaxy parameters
            galaxies = self.data_manager.get_simulation_ready_galaxies(10)  # Load more to find this one
            galaxy_params = None
            
            for g in galaxies:
                if g['name'] == galaxy_name:
                    galaxy_params = g
                    break
            
            if galaxy_params is None:
                print(f"   ❌ Galaxy {galaxy_name} not found")
                continue
            
            combo_results = {
                'galaxy': galaxy_name,
                'dm_model': dm_model,
                'exposure_results': {}
            }
            
            # Test different exposure times
            for exp_time in extended_exposure_times:
                print(f"   ⏱️  {exp_time/3600:.1f}h...", end=" ")
                
                result = self.diagnose_single_simulation(galaxy_params, dm_model, exp_time)
                
                if 'error' in result:
                    print("❌")
                    combo_results['exposure_results'][exp_time] = result
                    continue
                
                snr = result.get('snr', 0)
                detectable = result.get('detectable', False)
                
                print(f"SNR: {snr:.3f} {'✅' if detectable else '❌'}")
                combo_results['exposure_results'][exp_time] = result
            
            targeted_results.append(combo_results)
        
        return targeted_results
    
    def create_summary_report(self, feasibility_results: Dict, 
                            targeted_results: List[Dict] = None) -> str:
        """Create comprehensive summary report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
ENHANCED BEC DARK MATTER DETECTION STUDY
Generated: {timestamp}
{'=' * 70}

FEASIBILITY STUDY RESULTS:
- Galaxies tested: {feasibility_results.get('galaxies_tested', 0)}
- DM models tested: {len(feasibility_results.get('dm_models_tested', []))}
- Total parameter combinations: {feasibility_results.get('summary', {}).get('total_tests', 0)}
- Successful detections: {feasibility_results.get('summary', {}).get('successful_detections', 0)}
- Overall success rate: {feasibility_results.get('summary', {}).get('success_rate', 0):.1%}

DARK MATTER MODELS TESTED:
"""
        
        if 'dm_models_tested' in feasibility_results:
            for dm_model in feasibility_results['dm_models_tested']:
                if dm_model in self.dm_models:
                    params = self.dm_models[dm_model]
                    report += f"- {dm_model}:\n"
                    report += f"  Mass: {params['mass']:.2e} kg\n"
                    report += f"  Cross-section: {params['cross_section']:.2e} m²\n"
                    
                    # Enhanced model indicator
                    if dm_model in ENHANCED_DM_MODELS:
                        original_name = dm_model.split('_')[0]
                        if original_name in DM_MODELS:
                            enhancement = params['cross_section'] / DM_MODELS[original_name]['cross_section']
                            report += f"  Enhancement factor: {enhancement:.1e}x\n"
                    report += "\n"
        
        # Best combinations
        if feasibility_results.get('best_combinations'):
            report += "SUCCESSFUL DETECTION COMBINATIONS:\n"
            for combo in feasibility_results['best_combinations']:
                report += f"- {combo['galaxy']} + {combo['dm_model']}: SNR = {combo['snr']:.3f}\n"
            report += "\n"
        
        # Targeted simulation results
        if targeted_results:
            report += "EXTENDED EXPOSURE TIME STUDY:\n"
            for result in targeted_results:
                report += f"\n{result['galaxy']} + {result['dm_model']}:\n"
                
                for exp_time, exp_result in result.get('exposure_results', {}).items():
                    if 'error' not in exp_result:
                        hours = exp_time / 3600
                        snr = exp_result.get('snr', 0)
                        detectable = '✓' if exp_result.get('detectable', False) else '✗'
                        report += f"  {hours:4.1f}h: SNR = {snr:.4f} {detectable}\n"
        
        # Recommendations
        report += "\nRECOMMENDATIONS:\n"
        
        if feasibility_results.get('summary', {}).get('success_rate', 0) > 0:
            report += "✅ Detection appears feasible with enhanced parameters!\n"
            
            best_models = feasibility_results.get('summary', {}).get('best_dm_models', [])
            if best_models:
                report += f"- Focus on DM models: {', '.join(best_models[:3])}\n"
            
            best_galaxies = feasibility_results.get('summary', {}).get('best_galaxies', [])
            if best_galaxies:
                report += f"- Target galaxies: {', '.join(best_galaxies[:3])}\n"
                
            report += "- Use exposure times ≥ 4 hours for optimal sensitivity\n"
            report += "- Consider multi-detector correlation for noise reduction\n"
            
        else:
            report += "⚠️  No detections with current parameters\n"
            report += "- Consider even larger cross-sections (>10⁻³⁰ m²)\n"
            report += "- Optimize BEC atom number (>10¹² atoms)\n"
            report += "- Use very long exposure times (>24 hours)\n"
            report += "- Explore alternative detection methods\n"
        
        report += f"\n{'=' * 70}\n"
        
        return report

def main():
    """Main enhanced simulation function"""
    print("🚀 ENHANCED BEC DARK MATTER SIMULATION")
    print("=" * 60)
    
    # Initialize enhanced runner
    print("\n1. Initializing enhanced simulation environment...")
    runner = EnhancedGalaxySimulationRunner(use_enhanced_models=True)
    
    print(f"✅ Using {len(runner.dm_models)} DM models:")
    for name in runner.dm_models.keys():
        enhanced_indicator = " (enhanced)" if name in ENHANCED_DM_MODELS else ""
        print(f"   - {name}{enhanced_indicator}")
    
    # Quick feasibility study
    print("\n2. Running quick feasibility study...")
    feasibility_results = runner.run_quick_feasibility_study(n_galaxies=5)
    
    if not feasibility_results:
        print("❌ Feasibility study failed")
        return
    
    # If we found promising combinations, run detailed study
    best_combinations = feasibility_results.get('best_combinations', [])
    
    if best_combinations:
        print(f"\n✅ Found {len(best_combinations)} promising combinations!")
        print("3. Running targeted simulations with extended exposure times...")
        
        targeted_results = runner.run_targeted_simulation(
            best_combinations,
            extended_exposure_times=[1800, 3600, 7200, 14400, 28800]  # 0.5h to 8h
        )
        
        print("✅ Targeted simulations complete")
        
    else:
        print("\n⚠️  No immediately promising combinations found")
        print("3. Running parameter study to identify optimal ranges...")
        
        # Load one galaxy for parameter study
        galaxies = runner.data_manager.get_simulation_ready_galaxies(2)
        if galaxies:
            param_studies = []
            for galaxy_params in galaxies[:2]:  # Limit to 2 galaxies
                study_result = runner.run_parameter_study(
                    galaxy_params,
                    cross_section_range=(1e-45, 1e-30),
                    exposure_time_range=(600, 86400)
                )
                param_studies.append(study_result)
            
            # Create sensitivity plots
            runner.create_sensitivity_plots(param_studies)
        
        targeted_results = None
    
    # Generate comprehensive report
    print("\n4. Generating summary report...")
    report = runner.create_summary_report(feasibility_results, targeted_results)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"enhanced_simulation_report_{timestamp}.txt"
    report_path = runner.results_dir / report_filename
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📊 Report saved to: {report_path}")
    
    # Print key findings
    print("\n" + "=" * 60)
    print("🎯 KEY FINDINGS:")
    print("=" * 60)
    
    success_rate = feasibility_results.get('summary', {}).get('success_rate', 0)
    total_detections = feasibility_results.get('summary', {}).get('successful_detections', 0)
    
    if success_rate > 0:
        print(f"🌟 SUCCESS! Detection rate: {success_rate:.1%} ({total_detections} detections)")
        
        best_models = feasibility_results.get('summary', {}).get('best_dm_models', [])
        if best_models:
            print(f"🏆 Best DM models: {', '.join(best_models[:3])}")
        
        best_galaxies = feasibility_results.get('summary', {}).get('best_galaxies', [])
        if best_galaxies:
            print(f"🎯 Best targets: {', '.join(best_galaxies[:3])}")
            
        print("💡 Enhanced cross-sections enable detection!")
        
    else:
        print("⚠️  No detections with current enhanced parameters")
        print("💡 Consider further parameter optimization")
    
    print(f"\n📁 All results saved in: {runner.results_dir}")
    print("🎉 Enhanced simulation complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()