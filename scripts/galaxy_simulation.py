#!/usr/bin/env python3
"""
Galaxy BEC Dark Matter Simulation Script
Orchestrates the full simulation pipeline for detecting dark matter using astrophysical BECs
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

class GalaxySimulationRunner:
    """Main simulation runner for BEC dark matter detection"""
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        """
        Initialize simulation runner
        
        Args:
            data_dir: Directory containing galaxy data
            results_dir: Directory to save simulation results
        """
        self.data_manager = GalaxyDataManager(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (self.results_dir / "simulations").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "sensitivity_maps").mkdir(exist_ok=True)
        
        self.simulation_results = []
        
    def load_galaxies(self, n_galaxies: int = 5) -> List[Dict]:
        """
        Load galaxies for simulation
        
        Args:
            n_galaxies: Number of galaxies to simulate
            
        Returns:
            List of galaxy parameter dictionaries
        """
        print(f"Loading {n_galaxies} galaxies for simulation...")
        
        # Try to load existing processed galaxies first
        try:
            galaxies = self.data_manager.get_simulation_ready_galaxies(n_galaxies)
            if galaxies:
                print(f"✅ Loaded {len(galaxies)} pre-processed galaxies")
                return galaxies
        except Exception as e:
            print(f"⚠️  Could not load pre-processed galaxies: {e}")
        
        # If no pre-processed galaxies, create them
        print("Creating galaxy parameter set...")
        catalog = self.data_manager.load_galaxy_catalog()
        
        if catalog.empty:
            print("❌ No galaxy catalog available")
            return []
        
        selected = self.data_manager.select_representative_galaxies(n_galaxies)
        galaxies = []
        
        for _, row in selected.iterrows():
            params = self.data_manager.extract_galaxy_parameters(row['Galaxy'])
            if params:
                galaxies.append(params)
                
        return galaxies
    
    def run_single_galaxy_simulation(self, galaxy_params: Dict, 
                                   dm_models: Optional[List[str]] = None,
                                   exposure_times: Optional[List[float]] = None) -> Dict:
        """
        Run BEC simulation for a single galaxy across different DM models
        
        Args:
            galaxy_params: Galaxy parameter dictionary
            dm_models: List of DM model names to test
            exposure_times: List of exposure times to test (seconds)
            
        Returns:
            Dictionary with simulation results
        """
        if dm_models is None:
            dm_models = list(DM_MODELS.keys())
        
        if exposure_times is None:
            exposure_times = [3600, 7200, 14400]  # 1h, 2h, 4h
        
        galaxy_name = galaxy_params['name']
        print(f"\n🌌 Simulating galaxy: {galaxy_name}")
        print(f"   Distance: {galaxy_params['distance_Mpc']:.1f} Mpc")
        print(f"   Rotation velocity: {galaxy_params['v_flat_km_s']:.0f} km/s")
        print(f"   DM density: {galaxy_params['dm_density_local_kg_m3']:.2e} kg/m³")
        
        # Initialize galaxy BEC simulator
        try:
            bec_sim = GalaxyBECSimulator(galaxy_params)
        except Exception as e:
            print(f"   ❌ Failed to initialize BEC simulator: {e}")
            return {'galaxy': galaxy_name, 'error': str(e)}
        
        galaxy_results = {
            'galaxy': galaxy_name,
            'galaxy_params': galaxy_params,
            'dm_model_results': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Test each DM model
        for dm_model in dm_models:
            print(f"   🔬 Testing DM model: {dm_model}")
            
            if dm_model not in DM_MODELS:
                print(f"      ⚠️  Unknown DM model: {dm_model}")
                continue
            
            dm_params = DM_MODELS[dm_model]
            model_results = {
                'dm_mass': dm_params['mass'],
                'cross_section': dm_params['cross_section'],
                'exposure_results': {}
            }
            
            # Test different exposure times
            for exp_time in exposure_times:
                print(f"      ⏱️  Exposure time: {exp_time/3600:.1f} hours")
                
                try:
                    # Run the simulation
                    result = bec_sim.simulate_dm_interaction(
                        dm_mass=dm_params['mass'],
                        cross_section=dm_params['cross_section'],
                        exposure_time=exp_time
                    )
                    
                    model_results['exposure_results'][exp_time] = result
                    
                    # Print key results
                    if result.get('detectable', False):
                        print(f"         ✅ DETECTABLE! SNR: {result.get('snr', 0):.2f}")
                    else:
                        print(f"         ❌ Below threshold, SNR: {result.get('snr', 0):.2f}")
                        
                except Exception as e:
                    print(f"         ⚠️  Simulation error: {e}")
                    model_results['exposure_results'][exp_time] = {'error': str(e)}
            
            galaxy_results['dm_model_results'][dm_model] = model_results
        
        return galaxy_results
    
    def run_full_simulation_suite(self, galaxies: Optional[List[Dict]] = None,
                                dm_models: Optional[List[str]] = None,
                                exposure_times: Optional[List[float]] = None,
                                save_results: bool = True) -> List[Dict]:
        """
        Run complete simulation suite across multiple galaxies and DM models
        
        Args:
            galaxies: List of galaxy parameter dictionaries
            dm_models: List of DM model names to test
            exposure_times: List of exposure times (seconds)
            save_results: Whether to save results to files
            
        Returns:
            List of simulation results
        """
        print("=" * 60)
        print("🚀 STARTING FULL BEC DARK MATTER SIMULATION SUITE")
        print("=" * 60)
        
        # Load galaxies if not provided
        if galaxies is None:
            galaxies = self.load_galaxies()
        
        if not galaxies:
            print("❌ No galaxies available for simulation")
            return []
        
        # Set defaults
        if dm_models is None:
            dm_models = list(DM_MODELS.keys())
        
        if exposure_times is None:
            exposure_times = [1800, 3600, 7200, 14400]  # 30min, 1h, 2h, 4h
        
        print(f"📊 Simulation parameters:")
        print(f"   Galaxies: {len(galaxies)}")
        print(f"   DM models: {dm_models}")
        print(f"   Exposure times: {[t/3600 for t in exposure_times]} hours")
        
        # Run simulations
        all_results = []
        total_sims = len(galaxies) * len(dm_models) * len(exposure_times)
        completed_sims = 0
        
        for i, galaxy_params in enumerate(galaxies):
            print(f"\n📍 Galaxy {i+1}/{len(galaxies)}")
            
            try:
                galaxy_result = self.run_single_galaxy_simulation(
                    galaxy_params, dm_models, exposure_times
                )
                all_results.append(galaxy_result)
                
                # Update progress
                galaxy_sims = len(dm_models) * len(exposure_times)
                completed_sims += galaxy_sims
                progress = (completed_sims / total_sims) * 100
                print(f"   📈 Progress: {progress:.1f}% ({completed_sims}/{total_sims})")
                
            except Exception as e:
                print(f"   ❌ Galaxy simulation failed: {e}")
                all_results.append({
                    'galaxy': galaxy_params.get('name', 'Unknown'),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save results
        if save_results and all_results:
            self._save_simulation_results(all_results)
        
        self.simulation_results = all_results
        return all_results
    
    def _save_simulation_results(self, results: List[Dict]):
        """Save simulation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bec_dm_simulation_{timestamp}.json"
        filepath = self.results_dir / "simulations" / filename
        
        # Convert numpy types for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to: {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def analyze_detection_sensitivity(self, results: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze detection sensitivity across all simulations
        
        Args:
            results: Simulation results (uses stored results if None)
            
        Returns:
            Dictionary with sensitivity analysis
        """
        if results is None:
            results = self.simulation_results
        
        if not results:
            print("No simulation results available for analysis")
            return {}
        
        print("\n📊 ANALYZING DETECTION SENSITIVITY")
        print("=" * 40)
        
        sensitivity_analysis = {
            'total_galaxies': 0,
            'total_simulations': 0,
            'detectable_simulations': 0,
            'dm_model_performance': {},
            'galaxy_performance': {},
            'exposure_time_analysis': {},
            'best_targets': []
        }
        
        detection_data = []
        
        for galaxy_result in results:
            if 'error' in galaxy_result:
                continue
                
            galaxy_name = galaxy_result['galaxy']
            sensitivity_analysis['total_galaxies'] += 1
            sensitivity_analysis['galaxy_performance'][galaxy_name] = {
                'detections': 0,
                'total_tests': 0,
                'best_snr': 0
            }
            
            for dm_model, dm_results in galaxy_result.get('dm_model_results', {}).items():
                if dm_model not in sensitivity_analysis['dm_model_performance']:
                    sensitivity_analysis['dm_model_performance'][dm_model] = {
                        'detections': 0,
                        'total_tests': 0,
                        'detection_rate': 0
                    }
                
                for exp_time, exp_result in dm_results.get('exposure_results', {}).items():
                    if 'error' in exp_result:
                        continue
                    
                    sensitivity_analysis['total_simulations'] += 1
                    sensitivity_analysis['galaxy_performance'][galaxy_name]['total_tests'] += 1
                    sensitivity_analysis['dm_model_performance'][dm_model]['total_tests'] += 1
                    
                    # Track exposure time performance
                    if exp_time not in sensitivity_analysis['exposure_time_analysis']:
                        sensitivity_analysis['exposure_time_analysis'][exp_time] = {
                            'detections': 0,
                            'total_tests': 0
                        }
                    sensitivity_analysis['exposure_time_analysis'][exp_time]['total_tests'] += 1
                    
                    # Check if detectable
                    is_detectable = exp_result.get('detectable', False)
                    snr = exp_result.get('snr', 0)
                    
                    if is_detectable:
                        sensitivity_analysis['detectable_simulations'] += 1
                        sensitivity_analysis['galaxy_performance'][galaxy_name]['detections'] += 1
                        sensitivity_analysis['dm_model_performance'][dm_model]['detections'] += 1
                        sensitivity_analysis['exposure_time_analysis'][exp_time]['detections'] += 1
                    
                    # Track best SNR for galaxy
                    if snr > sensitivity_analysis['galaxy_performance'][galaxy_name]['best_snr']:
                        sensitivity_analysis['galaxy_performance'][galaxy_name]['best_snr'] = snr
                    
                    # Collect data for further analysis
                    detection_data.append({
                        'galaxy': galaxy_name,
                        'dm_model': dm_model,
                        'exposure_time': exp_time,
                        'snr': snr,
                        'detectable': is_detectable,
                        'rms_phase_shift': exp_result.get('rms_phase_shift', 0)
                    })
        
        # Calculate detection rates
        for dm_model in sensitivity_analysis['dm_model_performance']:
            model_data = sensitivity_analysis['dm_model_performance'][dm_model]
            if model_data['total_tests'] > 0:
                model_data['detection_rate'] = model_data['detections'] / model_data['total_tests']
        
        for exp_time in sensitivity_analysis['exposure_time_analysis']:
            exp_data = sensitivity_analysis['exposure_time_analysis'][exp_time]
            if exp_data['total_tests'] > 0:
                exp_data['detection_rate'] = exp_data['detections'] / exp_data['total_tests']
        
        # Find best targets
        best_targets = []
        for galaxy_name, galaxy_data in sensitivity_analysis['galaxy_performance'].items():
            if galaxy_data['total_tests'] > 0:
                detection_rate = galaxy_data['detections'] / galaxy_data['total_tests']
                best_targets.append({
                    'galaxy': galaxy_name,
                    'detection_rate': detection_rate,
                    'best_snr': galaxy_data['best_snr'],
                    'total_detections': galaxy_data['detections']
                })
        
        # Sort by detection rate and SNR
        best_targets.sort(key=lambda x: (x['detection_rate'], x['best_snr']), reverse=True)
        sensitivity_analysis['best_targets'] = best_targets[:5]  # Top 5
        
        # Print summary
        total_sims = sensitivity_analysis['total_simulations']
        detectable_sims = sensitivity_analysis['detectable_simulations']
        overall_rate = detectable_sims / total_sims if total_sims > 0 else 0
        
        print(f"Total simulations: {total_sims}")
        print(f"Detectable signals: {detectable_sims}")
        print(f"Overall detection rate: {overall_rate:.2%}")
        
        print(f"\n🥇 Best DM models:")
        for dm_model, data in sensitivity_analysis['dm_model_performance'].items():
            print(f"   {dm_model}: {data['detection_rate']:.2%} ({data['detections']}/{data['total_tests']})")
        
        print(f"\n🎯 Best target galaxies:")
        for target in sensitivity_analysis['best_targets']:
            print(f"   {target['galaxy']}: {target['detection_rate']:.2%}, SNR: {target['best_snr']:.2f}")
        
        return sensitivity_analysis
    
    def create_visualization_plots(self, results: Optional[List[Dict]] = None,
                                 save_plots: bool = True):
        """
        Create visualization plots for simulation results
        
        Args:
            results: Simulation results
            save_plots: Whether to save plots to files
        """
        if results is None:
            results = self.simulation_results
        
        if not results:
            print("No results available for plotting")
            return
        
        print("\n📊 Creating visualization plots...")
        
        # Collect data for plotting
        plot_data = []
        for galaxy_result in results:
            if 'error' in galaxy_result:
                continue
            
            galaxy_name = galaxy_result['galaxy']
            for dm_model, dm_results in galaxy_result.get('dm_model_results', {}).items():
                for exp_time, exp_result in dm_results.get('exposure_results', {}).items():
                    if 'error' not in exp_result:
                        plot_data.append({
                            'galaxy': galaxy_name,
                            'dm_model': dm_model,
                            'exposure_time_hours': exp_time / 3600,
                            'snr': exp_result.get('snr', 0),
                            'detectable': exp_result.get('detectable', False),
                            'rms_phase_shift': exp_result.get('rms_phase_shift', 0)
                        })
        
        if not plot_data:
            print("No valid data for plotting")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BEC Dark Matter Detection Simulation Results', fontsize=16)
        
        # Plot 1: SNR vs Exposure Time by DM Model
        ax1 = axes[0, 0]
        for dm_model in df['dm_model'].unique():
            model_data = df[df['dm_model'] == dm_model]
            ax1.scatter(model_data['exposure_time_hours'], model_data['snr'], 
                       label=dm_model, alpha=0.7, s=50)
        
        ax1.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Detection threshold')
        ax1.set_xlabel('Exposure Time (hours)')
        ax1.set_ylabel('Signal-to-Noise Ratio')
        ax1.set_title('SNR vs Exposure Time by DM Model')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Detection Rate by Galaxy
        ax2 = axes[0, 1]
        galaxy_detection_rates = []
        galaxy_names = []
        
        for galaxy in df['galaxy'].unique():
            galaxy_data = df[df['galaxy'] == galaxy]
            detection_rate = galaxy_data['detectable'].mean()
            galaxy_detection_rates.append(detection_rate)
            galaxy_names.append(galaxy)
        
        bars = ax2.bar(range(len(galaxy_names)), galaxy_detection_rates, 
                       color='skyblue', edgecolor='navy', alpha=0.7)
        ax2.set_xlabel('Galaxy')
        ax2.set_ylabel('Detection Rate')
        ax2.set_title('Detection Rate by Galaxy')
        ax2.set_xticks(range(len(galaxy_names)))
        ax2.set_xticklabels(galaxy_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, galaxy_detection_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom')
        
        # Plot 3: SNR Distribution
        ax3 = axes[1, 0]
        detectable_snr = df[df['detectable']]['snr']
        non_detectable_snr = df[~df['detectable']]['snr']
        
        ax3.hist(non_detectable_snr, bins=30, alpha=0.6, color='red', 
                label=f'Non-detectable ({len(non_detectable_snr)})')
        ax3.hist(detectable_snr, bins=30, alpha=0.6, color='green', 
                label=f'Detectable ({len(detectable_snr)})')
        
        ax3.set_xlabel('Signal-to-Noise Ratio')
        ax3.set_ylabel('Count')
        ax3.set_title('SNR Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Plot 4: Phase Shift vs SNR
        ax4 = axes[1, 1]
        colors = {'axion': 'blue', 'wimp': 'red', 'sterile_neutrino': 'green'}
        
        for dm_model in df['dm_model'].unique():
            model_data = df[df['dm_model'] == dm_model]
            detectable_mask = model_data['detectable']
            
            # Plot non-detectable points
            ax4.scatter(model_data[~detectable_mask]['rms_phase_shift'], 
                       model_data[~detectable_mask]['snr'],
                       c=colors.get(dm_model, 'gray'), alpha=0.4, s=20, marker='o')
            
            # Plot detectable points
            ax4.scatter(model_data[detectable_mask]['rms_phase_shift'], 
                       model_data[detectable_mask]['snr'],
                       c=colors.get(dm_model, 'gray'), alpha=0.8, s=50, 
                       marker='*', label=f'{dm_model} (detectable)')
        
        ax4.axhline(y=3, color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('RMS Phase Shift (rad)')
        ax4.set_ylabel('Signal-to-Noise Ratio')
        ax4.set_title('Phase Shift vs SNR')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        ax4.set_xscale('log')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"bec_dm_results_{timestamp}.png"
            plot_path = self.results_dir / "plots" / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to: {plot_path}")
        
        plt.show()
    
    def generate_summary_report(self, results: Optional[List[Dict]] = None) -> str:
        """
        Generate a text summary report of simulation results
        
        Args:
            results: Simulation results
            
        Returns:
            String containing the formatted report
        """
        if results is None:
            results = self.simulation_results
        
        if not results:
            return "No simulation results available"
        
        # Analyze sensitivity first
        sensitivity = self.analyze_detection_sensitivity(results)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
BEC DARK MATTER DETECTION SIMULATION REPORT
Generated: {timestamp}
{'=' * 60}

OVERVIEW:
- Total galaxies simulated: {sensitivity['total_galaxies']}
- Total simulations run: {sensitivity['total_simulations']}
- Detectable signals: {sensitivity['detectable_simulations']}
- Overall detection rate: {sensitivity['detectable_simulations']/sensitivity['total_simulations']:.2%}

DARK MATTER MODEL PERFORMANCE:
"""
        
        for dm_model, data in sensitivity['dm_model_performance'].items():
            report += f"- {dm_model.upper()}:\n"
            report += f"  Detection rate: {data['detection_rate']:.2%}\n"
            report += f"  Successful detections: {data['detections']}/{data['total_tests']}\n"
            
            # Get DM model parameters
            if dm_model in DM_MODELS:
                params = DM_MODELS[dm_model]
                report += f"  Mass: {params['mass']:.2e} kg\n"
                report += f"  Cross-section: {params['cross_section']:.2e} m²\n"
            report += "\n"
        
        report += "BEST TARGET GALAXIES:\n"
        for i, target in enumerate(sensitivity['best_targets'], 1):
            report += f"{i}. {target['galaxy']}:\n"
            report += f"   Detection rate: {target['detection_rate']:.2%}\n"
            report += f"   Best SNR achieved: {target['best_snr']:.2f}\n"
            report += f"   Total detections: {target['total_detections']}\n\n"
        
        report += "EXPOSURE TIME ANALYSIS:\n"
        for exp_time, data in sensitivity['exposure_time_analysis'].items():
            hours = exp_time / 3600
            report += f"- {hours:.1f} hours: {data['detection_rate']:.2%} "
            report += f"({data['detections']}/{data['total_tests']})\n"
        
        report += f"\n{'=' * 60}\n"
        report += "RECOMMENDATIONS:\n"
        
        # Add recommendations based on results
        best_dm_model = max(sensitivity['dm_model_performance'].keys(), 
                           key=lambda x: sensitivity['dm_model_performance'][x]['detection_rate'])
        best_galaxy = sensitivity['best_targets'][0]['galaxy'] if sensitivity['best_targets'] else "None"
        
        report += f"- Focus on {best_dm_model} dark matter model (highest detection rate)\n"
        report += f"- Prioritize observations of {best_galaxy} (best target galaxy)\n"
        
        # Find optimal exposure time
        best_exp_time = max(sensitivity['exposure_time_analysis'].keys(),
                           key=lambda x: sensitivity['exposure_time_analysis'][x]['detection_rate'])
        report += f"- Use exposure times ≥ {best_exp_time/3600:.1f} hours for optimal sensitivity\n"
        
        if sensitivity['detectable_simulations'] == 0:
            report += "- Consider improving BEC sensitivity or targeting denser DM regions\n"
        
        return report

def main():
    """Main simulation function"""
    print("🚀 BEC DARK MATTER GALAXY SIMULATION")
    print("=" * 50)
    
    # Initialize simulation runner
    print("\n1. Initializing simulation environment...")
    runner = GalaxySimulationRunner()
    
    # Load galaxies
    print("\n2. Loading galaxy data...")
    galaxies = runner.load_galaxies(n_galaxies=5)
    
    if not galaxies:
        print("❌ No galaxies available. Run download_and_organize.py first!")
        return
    
    print(f"✅ Loaded {len(galaxies)} galaxies for simulation")
    
    # Run simulations
    print("\n3. Running BEC dark matter simulations...")
    try:
        results = runner.run_full_simulation_suite(
            galaxies=galaxies,
            dm_models=['axion', 'wimp', 'sterile_neutrino'],
            exposure_times=[1800, 3600, 7200, 14400],  # 0.5h to 4h
            save_results=True
        )
        
        if not results:
            print("❌ No simulation results generated")
            return
        
        print(f"✅ Completed {len(results)} galaxy simulations")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Analyze results
    print("\n4. Analyzing detection sensitivity...")
    try:
        sensitivity = runner.analyze_detection_sensitivity(results)
        print("✅ Sensitivity analysis complete")
    except Exception as e:
        print(f"⚠️  Analysis error: {e}")
        sensitivity = {}
    
    # Create plots
    print("\n5. Creating visualization plots...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress matplotlib warnings
            runner.create_visualization_plots(results, save_plots=True)
        print("✅ Plots generated")
    except Exception as e:
        print(f"⚠️  Plotting error: {e}")
    
    # Generate report
    print("\n6. Generating summary report...")
    try:
        report = runner.generate_summary_report(results)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"simulation_report_{timestamp}.txt"
        report_path = runner.results_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📊 Report saved to: {report_path}")
        
        # Print key findings
        print("\n" + "=" * 50)
        print("KEY FINDINGS:")
        print("=" * 50)
        
        if sensitivity:
            total_sims = sensitivity['total_simulations']
            detectable = sensitivity['detectable_simulations']
            rate = detectable / total_sims if total_sims > 0 else 0
            
            print(f"🎯 Overall detection rate: {rate:.2%}")
            print(f"📊 Total simulations: {total_sims} ({detectable} detectable)")
            
            if sensitivity['best_targets']:
                best_target = sensitivity['best_targets'][0]
                print(f"🌟 Best target: {best_target['galaxy']} ({best_target['detection_rate']:.2%} rate)")
            
            if sensitivity['dm_model_performance']:
                best_model = max(sensitivity['dm_model_performance'].keys(),
                               key=lambda x: sensitivity['dm_model_performance'][x]['detection_rate'])
                best_rate = sensitivity['dm_model_performance'][best_model]['detection_rate']
                print(f"🔬 Best DM model: {best_model} ({best_rate:.2%} rate)")
        
        # Print sample of report
        print("\nSAMPLE REPORT:")
        print("-" * 30)
        report_lines = report.split('\n')
        for line in report_lines[:15]:  # First 15 lines
            print(line)
        print("...")
        print(f"(Full report saved to {report_filename})")
        
    except Exception as e:
        print(f"⚠️  Report generation error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 SIMULATION COMPLETE!")
    print("=" * 50)
    print(f"📁 Results saved in: {runner.results_dir}")
    print(f"📊 Plots available in: {runner.results_dir}/plots/")
    print(f"🗂️  Raw data in: {runner.results_dir}/simulations/")
    print("\n💡 Next steps:")
    print("   - Review the sensitivity analysis")
    print("   - Examine the visualization plots") 
    print("   - Consider optimizing BEC parameters for best targets")
    print("   - Plan observational campaigns based on results")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        print("Partial results may be available in the results/ directory")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Troubleshooting tips:")
        print("   - Ensure all source modules are in src/ directory")
        print("   - Run download_and_organize.py first to prepare galaxy data")
        print("   - Check that required packages are installed: numpy, scipy, matplotlib, pandas")
        print("   - Verify galaxy parameter files exist in data/galaxy_parameters/")