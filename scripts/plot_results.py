#!/usr/bin/env python3
# scripts/plot_results.py
"""
Comprehensive visualization suite for astrophysical BEC dark matter detection simulation.
Creates publication-quality plots showing phase shift signatures and spectral analysis.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils import RESULTS_DIR, FIGURES_DIR, TIME_SERIES_DIR, SPECTRA_DIR

class BECDarkMatterPlotter:
    """
    Comprehensive plotting suite for BEC dark matter detection analysis.
    """
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8-whitegrid'):
        """Initialize plotter with consistent styling."""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.figsize = figsize
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#E74C3C', 
            'accent': '#F39C12',
            'dark': '#2C3E50',
            'light': '#85C1E9'
        }
        
    def load_simulation_data(self, filename="delta_phi_test.npz"):
        """Load simulation results from file."""
        data_path = TIME_SERIES_DIR / filename
        if not data_path.exists():
            raise FileNotFoundError(f"Simulation data not found: {data_path}")
        
        data = np.load(data_path)
        return data['times'], data['delta_phi']
    
    def load_psd_data(self, filename="delta_phi_psd.npz"):
        """Load power spectral density data."""
        psd_path = SPECTRA_DIR / filename
        if not psd_path.exists():
            return None, None
        
        data = np.load(psd_path)
        return data['f'], data['Pxx']
    
    def plot_phase_evolution(self, times, delta_phi, dm_params=None, save_name="phase_evolution.png"):
        """
        Plot the time evolution of phase shift with DM detection signatures.
        
        Args:
            times: time array (seconds)
            delta_phi: phase difference array (radians)
            dm_params: dict with DM parameters for annotation
            save_name: output filename
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
        
        # Main phase evolution plot
        ax1.plot(times * 1000, delta_phi, color=self.colors['primary'], 
                linewidth=1.5, alpha=0.8, label='Measured Phase Shift')
        
        # Add running average to show trends
        if len(delta_phi) > 20:
            window = max(len(delta_phi) // 20, 10)
            running_avg = np.convolve(delta_phi, np.ones(window)/window, mode='same')
            ax1.plot(times * 1000, running_avg, color=self.colors['secondary'], 
                    linewidth=2, alpha=0.9, label=f'Running Average (N={window})')
        
        # Highlight potential detection regions
        phase_std = np.std(delta_phi)
        detection_threshold = 3 * phase_std
        detection_mask = np.abs(delta_phi) > detection_threshold
        
        if np.any(detection_mask):
            ax1.fill_between(times * 1000, -np.pi, np.pi, 
                           where=detection_mask, alpha=0.2, 
                           color=self.colors['accent'], 
                           label=f'Potential DM Signal (>3σ)')
        
        ax1.set_ylabel('Relative Phase Shift Δφ (rad)', fontsize=12)
        ax1.set_title('Astrophysical BEC Dark Matter Detection: Phase Evolution', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_ylim(-np.pi, np.pi)
        
        # Add parameter annotation if provided
        if dm_params:
            info_text = f"ULDM Mass: {dm_params.get('m_phi_ev', 'N/A')} eV\n"
            info_text += f"Amplitude: {dm_params.get('amplitude_J', 'N/A')} J\n"
            info_text += f"Frequency: {dm_params.get('omega', 'N/A'):.2e} rad/s"
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Phase derivative (rate of change)
        dt = times[1] - times[0]
        phase_derivative = np.gradient(delta_phi, dt)
        ax2.plot(times * 1000, phase_derivative, color=self.colors['dark'], 
                linewidth=1, alpha=0.7)
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('dΔφ/dt (rad/s)', fontsize=12)
        ax2.set_title('Phase Shift Rate', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Phase evolution plot saved: {FIGURES_DIR / save_name}")
    
    def plot_frequency_analysis(self, times, delta_phi, dm_params=None, save_name="frequency_analysis.png"):
        """
        Comprehensive frequency domain analysis showing DM signatures.
        """
        dt = times[1] - times[0]
        fs = 1.0 / dt
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, height_ratios=[2, 2, 1], width_ratios=[2, 1])
        
        # 1. Power Spectral Density
        ax1 = fig.add_subplot(gs[0, 0])
        f, Pxx = welch(delta_phi, fs=fs, nperseg=min(256, len(delta_phi)))
        ax1.loglog(f[1:], Pxx[1:], color=self.colors['primary'], linewidth=1.5)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power Spectral Density')
        ax1.set_title('Power Spectrum of Phase Shifts', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Highlight expected DM frequency if parameters provided
        if dm_params and 'omega' in dm_params:
            f_dm = dm_params['omega'] / (2 * np.pi)
            if f_dm < max(f):
                ax1.axvline(f_dm, color=self.colors['secondary'], linestyle='--', 
                           linewidth=2, alpha=0.8, label=f'Expected DM freq: {f_dm:.2e} Hz')
                ax1.legend()
        
        # 2. FFT Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        fft_freqs = fftfreq(len(delta_phi), dt)
        fft_vals = np.abs(fft(delta_phi))
        positive_freq_mask = fft_freqs > 0
        ax2.semilogy(fft_freqs[positive_freq_mask], fft_vals[positive_freq_mask], 
                    color=self.colors['accent'], linewidth=1.5)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('FFT Amplitude')
        ax2.set_title('FFT Spectrum')
        ax2.grid(True, alpha=0.3)
        
        # 3. Spectrogram
        ax3 = fig.add_subplot(gs[1, :])
        f_spec, t_spec, Sxx = plt.specgram(delta_phi, Fs=fs, cmap='viridis')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_title('Time-Frequency Analysis (Spectrogram)', fontweight='bold')
        
        # 4. Phase coherence analysis
        ax4 = fig.add_subplot(gs[2, :])
        # Calculate phase coherence over time windows
        window_size = len(delta_phi) // 10
        coherence_times = []
        phase_vars = []
        
        for i in range(0, len(delta_phi) - window_size, window_size//2):
            window_data = delta_phi[i:i+window_size]
            coherence_times.append(times[i + window_size//2])
            phase_vars.append(np.var(window_data))
        
        ax4.plot(np.array(coherence_times) * 1000, phase_vars, 
                color=self.colors['secondary'], marker='o', markersize=4, linewidth=1.5)
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Phase Variance (rad²)')
        ax4.set_title('Temporal Phase Coherence Analysis')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Frequency analysis plot saved: {FIGURES_DIR / save_name}")
    
    def plot_detection_summary(self, times, delta_phi, dm_params=None, save_name="detection_summary.png"):
        """
        Create a comprehensive detection summary dashboard.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Astrophysical BEC Dark Matter Detection Summary', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Phase evolution with statistics
        ax1 = axes[0, 0]
        ax1.plot(times * 1000, delta_phi, color=self.colors['primary'], 
                linewidth=1.5, alpha=0.8)
        
        # Add statistical annotations
        phase_mean = np.mean(delta_phi)
        phase_std = np.std(delta_phi)
        phase_rms = np.sqrt(np.mean(delta_phi**2))
        
        ax1.axhline(phase_mean, color=self.colors['secondary'], linestyle='--', 
                   alpha=0.7, label=f'Mean: {phase_mean:.3f} rad')
        ax1.axhline(phase_mean + phase_std, color=self.colors['accent'], 
                   linestyle=':', alpha=0.7, label=f'±1σ: {phase_std:.3f} rad')
        ax1.axhline(phase_mean - phase_std, color=self.colors['accent'], 
                   linestyle=':', alpha=0.7)
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Phase Shift (rad)')
        ax1.set_title('Phase Evolution & Statistics')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram of phase values
        ax2 = axes[0, 1]
        counts, bins, patches = ax2.hist(delta_phi, bins=50, density=True, 
                                        alpha=0.7, color=self.colors['light'], 
                                        edgecolor=self.colors['dark'])
        
        # Overlay theoretical normal distribution
        x_theory = np.linspace(min(delta_phi), max(delta_phi), 100)
        y_theory = (1/np.sqrt(2*np.pi*phase_std**2)) * np.exp(-0.5*((x_theory-phase_mean)/phase_std)**2)
        ax2.plot(x_theory, y_theory, color=self.colors['secondary'], 
                linewidth=2, label='Normal Distribution')
        
        ax2.set_xlabel('Phase Shift (rad)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Phase Distribution Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Power spectrum with peak detection
        ax3 = axes[1, 0]
        dt = times[1] - times[0]
        fs = 1.0 / dt
        try:
            f, Pxx = welch(delta_phi, fs=fs, nperseg=min(256, len(delta_phi)))
            
            ax3.loglog(f[1:], Pxx[1:], color=self.colors['primary'], linewidth=1.5)
            
            # Find and mark peaks
            peaks, properties = find_peaks(Pxx[1:], height=np.max(Pxx[1:])*0.1)
            if len(peaks) > 0:
                ax3.scatter(f[peaks+1], Pxx[peaks+1], color=self.colors['secondary'], 
                           s=50, marker='o', zorder=5, label=f'{len(peaks)} peaks found')
                ax3.legend()
            
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Power Spectral Density')
            ax3.set_title('Spectral Peak Analysis')
            ax3.grid(True, alpha=0.3)
        except Exception as e:
            ax3.text(0.5, 0.5, f'PSD Error: {str(e)}', transform=ax3.transAxes, ha='center', va='center')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Power Spectral Density')
            ax3.set_title('Spectral Peak Analysis')
            ax3.grid(True, alpha=0.3)
        
        # 4. Detection metrics and parameters
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate detection metrics
        snr_estimate = phase_rms / (phase_std if phase_std > 0 else 1e-10)
        detection_events = np.sum(np.abs(delta_phi) > 3*phase_std)
        duty_cycle = detection_events / len(delta_phi) * 100
        
        # Create metrics table
        metrics_text = f"""
DETECTION METRICS
{'='*25}
Signal-to-Noise Ratio: {snr_estimate:.2f}
Phase RMS: {phase_rms:.4f} rad
Standard Deviation: {phase_std:.4f} rad
Detection Events (>3σ): {detection_events}
Detection Duty Cycle: {duty_cycle:.1f}%
Simulation Duration: {max(times)*1000:.1f} ms
Sample Rate: {fs:.0f} Hz

SYSTEM PARAMETERS
{'='*25}"""
        
        if dm_params:
            metrics_text += f"""
ULDM Mass: {dm_params.get('m_phi_ev', 'N/A')} eV
DM Frequency: {dm_params.get('omega', 0)/(2*np.pi):.2e} Hz
Amplitude: {dm_params.get('amplitude_J', 'N/A')} J
Interaction Strength: {dm_params.get('g', 'N/A')} J·m²
"""
        
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Detection summary plot saved: {FIGURES_DIR / save_name}")
    
    def plot_sensitivity_analysis(self, times, delta_phi, save_name="sensitivity_analysis.png"):
        """
        Analyze and visualize detection sensitivity limits.
        """
        dt = times[1] - times[0]
        fs = 1.0 / dt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('BEC Dark Matter Detection Sensitivity Analysis', 
                     fontsize=14, fontweight='bold')
        
        # 1. Sensitivity vs integration time
        ax1 = axes[0, 0]
        integration_times = np.logspace(-3, np.log10(max(times)), 20)
        sensitivities = []
        
        for t_int in integration_times:
            n_samples = int(t_int / dt)
            if n_samples < len(delta_phi):
                sensitivity = np.std(delta_phi[:n_samples]) / np.sqrt(n_samples)
                sensitivities.append(sensitivity)
            else:
                sensitivities.append(np.nan)
        
        valid_mask = ~np.isnan(sensitivities)
        ax1.loglog(integration_times[valid_mask] * 1000, 
                  np.array(sensitivities)[valid_mask], 
                  color=self.colors['primary'], linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Integration Time (ms)')
        ax1.set_ylabel('Phase Sensitivity (rad/√Hz)')
        ax1.set_title('Sensitivity vs Integration Time')
        ax1.grid(True, alpha=0.3)
        
        # 2. Allan deviation analysis
        ax2 = axes[0, 1]
        # Simplified Allan deviation calculation
        tau_values = np.logspace(np.log10(dt), np.log10(max(times)/10), 15)
        allan_dev = []
        
        for tau in tau_values:
            m = int(tau / dt)
            if m < len(delta_phi)//3:
                # Calculate Allan deviation
                y_avg = []
                for i in range(0, len(delta_phi) - 2*m, m):
                    y_avg.append(np.mean(delta_phi[i:i+m]))
                
                if len(y_avg) > 1:
                    allan_var = 0.5 * np.mean(np.diff(y_avg)**2)
                    allan_dev.append(np.sqrt(allan_var))
                else:
                    allan_dev.append(np.nan)
            else:
                allan_dev.append(np.nan)
        
        valid_mask = ~np.isnan(allan_dev)
        if np.any(valid_mask):
            ax2.loglog(tau_values[valid_mask] * 1000, 
                      np.array(allan_dev)[valid_mask],
                      color=self.colors['secondary'], linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Averaging Time (ms)')
        ax2.set_ylabel('Allan Deviation (rad)')
        ax2.set_title('Allan Deviation Analysis')
        ax2.grid(True, alpha=0.3)
        
        # 3. Frequency-dependent sensitivity
        ax3 = axes[1, 0]
        f, Pxx = welch(delta_phi, fs=fs, nperseg=min(128, len(delta_phi)))
        # Convert PSD to amplitude spectral density
        asd = np.sqrt(Pxx * fs)
        ax3.loglog(f[1:], asd[1:], color=self.colors['accent'], linewidth=1.5)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Amplitude Spectral Density (rad/√Hz)')
        ax3.set_title('Frequency-Dependent Sensitivity')
        ax3.grid(True, alpha=0.3)
        
        # 4. Detection threshold analysis
        ax4 = axes[1, 1]
        thresholds = np.linspace(0.1, 10, 50) * np.std(delta_phi)
        detection_rates = []
        false_positive_rates = []
        
        for threshold in thresholds:
            detections = np.sum(np.abs(delta_phi) > threshold)
            detection_rate = detections / len(delta_phi)
            detection_rates.append(detection_rate)
            
            # Estimate false positive rate assuming Gaussian noise
            false_positive_rate = 2 * (1 - 0.5 * (1 + np.sign(threshold) * 
                                     np.sqrt(2/np.pi) * threshold / np.std(delta_phi)))
            false_positive_rates.append(max(false_positive_rate, 1e-6))
        
        ax4.semilogy(thresholds / np.std(delta_phi), detection_rates, 
                    color=self.colors['primary'], linewidth=2, label='Detection Rate')
        ax4.semilogy(thresholds / np.std(delta_phi), false_positive_rates, 
                    color=self.colors['secondary'], linewidth=2, linestyle='--', 
                    label='False Positive Rate')
        ax4.set_xlabel('Threshold (σ units)')
        ax4.set_ylabel('Rate')
        ax4.set_title('Detection Threshold Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sensitivity analysis plot saved: {FIGURES_DIR / save_name}")

def main():
    """Main plotting function."""
    plotter = BECDarkMatterPlotter()
    
    try:
        # Load simulation data
        times, delta_phi = plotter.load_simulation_data()
        
        # Example DM parameters (replace with actual values from your simulation)
        dm_params = {
            'm_phi_ev': 1e-18,
            'amplitude_J': 1e-30,
            'omega': 1e15,  # This should match your actual omega calculation
            'g': 1e-52
        }
        
        print(f"Loaded data: {len(times)} time points, {max(times)*1000:.1f} ms duration")
        print(f"Phase shift range: [{min(delta_phi):.4f}, {max(delta_phi):.4f}] rad")
        
        # Generate all plots
        print("\nGenerating plots...")
        plotter.plot_phase_evolution(times, delta_phi, dm_params, "phase_evolution_detailed.png")
        plotter.plot_frequency_analysis(times, delta_phi, dm_params, "frequency_analysis_comprehensive.png")
        plotter.plot_detection_summary(times, delta_phi, dm_params, "detection_summary_dashboard.png")
        plotter.plot_sensitivity_analysis(times, delta_phi, "sensitivity_analysis_detailed.png")
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print(f"All plots saved to: {FIGURES_DIR}")
        print(f"Check the following files:")
        print("- phase_evolution_detailed.png")
        print("- frequency_analysis_comprehensive.png") 
        print("- detection_summary_dashboard.png")
        print("- sensitivity_analysis_detailed.png")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run your simulation first to generate the data files.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check your data files and try again.")

if __name__ == "__main__":
    main()