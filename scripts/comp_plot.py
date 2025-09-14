#!/usr/bin/env python3
"""
comp_plot.py

Run a BEC simulation *without* the DM potential, compute PSDs for:
  - the previously-saved DM run (results/time_series/delta_phi_test.npz)
  - the new no-DM run (saved to results/time_series/delta_phi_no_dm.npz)

Then plot both PSDs together and save to results/plots/psd_comparison.png.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "time_series"
PLOTS_DIR = BASE_DIR / "results" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Import simulation pieces
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.bec_simulation import BECSimulation
from src.environment import create_environment_potential, neutron_star_potential

# --- User-tunable parameters (match your DM run choices) ---
# If your DM run used a different m_phi, change this to match the red vertical line.
m_phi_ev_for_label = 1e-18  # eV (used for marking expected DM frequency line)
# Simulation parameters for the no-DM run (match your DM-run settings for a fair comparison)
nx = ny = 128
dx = dy = 1.0
m_particle = 1.6726219e-27
g = 1e-52
dt = 1e-3
# t_total will be inferred from the existing DM times (if found), else fallback
default_t_total = 20.0

# --- Helpers ---
def compute_psd(times, signal):
    """Return (freqs, psd) using Welch with safe nperseg."""
    dt = float(times[1] - times[0])
    fs = 1.0 / dt
    nperseg = min(2048, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd

def load_dm_result():
    dm_file = RESULTS_DIR / "delta_phi_test.npz"
    if not dm_file.exists():
        raise FileNotFoundError(f"DM results file not found: {dm_file}\nRun your DM simulation first.")
    data = np.load(dm_file)
    times = data["times"]
    delta_phi = data["delta_phi"]
    print(f"[INFO] Loaded DM results: {dm_file}  (N={len(times)})")
    return times, delta_phi

def run_no_dm_sim(t_total, nx=128, ny=128, dx=1.0, dy=1.0, dt=1e-3):
    """
    Run a BECSimulation with only the environment potential (no DM).
    Saves results to results/time_series/delta_phi_no_dm.npz and returns (times, delta_phi).
    """
    print("[INFO] Running simulation WITHOUT DM...")
    sim = BECSimulation(nx=nx, ny=ny, dx=dx, dy=dy,
                        m_particle=m_particle, g=g, dt=dt, t_total=t_total)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    # Build static environment array (neutron star)
    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential)
    print(f"[DEBUG] V_env: shape={V_env.shape}, min={V_env.min():.3e}, max={V_env.max():.3e}, std={V_env.std():.3e}")

    def total_potential(coords, t):
        # coords given as (X,Y); this returns a grid-shaped array (no DM component)
        return V_env

    result = sim.run(V_function=total_potential, snapshot_interval=0)

    out_file = RESULTS_DIR / "delta_phi_no_dm.npz"
    np.savez_compressed(out_file, times=result.times, delta_phi=result.delta_phi)
    print(f"[INFO] Saved no-DM timeseries to: {out_file}")
    return result.times, result.delta_phi

# --- Main workflow ---
def main():
    # 1) load DM results
    times_dm, dp_dm = load_dm_result()

    # 2) compute PSD for DM result
    freqs_dm, psd_dm = compute_psd(times_dm, dp_dm)
    print(f"[INFO] DM PSD computed: {len(freqs_dm)} freq bins")

    # 3) prepare/run no-DM: determine t_total and dt from DM run for fairness
    if len(times_dm) >= 2:
        dt_from_dm = float(times_dm[1] - times_dm[0])
        t_total_from_dm = float(times_dm[-1] + dt_from_dm)
    else:
        dt_from_dm = dt
        t_total_from_dm = default_t_total

    # Use same dt as DM results for comparable freq axes
    print(f"[INFO] Using dt={dt_from_dm:.3e}, t_total={t_total_from_dm:.3e} for no-DM run")

    # If no-DM file exists, load it; else run the sim
    no_dm_file = RESULTS_DIR / "delta_phi_no_dm.npz"
    if no_dm_file.exists():
        data = np.load(no_dm_file)
        times_no_dm, dp_no_dm = data["times"], data["delta_phi"]
        print(f"[INFO] Loaded existing no-DM results: {no_dm_file}")
    else:
        # run using same grid sizes as the DM run (nx,ny defined above)
        times_no_dm, dp_no_dm = run_no_dm_sim(t_total_from_dm, nx=nx, ny=ny, dx=dx, dy=dy, dt=dt_from_dm)

    freqs_no_dm, psd_no_dm = compute_psd(times_no_dm, dp_no_dm)

    # 4) Plot comparison PSD (semilogy for PSD vs linear frequency)
    plt.figure(figsize=(10,4))
    plt.semilogy(freqs_dm, psd_dm, label="With DM", color="tab:blue")
    plt.semilogy(freqs_no_dm, psd_no_dm, label="No DM", color="tab:orange", linestyle="--")

    # Expected DM frequency (vertical marker)
    eV_to_J = 1.602176634e-19
    hbar = 1.054571817e-34
    omega_dm = (m_phi_ev_for_label * eV_to_J) / hbar
    f_dm = omega_dm / (2 * np.pi)
    plt.axvline(f_dm, color="red", linestyle="--", label=f"Expected DM freq = {f_dm:.3e} Hz")

    plt.xlim(left=0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title("Power Spectral Density of Δφ(t) — With vs Without DM")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    out_fig = PLOTS_DIR / "psd_comparison.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved PSD comparison plot to: {out_fig}")

    # 5) Optional: save combined numeric PSDs for later analysis
    np.savez_compressed(PLOTS_DIR / "psd_comparison_data.npz",
                        freqs_dm=freqs_dm, psd_dm=psd_dm,
                        freqs_no_dm=freqs_no_dm, psd_no_dm=psd_no_dm)

if __name__ == "__main__":
    main()
