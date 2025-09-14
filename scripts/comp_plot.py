#!/usr/bin/env python3
"""
comp_plot.py

Generates two separate PSD plots:
  - psd_with_dm.png (using saved DM simulation)
  - psd_no_dm.png (fresh run without DM)

Also saves diagnostic info about the DM potential to confirm it's working.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch
import sys

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "time_series"
PLOTS_DIR = BASE_DIR / "results" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.bec_simulation import BECSimulation
from src.environment import create_environment_potential, neutron_star_potential

# Constants
m_particle = 1.6726219e-27
g = 1e-52
nx = ny = 128
dx = dy = 1.0

# Helpers
def compute_psd(times, signal):
    dt = float(times[1] - times[0])
    fs = 1.0 / dt
    freqs, psd = welch(signal, fs=fs, nperseg=min(2048, len(signal)))
    return freqs, psd

def plot_psd(freqs, psd, title, out_file, f_dm=None):
    plt.figure(figsize=(8,4))
    plt.semilogy(freqs, psd, color="tab:blue")
    if f_dm:
        plt.axvline(f_dm, color="red", linestyle="--", label=f"Expected DM freq = {f_dm:.3e} Hz")
        plt.legend()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot: {out_file}")

def run_no_dm_sim(t_total, dt):
    print("[INFO] Running no-DM simulation...")
    sim = BECSimulation(nx=nx, ny=ny, dx=dx, dy=dy,
                        m_particle=m_particle, g=g, dt=dt, t_total=t_total)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)
    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential)

    def V_total(coords, t):
        return V_env

    result = sim.run(V_function=V_total, snapshot_interval=0)
    out_file = RESULTS_DIR / "delta_phi_no_dm.npz"
    np.savez_compressed(out_file, times=result.times, delta_phi=result.delta_phi)
    print(f"[INFO] Saved no-DM results to {out_file}")
    return result.times, result.delta_phi

def main():
    # Load DM results
    dm_file = RESULTS_DIR / "delta_phi_test.npz"
    if not dm_file.exists():
        raise FileNotFoundError("Run the DM simulation first to create delta_phi_test.npz")
    dm_data = np.load(dm_file)
    times_dm, dp_dm = dm_data["times"], dm_data["delta_phi"]
    print(f"[INFO] Loaded DM run ({len(times_dm)} steps)")

    # Compute DM PSD
    freqs_dm, psd_dm = compute_psd(times_dm, dp_dm)

    # Save DM PSD plot
    eV_to_J = 1.602176634e-19
    hbar = 1.054571817e-34
    m_phi_ev = 1e-18
    omega_dm = (m_phi_ev * eV_to_J) / hbar
    f_dm = omega_dm / (2*np.pi)
    plot_psd(freqs_dm, psd_dm, "PSD of Δφ(t) — With DM",
             PLOTS_DIR / "psd_with_dm.png", f_dm=f_dm)

    # Run or load no-DM
    no_dm_file = RESULTS_DIR / "delta_phi_no_dm.npz"
    if no_dm_file.exists():
        nd = np.load(no_dm_file)
        times_no_dm, dp_no_dm = nd["times"], nd["delta_phi"]
        print("[INFO] Loaded existing no-DM run")
    else:
        dt = float(times_dm[1] - times_dm[0])
        t_total = float(times_dm[-1])
        times_no_dm, dp_no_dm = run_no_dm_sim(t_total, dt)

    # Compute no-DM PSD
    freqs_no_dm, psd_no_dm = compute_psd(times_no_dm, dp_no_dm)

    # Save no-DM PSD plot
    plot_psd(freqs_no_dm, psd_no_dm, "PSD of Δφ(t) — No DM",
             PLOTS_DIR / "psd_no_dm.png", f_dm=None)

if __name__ == "__main__":
    main()
