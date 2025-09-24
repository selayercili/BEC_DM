#!/usr/bin/env python3
"""
comp_plot.py

Run a BEC simulation with a boosted test DM mass (for visualization) and a
separate simulation without DM. Produce separate PSD and time-domain plots
with a SHARED y-axis range so the curves are directly comparable.

Usage:
    python scripts/comp_plot.py
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

# Ensure project root on path
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.bec_simulation import BECSimulation
from src.environment import create_environment_potential, neutron_star_potential
from src.dark_matter import ul_dm_cosine_potential

# --- USER VISUAL DEBUG PARAMETERS (boosted) ---
# Use a test mass to get visible oscillations quickly (tweakable)
TEST_MPHI_EV = 1e-12      # eV (boosted for visualization)
TEST_AMPLITUDE_J = 1e-18  # potential amplitude (boosted)
TEST_T_TOTAL = 7.0        # seconds
# Simulation grid/time parameters (match your normal run)
NX = NY = 128
DX = DY = 1.0
DT = 1e-3
G = 1e-52
M_PARTICLE = 1.6726219e-27

# --- helpers ---
def compute_psd(times, signal):
    dt = float(times[1] - times[0])
    fs = 1.0 / dt
    nperseg = min(2048, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd

def save_time_plot(times, signal, out_path, title="Phase vs time",
                   zoom_first_sec=True, y_limits=None):
    """Save main and optional zoomed time-domain plots. Supports fixed y-limits."""
    plt.figure(figsize=(8,3))
    plt.plot(times, signal, alpha=0.8)
    # running average
    N = max(3, len(signal)//200)
    avg = np.convolve(signal, np.ones(N)/N, mode='same')
    plt.plot(times, avg, color='red', linewidth=1.5, label=f"running avg (N={N})")

    # y-axis scaling (shared or auto)
    if y_limits is not None:
        plt.ylim(*y_limits)
    else:
        ymin, ymax = np.min(signal), np.max(signal)
        pad = 0.1 * (ymax - ymin) if ymax > ymin else 1.0
        plt.ylim(ymin - pad, ymax + pad)

    plt.xlabel("Time (s)")
    plt.ylabel("Δφ [rad]")
    plt.title(title)
    plt.grid(True, ls='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    if zoom_first_sec:
        plt.figure(figsize=(6,2.5))
        plt.plot(times, signal, alpha=0.8)
        plt.plot(times, avg, color='red', linewidth=1.5)
        plt.xlim(0, min(1.0, times[-1]))
        if y_limits is not None:
            plt.ylim(*y_limits)  # keep zoom on same vertical scale for fair comparison
        else:
            ymin, ymax = np.min(signal), np.max(signal)
            pad = 0.1 * (ymax - ymin) if ymax > ymin else 1.0
            plt.ylim(ymin - pad, ymax + pad)
        plt.xlabel("Time (s)")
        plt.ylabel("Δφ [rad]")
        plt.title(title + " (zoom)")
        plt.grid(True, ls='--', alpha=0.4)
        plt.tight_layout()
        out_zoom = out_path.with_name(out_path.stem + "_zoom.png")
        plt.savefig(out_zoom, dpi=150, bbox_inches='tight')
        plt.close()

def save_psd_plot(freqs, psd, out_path, f_dm=None, xlim_max=None):
    plt.figure(figsize=(8,3.6))
    plt.semilogy(freqs, psd, color='tab:blue')
    if f_dm is not None:
        plt.axvline(f_dm, color='red', linestyle='--', label=f"f_DM = {f_dm:.3e} Hz")
        plt.legend()
    if xlim_max:
        plt.xlim(0, xlim_max)
    psd_nonzero = psd[np.isfinite(psd) & (psd > 0)]
    if psd_nonzero.size:
        vlow = np.percentile(psd_nonzero, 1.0)
        vhigh = np.percentile(psd_nonzero, 99.0)
        if vhigh > vlow:
            plt.ylim(max(vlow*0.1, 1e-40), vhigh*10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.grid(True, ls='--', alpha=0.4, which='both')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

# --- Run or load DM run (but override m_phi/amplitude for visualization) ---
def run_with_dm_test():
    # Create sim with same grid and dt as your usual run
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=TEST_T_TOTAL)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    # ULDM potential (boosted for visibility)
    V_dm = ul_dm_cosine_potential((sim.X, sim.Y),
                                  amplitude_J=TEST_AMPLITUDE_J,
                                  m_phi_ev=TEST_MPHI_EV,
                                  phase0=0.0,
                                  v_dm=220e3,
                                  direction=0.0,
                                  spatial_modulation=True)

    # Environment potential (neutron star centered at origin)
    x0, y0 = 0.0, 0.0
    R_ns = 1.0  # in grid units
    V_env = create_environment_potential(sim.X, sim.Y,
                                         neutron_star_potential,
                                         center_offset=(x0, y0),
                                         R_ns=R_ns)

    def V_total(coords, t):
        return V_dm(coords, t) + V_env

    # Diagnostic: sample potential at center over a short time
    ts = np.linspace(0, min(1.0, TEST_T_TOTAL), 400)
    try:
        vals = np.array([V_total((sim.X, sim.Y), tt)[sim.ny//2, sim.nx//2] for tt in ts])
    except Exception as e:
        print("[WARN] Potential diagnostic failed:", e)
        vals = np.zeros_like(ts)
    diag_path = PLOTS_DIR / "potential_diagnostic_with_dm.png"
    plt.figure(figsize=(5,2))
    plt.plot(ts, vals)
    plt.xlabel("time [s]"); plt.ylabel("V_dm+V_env at center [J]")
    plt.title("Potential diagnostic (center) — with DM")
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(diag_path, bbox_inches='tight', dpi=140)
    plt.close()
    print("[INFO] Saved potential diagnostic (with DM) to", diag_path)

    # Run the full sim
    print("[INFO] Running WITH-DM test simulation (boosted mass)...")
    res = sim.run(V_function=V_total, snapshot_interval=0)
    out_file = RESULTS_DIR / "delta_phi_test_visual.npz"
    np.savez_compressed(out_file, times=res.times, delta_phi=res.delta_phi)
    print("[INFO] Saved boosted-DM timeseries to:", out_file)
    return res.times, res.delta_phi, vals, ts

# --- Run the no-DM sim ---
def run_no_dm(t_total):
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=t_total)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)
    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential)

    def V_total(coords, t):
        return V_env

    # diagnostic of env at center
    ts = np.linspace(0, min(1.0, t_total), 200)
    vals = np.array([V_total((sim.X, sim.Y), tt)[sim.ny//2, sim.nx//2] for tt in ts])
    diag_path = PLOTS_DIR / "potential_diagnostic_no_dm.png"
    plt.figure(figsize=(5,2))
    plt.plot(ts, vals)
    plt.xlabel("time [s]"); plt.ylabel("V_env at center [J]")
    plt.title("Potential diagnostic (center) — no DM")
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(diag_path, bbox_inches='tight', dpi=140)
    plt.close()
    print("[INFO] Saved potential diagnostic (no DM) to", diag_path)

    print("[INFO] Running NO-DM simulation...")
    res = sim.run(V_function=V_total, snapshot_interval=0)
    out_file = RESULTS_DIR / "delta_phi_no_dm_visual.npz"
    np.savez_compressed(out_file, times=res.times, delta_phi=res.delta_phi)
    print("[INFO] Saved no-DM timeseries to:", out_file)
    return res.times, res.delta_phi, vals, ts

def main():
    # 1) run boosted DM test run (or load if exists)
    boosted_file = RESULTS_DIR / "delta_phi_test_visual.npz"
    if boosted_file.exists():
        data = np.load(boosted_file)
        times_dm, dp_dm = data["times"], data["delta_phi"]
        vals_dm = None
        print("[INFO] Loaded existing boosted DM visual run.")
    else:
        times_dm, dp_dm, vals_dm, ts_dm = run_with_dm_test()

    # 2) run no-DM (or load if exists)
    no_dm_file = RESULTS_DIR / "delta_phi_no_dm_visual.npz"
    if no_dm_file.exists():
        data = np.load(no_dm_file)
        times_no_dm, dp_no_dm = data["times"], data["delta_phi"]
        print("[INFO] Loaded existing no-DM visual run.")
        vals_no_dm = None
    else:
        times_no_dm, dp_no_dm, vals_no_dm, ts_no_dm = run_no_dm(TEST_T_TOTAL)

    # --- Align y-axis for fair comparison across BOTH plots (and their zooms) ---
    combined_min = float(min(dp_dm.min(), dp_no_dm.min()))
    combined_max = float(max(dp_dm.max(), dp_no_dm.max()))
    pad = 0.1 * (combined_max - combined_min) if combined_max > combined_min else 1.0
    y_limits = (combined_min - pad, combined_max + pad)

    # save and plot time-domain with shared y-axis
    save_time_plot(times_dm, dp_dm, PLOTS_DIR / "phase_with_dm.png",
               title="Δφ(t) — With (visual-test) DM", y_limits=y_limits)

    save_time_plot(times_no_dm, dp_no_dm, PLOTS_DIR / "phase_no_dm.png",
               title="Δφ(t) — No DM", y_limits=y_limits)

    # PSD plots
    freqs_dm, psd_dm = compute_psd(times_dm, dp_dm)
    save_psd_plot(freqs_dm, psd_dm, PLOTS_DIR / "psd_with_dm.png", f_dm=None, xlim_max=None)

    freqs_no_dm, psd_no_dm = compute_psd(times_no_dm, dp_no_dm)
    save_psd_plot(freqs_no_dm, psd_no_dm, PLOTS_DIR / "psd_no_dm.png", f_dm=None, xlim_max=None)

    # 3) Save PSD numeric data for later
    np.savez_compressed(PLOTS_DIR / "psd_visual_data.npz",
                        freqs_dm=freqs_dm, psd_dm=psd_dm,
                        freqs_no_dm=freqs_no_dm, psd_no_dm=psd_no_dm)

    print("[DONE] Generated: phase_with_dm.png, psd_with_dm.png, phase_no_dm.png, psd_no_dm.png")
    print("Potential diagnostics saved as potential_diagnostic_with_dm.png and potential_diagnostic_no_dm.png")

if __name__ == "__main__":
    main()
