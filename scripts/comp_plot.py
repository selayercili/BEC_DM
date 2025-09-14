#!/usr/bin/env python3
"""
comp_plot.py

Run a BEC simulation with a boosted test DM mass (for visualization) and a
separate simulation without DM. Produce separate PSD and time-domain plots.

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
# If you have an existing dark_matter.ul_dm_cosine_potential ensure it's compatible.
from src.dark_matter import ul_dm_cosine_potential

# --- USER VISUAL DEBUG PARAMETERS (boosted) ---
# Use a test mass to get visible oscillations quickly (tweakable)
TEST_MPHI_EV = 1e-12     # eV (boosted for visualization)
TEST_AMPLITUDE_J = 1e-18 # potential amplitude (boosted)
TEST_T_TOTAL = 7.0       # seconds (you said you set 7 s)
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

def save_time_plot(times, signal, out_path, title="Phase vs time", zoom_first_sec=True):
    plt.figure(figsize=(8,3))
    plt.plot(times, signal, alpha=0.8)
    # running average
    N = max(3, len(signal)//200)
    avg = np.convolve(signal, np.ones(N)/N, mode='same')
    plt.plot(times, avg, color='red', linewidth=1.5, label=f"running avg (N={N})")
    # autoscale y to data range (small margin)
    ymin, ymax = np.min(signal), np.max(signal)
    if np.isfinite(ymin) and np.isfinite(ymax) and (ymax - ymin) > 0:
        pad = 0.1 * (ymax - ymin)
        plt.ylim(ymin - pad, ymax + pad)
    plt.xlabel("Time (s)")
    plt.ylabel("Δφ [rad]")
    plt.title(title)
    plt.grid(True, ls='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    # zoom first second
    if zoom_first_sec:
        plt.figure(figsize=(6,2.5))
        plt.plot(times, signal, alpha=0.8)
        plt.plot(times, avg, color='red', linewidth=1.5)
        plt.xlim(0, min(1.0, times[-1]))
        if np.isfinite(ymin) and np.isfinite(ymax) and (ymax - ymin) > 0:
            pad = 0.1 * (ymax - ymin)
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
    # autoscale x and y to focus view:
    if xlim_max:
        plt.xlim(0, xlim_max)
    # tighten y-limits based on percentile to avoid huge tails:
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

    # Build ULDM potential callable using the (X,Y) grid; use provided ul_dm_cosine_potential
    base = ul_dm_cosine_potential((sim.X, sim.Y),
                                  amplitude_J=TEST_AMPLITUDE_J,
                                  m_phi_ev=TEST_MPHI_EV,
                                  phase0=0.0,
                                  v_dm=220e3,
                                  direction=0.0,
                                  spatial_modulation=True)
    # base should be callable: either base((X,Y),t) or base(t) that returns an array.
    def V_total(coords, t):
        Xc, Yc = coords
        # try coords + t call first
        try:
            V = base((Xc, Yc), t)
        except TypeError:
            V = base(t)
            # if scalar broadcast
            V = np.ones_like(Xc) * float(V)
        V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential)
        return V + V_env

    # Diagnostic: sample potential at center over a short time
    ts = np.linspace(0, min(1.0, TEST_T_TOTAL), 400)
    try:
        vals = np.array([V_total((sim.X, sim.Y), tt)[sim.ny//2, sim.nx//2] for tt in ts])
    except Exception as e:
        print("[WARN] Potential diagnostic failed:", e)
        vals = np.zeros_like(ts)
    # Save a small diagnostic figure
    plt.figure(figsize=(5,2))
    plt.plot(ts, vals)
    plt.xlabel("time [s]")
    plt.ylabel("V_dm+V_env at center [J]")
    plt.title("Potential diagnostic (center point)")
    plt.grid(True, ls='--', alpha=0.4)
    diag_path = PLOTS_DIR / "potential_diagnostic_with_dm.png"
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
    plt.figure(figsize=(5,2)); plt.plot(ts, vals); plt.xlabel("time [s]"); plt.ylabel("V_env at center [J]")
    plt.grid(True, ls='--', alpha=0.4); plt.title("Potential diagnostic (no DM)")
    plt.savefig(diag_path, bbox_inches='tight', dpi=140); plt.close()
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

    # save and plot time-domain for DM
    save_time_plot(times_dm, dp_dm, PLOTS_DIR / "phase_with_dm.png", title="Δφ(t) — With (visual-test) DM")
    freqs_dm, psd_dm = compute_psd(times_dm, dp_dm)
    save_psd_plot(freqs_dm, psd_dm, PLOTS_DIR / "psd_with_dm.png", f_dm=None, xlim_max=None)

    # 2) run no-DM (or load if exists)
    no_dm_file = RESULTS_DIR / "delta_phi_no_dm_visual.npz"
    if no_dm_file.exists():
        data = np.load(no_dm_file)
        times_no_dm, dp_no_dm = data["times"], data["delta_phi"]
        print("[INFO] Loaded existing no-DM visual run.")
        vals_no_dm = None
    else:
        times_no_dm, dp_no_dm, vals_no_dm, ts_no_dm = run_no_dm(TEST_T_TOTAL)

    # save and plot time-domain for no-DM
    save_time_plot(times_no_dm, dp_no_dm, PLOTS_DIR / "phase_no_dm.png", title="Δφ(t) — No DM")
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
