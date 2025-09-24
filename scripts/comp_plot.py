#!/usr/bin/env python3
"""
comp_plot.py

Run a BEC simulation with a boosted test DM mass (for visualization) and a
separate simulation without DM. Produce:
  1) RAW Δφ(t) plots with a SHARED y-axis (fair comparison)
  2) DETRENDED comparison plot (both runs on same axes)
  3) RESIDUAL plot: (detrended_withDM - detrended_noDM)
  4) PSDs for both runs

Also fixes the end-of-series "drop" by using an edge-safe running average.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d  # edge-safe smoothing

# ── Project paths ──────────────────────────────────────────────────────────────
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

# ── User visual/debug parameters (boosted to be visible quickly) ───────────────
TEST_MPHI_EV     = 1e-12      # eV
TEST_AMPLITUDE_J = 1e-18      # J
TEST_T_TOTAL     = 7.0        # s

# Simulation grid/time parameters (match your normal run)
NX = NY = 128
DX = DY = 1.0
DT = 1e-3
G = 1e-52
M_PARTICLE = 1.6726219e-27

# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_psd(times, signal):
    dt = float(times[1] - times[0])
    fs = 1.0 / dt
    nperseg = min(2048, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd

def running_avg_safe(x, N):
    """Edge-safe running average using nearest padding (no end drop)."""
    N = max(3, int(N))
    return uniform_filter1d(x, size=N, mode="nearest")

def detrend_linear(times, y):
    """Fit y ≈ a*t + b and return (y - trend), (a, b)."""
    p = np.polyfit(times, y, 1)
    trend = np.polyval(p, times)
    return y - trend, (p[0], p[1])

def save_time_plot(times, signal, out_path, title="Phase vs time",
                   y_limits=None, avg_window=None, zoom_first_sec=False):
    """Save a time plot (edge-safe average; optional fixed y-limits)."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, signal, alpha=0.8, label="Δφ(t)")
    # edge-safe running average
    N = avg_window if (avg_window is not None) else max(3, len(signal)//200)
    avg = running_avg_safe(signal, N)
    ax.plot(times, avg, linewidth=1.8, label=f"running avg (N={N})")

    if y_limits is not None:
        ax.set_ylim(*y_limits)
        ax.set_autoscale_on(False)
    else:
        ymin, ymax = np.min(signal), np.max(signal)
        pad = 0.1 * (ymax - ymin) if ymax > ymin else 1.0
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δφ [rad]")
    ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Optional zoom (keeps same y-range only if y_limits provided)
    if zoom_first_sec:
        fig2, ax2 = plt.subplots(figsize=(6, 2.5))
        ax2.plot(times, signal, alpha=0.8)
        ax2.plot(times, avg, linewidth=1.8)
        ax2.set_xlim(0, min(1.0, times[-1]))
        if y_limits is not None:
            ax2.set_ylim(*y_limits)
            ax2.set_autoscale_on(False)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Δφ [rad]")
        ax2.set_title(title + " (zoom)")
        ax2.grid(True, ls='--', alpha=0.4)
        fig2.tight_layout()
        out_zoom = out_path.with_name(out_path.stem + "_zoom.png")
        fig2.savefig(out_zoom, dpi=150, bbox_inches='tight')
        plt.close(fig2)

def save_psd_plot(freqs, psd, out_path, f_dm=None, xlim_max=None):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.semilogy(freqs, psd)
    if f_dm is not None:
        ax.axvline(f_dm, color='red', linestyle='--', label=f"f_DM = {f_dm:.3e} Hz")
        ax.legend()
    if xlim_max:
        ax.set_xlim(0, xlim_max)
    psd_nonzero = psd[np.isfinite(psd) & (psd > 0)]
    if psd_nonzero.size:
        vlow = np.percentile(psd_nonzero, 1.0)
        vhigh = np.percentile(psd_nonzero, 99.0)
        if vhigh > vlow:
            ax.set_ylim(max(vlow*0.1, 1e-40), vhigh*10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(True, ls='--', alpha=0.4, which='both')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ── Sim runners ────────────────────────────────────────────────────────────────
def run_with_dm_test():
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=TEST_T_TOTAL)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    V_dm = ul_dm_cosine_potential((sim.X, sim.Y),
                                  amplitude_J=TEST_AMPLITUDE_J,
                                  m_phi_ev=TEST_MPHI_EV,
                                  phase0=0.0, v_dm=220e3, direction=0.0,
                                  spatial_modulation=True)

    V_env = create_environment_potential(
        sim.X, sim.Y, neutron_star_potential,
        center_offset=(0.0, 0.0), R_ns=1.0
    )

    def V_total(coords, t):
        return V_dm(coords, t) + V_env

    # small potential diagnostic at center
    ts = np.linspace(0, min(1.0, TEST_T_TOTAL), 400)
    try:
        vals = np.array([V_total((sim.X, sim.Y), tt)[sim.ny//2, sim.nx//2] for tt in ts])
    except Exception:
        vals = np.zeros_like(ts)
    diag_path = PLOTS_DIR / "potential_diagnostic_with_dm.png"
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(ts, vals); ax.set_xlabel("time [s]"); ax.set_ylabel("V_dm+V_env [J]")
    ax.set_title("Potential diagnostic (center) — with DM")
    ax.grid(True, ls='--', alpha=0.4); fig.tight_layout()
    fig.savefig(diag_path, dpi=140, bbox_inches='tight'); plt.close(fig)

    print("[INFO] Running WITH-DM test simulation (boosted mass)...")
    res = sim.run(V_function=V_total, snapshot_interval=0)
    out_file = RESULTS_DIR / "delta_phi_test_visual.npz"
    np.savez_compressed(out_file, times=res.times, delta_phi=res.delta_phi)
    print("[INFO] Saved boosted-DM timeseries to:", out_file)
    return res.times, res.delta_phi

def run_no_dm(t_total):
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=t_total)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential)

    def V_total(coords, t):
        return V_env

    ts = np.linspace(0, min(1.0, t_total), 200)
    vals = np.array([V_total((sim.X, sim.Y), tt)[sim.ny//2, sim.nx//2] for tt in ts])
    diag_path = PLOTS_DIR / "potential_diagnostic_no_dm.png"
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(ts, vals); ax.set_xlabel("time [s]"); ax.set_ylabel("V_env [J]")
    ax.set_title("Potential diagnostic (center) — no DM")
    ax.grid(True, ls='--', alpha=0.4); fig.tight_layout()
    fig.savefig(diag_path, dpi=140, bbox_inches='tight'); plt.close(fig)

    print("[INFO] Running NO-DM simulation...")
    res = sim.run(V_function=V_total, snapshot_interval=0)
    out_file = RESULTS_DIR / "delta_phi_no_dm_visual.npz"
    np.savez_compressed(out_file, times=res.times, delta_phi=res.delta_phi)
    print("[INFO] Saved no-DM timeseries to:", out_file)
    return res.times, res.delta_phi

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # WITH DM (re-use if exists)
    boosted_file = RESULTS_DIR / "delta_phi_test_visual.npz"
    if boosted_file.exists():
        d = np.load(boosted_file)
        times_dm, dp_dm = d["times"], d["delta_phi"]
        print("[INFO] Loaded existing boosted DM visual run.")
    else:
        times_dm, dp_dm = run_with_dm_test()

    # NO DM (re-use if exists)
    no_dm_file = RESULTS_DIR / "delta_phi_no_dm_visual.npz"
    if no_dm_file.exists():
        d = np.load(no_dm_file)
        times_no_dm, dp_no_dm = d["times"], d["delta_phi"]
        print("[INFO] Loaded existing no-DM visual run.")
    else:
        times_no_dm, dp_no_dm = run_no_dm(TEST_T_TOTAL)

    # ── 1) RAW plots with SHARED y-axis ────────────────────────────────────────
    combined_min = float(min(dp_dm.min(), dp_no_dm.min()))
    combined_max = float(max(dp_dm.max(), dp_no_dm.max()))
    pad = 0.1 * (combined_max - combined_min) if combined_max > combined_min else 1.0
    y_limits_raw = (combined_min - pad, combined_max + pad)

    save_time_plot(times_dm, dp_dm, PLOTS_DIR / "phase_with_dm_raw_shared.png",
                   title="Δφ(t) — With (visual-test) DM (raw, shared y)",
                   y_limits=y_limits_raw, avg_window=35, zoom_first_sec=False)

    save_time_plot(times_no_dm, dp_no_dm, PLOTS_DIR / "phase_no_dm_raw_shared.png",
                   title="Δφ(t) — No DM (raw, shared y)",
                   y_limits=y_limits_raw, avg_window=35, zoom_first_sec=False)

    # ── 2) DETREND both and plot together (best apples-to-apples view) ────────
    detr_dm, coef_dm = detrend_linear(times_dm, dp_dm)
    detr_no, coef_no = detrend_linear(times_no_dm, dp_no_dm)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times_dm, running_avg_safe(detr_dm, 35), label="With DM (detrended)")
    ax.plot(times_no_dm, running_avg_safe(detr_no, 35), label="No DM (detrended)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δφ detrended [rad]")
    ax.set_title("Detrended Δφ(t): With DM vs No DM")
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase_detrended_overlay.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 3) RESIDUAL = (With DM detrended) - (No DM detrended) ─────────────────
    # (Assumes same time grid lengths; if not, we interpolate to the shorter.)
    if len(times_dm) == len(times_no_dm) and np.allclose(times_dm, times_no_dm):
        times_res = times_dm
        resid = detr_dm - detr_no
    else:
        # Align to the shorter range via interpolation
        tmin = max(times_dm.min(), times_no_dm.min())
        tmax = min(times_dm.max(), times_no_dm.max())
        times_res = np.linspace(tmin, tmax, min(len(times_dm), len(times_no_dm)))
        resid = np.interp(times_res, times_dm, detr_dm) - np.interp(times_res, times_no_dm, detr_no)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times_res, running_avg_safe(resid, 35))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual [rad]")
    ax.set_title("Residual (With DM detrended − No DM detrended)")
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase_residual_detrended.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 4) PSDs (unchanged) ───────────────────────────────────────────────────
    freqs_dm, psd_dm = compute_psd(times_dm, dp_dm)
    save_psd_plot(freqs_dm, psd_dm, PLOTS_DIR / "psd_with_dm.png", f_dm=None, xlim_max=None)

    freqs_no_dm, psd_no_dm = compute_psd(times_no_dm, dp_no_dm)
    save_psd_plot(freqs_no_dm, psd_no_dm, PLOTS_DIR / "psd_no_dm.png", f_dm=None, xlim_max=None)

    # Numeric dump for later analysis
    np.savez_compressed(PLOTS_DIR / "psd_visual_data.npz",
                        freqs_dm=freqs_dm, psd_dm=psd_dm,
                        freqs_no_dm=freqs_no_dm, psd_no_dm=psd_no_dm)

    print("[DONE] Generated:")
    print(" - phase_with_dm_raw_shared.png, phase_no_dm_raw_shared.png")
    print(" - phase_detrended_overlay.png, phase_residual_detrended.png")
    print(" - psd_with_dm.png, psd_no_dm.png")
    print("Potential diagnostics saved as potential_diagnostic_with_dm.png and potential_diagnostic_no_dm.png")

if __name__ == "__main__":
    main()
