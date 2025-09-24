#!/usr/bin/env python3
"""
comp_plot.py

Produces:
  A) RAW Δφ(t) (shared y) for with-DM and no-DM
  B) DETRENDED Δφ overlay and residual
  C) DETRENDED center-phase overlay (common-mode sensitive)
  D) Lock-in amplitude at f_DM for center-phase (with vs no DM)
Keeps physics intact; no changes to BECSimulation internals.
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

# ── Constants ─────────────────────────────────────────────────────────────────
HBAR = 1.054571817e-34
EV2J = 1.602176634e-19

# ── User visual/debug parameters ──────────────────────────────────────────────
TEST_MPHI_EV     = 1e-12      # eV → f_DM ≈ 242 Hz (resolvable at dt=1e-3 s)
TEST_AMPLITUDE_J = 1e-18      # stronger than physical to visualize
TEST_T_TOTAL     = 7.0        # s
NX = NY = 128
DX = DY = 1.0
DT = 1e-3
G = 1e-52
M_PARTICLE = 1.6726219e-27

FORCE_RERUN = False   # set True to always recompute, False to reuse NPZ if present

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_psd(times, signal):
    dt = float(times[1] - times[0])
    fs = 1.0 / dt
    nperseg = min(2048, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd

def running_avg_safe(x, N):
    N = max(3, int(N))
    return uniform_filter1d(x, size=N, mode="nearest")

def detrend_linear(times, y):
    p = np.polyfit(times, y, 1)
    trend = np.polyval(p, times)
    return y - trend, (p[0], p[1])

def save_time_plot(times, signal, out_path, title,
                   y_limits=None, avg_window=None):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, signal, alpha=0.8, label="signal")
    N = avg_window if (avg_window is not None) else max(3, len(signal)//200)
    avg = running_avg_safe(signal, N)
    ax.plot(times, avg, linewidth=1.8, label=f"running avg (N={N})")

    if y_limits is not None:
        ax.set_ylim(*y_limits); ax.set_autoscale_on(False)
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

def save_psd_plot(freqs, psd, out_path, f_dm=None, xlim_max=None):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.semilogy(freqs, psd)
    if f_dm is not None:
        ax.axvline(f_dm, color='red', linestyle='--', label=f"f_DM = {f_dm:.3e} Hz")
        ax.legend()
    if xlim_max: ax.set_xlim(0, xlim_max)
    psd_nonzero = psd[np.isfinite(psd) & (psd > 0)]
    if psd_nonzero.size:
        vlow  = np.percentile(psd_nonzero, 1.0)
        vhigh = np.percentile(psd_nonzero, 99.0)
        if vhigh > vlow: ax.set_ylim(max(vlow*0.1, 1e-40), vhigh*10)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    ax.grid(True, ls='--', alpha=0.4, which='both')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)

def lock_in_amplitude(times, y, f0, tau=0.2):
    """
    Simple lock-in: multiply by cos/sin(2π f0 t), low-pass via running average with window tau.
    Returns amplitude(t) = sqrt(I^2 + Q^2) after smoothing.
    """
    t = np.asarray(times)
    y = np.asarray(y)
    omega = 2*np.pi*f0
    cos_ref = np.cos(omega*t)
    sin_ref = np.sin(omega*t)
    I = y * cos_ref
    Q = y * sin_ref
    # smooth with window ~ tau seconds
    dt = t[1]-t[0]
    N = max(3, int(round(tau/dt)))
    I_f = running_avg_safe(I, N)
    Q_f = running_avg_safe(Q, N)
    return np.sqrt(I_f**2 + Q_f**2)

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
    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential,
                                         center_offset=(0.0, 0.0), R_ns=1.0)
    def V_total(coords, t): return V_dm(coords, t) + V_env

    print("[INFO] Running WITH-DM simulation…")
    res = sim.run(V_function=V_total, snapshot_interval=0)
    np.savez_compressed(RESULTS_DIR / "delta_phi_test_visual.npz",
                        times=res.times, delta_phi=res.delta_phi)
    return res

def run_no_dm(t_total):
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=t_total)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)
    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential)
    def V_total(coords, t): return V_env

    print("[INFO] Running NO-DM simulation…")
    res = sim.run(V_function=V_total, snapshot_interval=0)
    np.savez_compressed(RESULTS_DIR / "delta_phi_no_dm_visual.npz",
                        times=res.times, delta_phi=res.delta_phi)
    return res

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    f_dm = (TEST_MPHI_EV * EV2J / HBAR) / (2*np.pi)

    # Load or run
    dm_npz = RESULTS_DIR / "delta_phi_test_visual.npz"
    no_npz = RESULTS_DIR / "delta_phi_no_dm_visual.npz"

    if FORCE_RERUN or (not dm_npz.exists()):
        res_dm = run_with_dm_test()
    else:
        print("[INFO] Loaded existing WITH-DM NPZ.")
        d = np.load(dm_npz); res_dm = type("R", (), {})()
        res_dm.times, res_dm.delta_phi = d["times"], d["delta_phi"]
        # we don't have center/ref in NPZ; so recompute by running quickly if needed
        # but for simplicity we accept Δφ-only for now in cached mode
        res_dm.center_phases = None; res_dm.ref_phases = None

    if FORCE_RERUN or (not no_npz.exists()):
        res_no = run_no_dm(TEST_T_TOTAL)
    else:
        print("[INFO] Loaded existing NO-DM NPZ.")
        d = np.load(no_npz); res_no = type("R", (), {})()
        res_no.times, res_no.delta_phi = d["times"], d["delta_phi"]
        res_no.center_phases = None; res_no.ref_phases = None

    # ── A) RAW Δφ with shared y ────────────────────────────────────────────────
    combined_min = float(min(res_dm.delta_phi.min(), res_no.delta_phi.min()))
    combined_max = float(max(res_dm.delta_phi.max(), res_no.delta_phi.max()))
    pad = 0.1 * (combined_max - combined_min) if combined_max > combined_min else 1.0
    y_limits_raw = (combined_min - pad, combined_max + pad)

    save_time_plot(res_dm.times, res_dm.delta_phi,
                   PLOTS_DIR / "phase_with_dm_raw_shared.png",
                   "Δφ(t) — With DM (raw, shared y)",
                   y_limits=y_limits_raw, avg_window=35)

    save_time_plot(res_no.times, res_no.delta_phi,
                   PLOTS_DIR / "phase_no_dm_raw_shared.png",
                   "Δφ(t) — No DM (raw, shared y)",
                   y_limits=y_limits_raw, avg_window=35)

    # ── B) DETRENDED Δφ overlay + residual ────────────────────────────────────
    detr_dm, _ = detrend_linear(res_dm.times, res_dm.delta_phi)
    detr_no, _ = detrend_linear(res_no.times, res_no.delta_phi)

    # overlay
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(res_dm.times, running_avg_safe(detr_dm, 35), label="With DM (detrended)")
    ax.plot(res_no.times, running_avg_safe(detr_no, 35), label="No DM (detrended)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Δφ detrended [rad]")
    ax.set_title("Detrended Δφ(t): With DM vs No DM")
    ax.grid(True, ls='--', alpha=0.4); ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase_detrended_overlay.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # residual
    if len(res_dm.times) == len(res_no.times) and np.allclose(res_dm.times, res_no.times):
        times_res = res_dm.times
        resid = detr_dm - detr_no
    else:
        tmin = max(res_dm.times.min(), res_no.times.min())
        tmax = min(res_dm.times.max(), res_no.times.max())
        times_res = np.linspace(tmin, tmax, min(len(res_dm.times), len(res_no.times)))
        resid = np.interp(times_res, res_dm.times, detr_dm) - \
                np.interp(times_res, res_no.times, detr_no)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times_res, running_avg_safe(resid, 35))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Residual [rad]")
    ax.set_title("Residual (With DM detrended − No DM detrended)")
    ax.grid(True, ls='--', alpha=0.4); fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase_residual_detrended.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── C) DETRENDED center-phase overlay (common-mode sensitive) ─────────────
    # If we have center_phases (when FORCE_RERUN True), use them. Otherwise skip gracefully.
    if getattr(res_dm, "center_phases", None) is not None and getattr(res_no, "center_phases", None) is not None:
        c_dm, _ = detrend_linear(res_dm.times, res_dm.center_phases)
        c_no, _ = detrend_linear(res_no.times, res_no.center_phases)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(res_dm.times, running_avg_safe(c_dm, 35), label="With DM (center, detrended)")
        ax.plot(res_no.times, running_avg_safe(c_no, 35), label="No DM (center, detrended)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Center phase detrended [rad]")
        ax.set_title("Detrended center phase (common-mode)")
        ax.grid(True, ls='--', alpha=0.4); ax.legend(); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "center_phase_detrended_overlay.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        print("[NOTE] center_phases not available from NPZ; set FORCE_RERUN=True to compute common-mode plots.")

    # ── D) Lock-in amplitude at f_DM for center-phase ─────────────────────────
    if getattr(res_dm, "center_phases", None) is not None and getattr(res_no, "center_phases", None) is not None:
        # remove slow drift first so lock-in sees the oscillation
        c_dm_detr, _ = detrend_linear(res_dm.times, res_dm.center_phases)
        c_no_detr, _ = detrend_linear(res_no.times, res_no.center_phases)
        amp_dm = lock_in_amplitude(res_dm.times, c_dm_detr, f_dm, tau=0.2)
        amp_no = lock_in_amplitude(res_no.times, c_no_detr, f_dm, tau=0.2)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(res_dm.times, amp_dm, label="With DM")
        ax.plot(res_no.times, amp_no, label="No DM")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Lock-in amplitude [rad]")
        ax.set_title(f"Lock-in amplitude at f_DM≈{f_dm:.1f} Hz (center phase)")
        ax.grid(True, ls='--', alpha=0.4); ax.legend(); fig.tight_layout()
        fig.savefig(PLOTS_DIR / "center_phase_lockin.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── PSDs of Δφ (as before) ────────────────────────────────────────────────
    freqs_dm, psd_dm = compute_psd(res_dm.times, res_dm.delta_phi)
    save_psd_plot(freqs_dm, psd_dm, PLOTS_DIR / "psd_with_dm.png", f_dm=f_dm, xlim_max=None)
    freqs_no, psd_no = compute_psd(res_no.times, res_no.delta_phi)
    save_psd_plot(freqs_no, psd_no, PLOTS_DIR / "psd_no_dm.png", f_dm=f_dm, xlim_max=None)

    # store PSD arrays too
    np.savez_compressed(PLOTS_DIR / "psd_visual_data.npz",
                        freqs_dm=freqs_dm, psd_dm=psd_dm,
                        freqs_no=freqs_no, psd_no=psd_no)

    print("[DONE] Wrote:")
    print("  phase_with_dm_raw_shared.png, phase_no_dm_raw_shared.png")
    print("  phase_detrended_overlay.png, phase_residual_detrended.png")
    print("  center_phase_detrended_overlay.png (if FORCE_RERUN), center_phase_lockin.png (if FORCE_RERUN)")
    print("  psd_with_dm.png, psd_no_dm.png")

if __name__ == "__main__":
    main()
