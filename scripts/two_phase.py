#!/usr/bin/env python3
"""
two_state.py  —  Two-state differential DM detection demo

Idea: run two otherwise-identical GPE simulations but scale the DM coupling
by c1 and c2 (e.g., 1.0 and 0.6). Use the *center absolute phase* from each
state and form Δφ_int = φ_center(c1) - φ_center(c2). A uniform ULDM field
does not cancel here if c1 != c2. We then show:
  • Detrended Δφ_int(t)
  • PSD of Δφ_int with f_DM marked
  • Lock-in amplitude at f_DM for Δφ_int
Requires BECSimulation.run() to return: times, delta_phi, center_phases, ref_phases
(as in the small patch we added earlier).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[1]
RESULTS_TS = BASE_DIR / "results" / "two_state" / "time_series"
PLOTS_DIR  = BASE_DIR / "results" / "two_state" / "plots"
RESULTS_TS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.bec_simulation import BECSimulation
from src.environment   import create_environment_potential, neutron_star_potential
from src.dark_matter   import ul_dm_cosine_potential

# ── Constants & params ────────────────────────────────────────────────────────
HBAR = 1.054571817e-34
EV2J = 1.602176634e-19

# Visual demo parameters (keep physics consistent; amplitude boosted to see effect)
M_PHI_EV       = 1e-12          # ULDM mass (eV) → f_DM ≈ 242 Hz
AMPLITUDE_J    = 1e-18          # Base DM potential amplitude (J)
COUPLING_1     = 1.0            # c1 (dimensionless DM coupling scale)
COUPLING_2     = 0.6            # c2 (≠ c1)
T_TOTAL        = 7.0            # seconds (increase for higher SNR)
NX = NY = 128
DX = DY = 1.0
DT = 1e-3
G  = 1e-52
M_PARTICLE = 1.6726219e-27

FORCE_RERUN = True   # set True to recompute; False to reuse saved NPZ

# ── Small helpers ─────────────────────────────────────────────────────────────
def running_avg_safe(x, N):
    N = max(3, int(N))
    return uniform_filter1d(np.asarray(x), size=N, mode="nearest")

def detrend_linear(t, y):
    p = np.polyfit(t, y, 1)
    return y - np.polyval(p, t), p

def compute_psd(times, y):
    dt = float(times[1] - times[0])
    fs = 1.0 / dt
    nperseg = min(2048, len(y))
    f, Pxx = welch(y, fs=fs, nperseg=nperseg)
    return f, Pxx

def lock_in_amplitude(times, y, f0, tau=0.2):
    t = np.asarray(times); y = np.asarray(y)
    w = 2*np.pi*f0
    I = y * np.cos(w*t)
    Q = y * np.sin(w*t)
    N = max(3, int(round(tau / (t[1]-t[0]))))
    I_f = running_avg_safe(I, N)
    Q_f = running_avg_safe(Q, N)
    return np.sqrt(I_f**2 + Q_f**2)

def save_time_plot(t, y, path, title, ylabel="[rad]", avgN=35):
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, y, alpha=0.85, label="signal")
    ax.plot(t, running_avg_safe(y, avgN), lw=1.8, label=f"running avg (N={avgN})")
    ymin, ymax = np.min(y), np.max(y)
    pad = 0.1 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.4); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)

def save_psd_plot(f, P, out_path, f_dm=None, title="PSD", ylabel="PSD [arb.]"):
    fig, ax = plt.subplots(figsize=(8,3.6))
    ax.semilogy(f, P)
    if f_dm is not None:
        ax.axvline(f_dm, color="red", ls="--", label=f"f_DM ≈ {f_dm:.2f} Hz")
        ax.legend()
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.4, which='both'); fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)

# ── One run with a given coupling scale c (returns center phase) ──────────────
def run_state(coupling_scale, tag):
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=T_TOTAL)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    V_dm  = ul_dm_cosine_potential((sim.X, sim.Y),
                                   amplitude_J=AMPLITUDE_J * coupling_scale,
                                   m_phi_ev=M_PHI_EV,
                                   phase0=0.0, v_dm=220e3, direction=0.0,
                                   spatial_modulation=True)
    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential,
                                         center_offset=(0.0,0.0), R_ns=1.0)

    def V_total(coords, t):
        return V_dm(coords, t) + V_env

    print(f"[INFO] Running state {tag} with coupling c={coupling_scale:.3f} …")
    res = sim.run(V_function=V_total, snapshot_interval=0)

    # Save raw time series (center_phases needed!)
    np.savez_compressed(RESULTS_TS / f"two_state_{tag}.npz",
                        times=res.times,
                        center_phases=res.center_phases,
                        ref_phases=res.ref_phases,
                        delta_phi=res.delta_phi)
    return res

# ── Main: run both states, build Δφ_int, and plot detection panels ───────────
def main():
    f_dm = (M_PHI_EV * EV2J / HBAR) / (2*np.pi)

    f1 = RESULTS_TS / "two_state_c1.npz"
    f2 = RESULTS_TS / "two_state_c2.npz"

    if FORCE_RERUN or (not f1.exists()) or (not f2.exists()):
        res1 = run_state(COUPLING_1, "c1")
        res2 = run_state(COUPLING_2, "c2")
    else:
        d1 = np.load(f1); d2 = np.load(f2)
        # pack as simple objects
        res1 = type("R", (), {})()
        res2 = type("R", (), {})()
        res1.times = d1["times"]; res1.center_phases = d1["center_phases"]; res1.delta_phi = d1["delta_phi"]
        res2.times = d2["times"]; res2.center_phases = d2["center_phases"]; res2.delta_phi = d2["delta_phi"]
        print("[INFO] Loaded existing two-state NPZ files.")

    # Sanity: align times (interpolate to common grid if needed)
    if len(res1.times) == len(res2.times) and np.allclose(res1.times, res2.times):
        t = res1.times
        phi1 = res1.center_phases
        phi2 = res2.center_phases
    else:
        tmin = max(res1.times.min(), res2.times.min())
        tmax = min(res1.times.max(), res2.times.max())
        n    = min(len(res1.times), len(res2.times))
        t    = np.linspace(tmin, tmax, n)
        phi1 = np.interp(t, res1.times, res1.center_phases)
        phi2 = np.interp(t, res2.times, res2.center_phases)

    # Differential internal-state phase (common-mode cancels if couplings equal)
    dphi_int = phi1 - phi2

    # Detrend (remove slow environment drift)
    dphi_int_detr, _ = detrend_linear(t, dphi_int)

    # A) Detrended Δφ_int(t)
    save_time_plot(t, dphi_int_detr, PLOTS_DIR / "two_state_detrended.png",
                   title="Two-state differential phase (detrended)",
                   ylabel="Δφ_int detrended [rad]")

    # B) PSD of Δφ_int
    f, P = compute_psd(t, dphi_int_detr)
    save_psd_plot(f, P, PLOTS_DIR / "two_state_psd.png",
                  f_dm=f_dm, title="PSD of two-state Δφ_int")

    # C) Lock-in amplitude at f_DM
    amp = lock_in_amplitude(t, dphi_int_detr, f_dm, tau=0.2)
    save_time_plot(t, amp, PLOTS_DIR / "two_state_lockin.png",
                   title=f"Two-state lock-in amplitude at f_DM≈{f_dm:.2f} Hz",
                   ylabel="Lock-in amplitude [rad]")

    print("[DONE] Wrote two-state plots to:", PLOTS_DIR)

if __name__ == "__main__":
    main()
