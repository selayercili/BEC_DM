#!/usr/bin/env python3
# scripts/two_phase.py  — robust, debuggable two-state differential phase runner

from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d

# =========================
# Project paths (src/scripts)
# =========================
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
SRC  = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.bec_simulation import BECSimulation
from src.environment import create_environment_potential, neutron_star_potential
from src.dark_matter import ul_dm_cosine_potential

# =========================
# Output directories
# =========================
RESULTS_TS = ROOT / "results" / "two_state" / "time_series"
PLOTS_DIR  = ROOT / "results" / "two_state" / "plots"
RESULTS_TS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Global constants / params
# =========================
HBAR = 1.054571817e-34
EV2J = 1.602176634e-19

# ---- DM parameters (make first run clearly visible; dial back later)
M_PHI_EV     = 1e-12     # ULDM mass [eV] → f_DM ≈ 242 Hz
DM_AMPLITUDE = 5e-17     # base DM potential amplitude [J] (debug-friendly)

# ---- Couplings for the two states (opposite sign = large Δφ)
COUPLING_1 =  1.0
COUPLING_2 = -1.0

# ---- Environment scaling for numerical conditioning
ENV_SCALE   = 1e-30      # multiply entire environment by this (debug)
ENV_REMOVE_DC = True     # subtract env center value (removes huge DC safely)

# ---- GPE / grid
NX = NY = 128
DX = DY = 1.0
DT = 1e-3
T_TOTAL = 20.0
G  = 1e-52
M_PARTICLE = 1.6726219e-27

FORCE_RERUN = True
DEBUG = True

# =========================
# Helpers
# =========================
def running_avg(x, N): 
    return uniform_filter1d(np.asarray(x), size=max(3, int(N)), mode="nearest")

def detrend_linear(t, y):
    p = np.polyfit(t, y, 1)
    return y - np.polyval(p, t), p

def compute_psd(times, y):
    times = np.asarray(times); y = np.asarray(y)
    if len(times) < 16 or len(y) < 16:
        return np.array([]), np.array([])
    dt = float(times[1] - times[0])
    fs = 1.0 / dt
    f, Pxx = welch(y, fs=fs, nperseg=min(2048, len(y)))
    return f, Pxx

def lock_in_amplitude(times, y, f0, tau=0.2):
    t = np.asarray(times); y = np.asarray(y)
    if len(t) < 4:
        return np.zeros_like(t)
    w = 2*np.pi*f0
    N = int(max(3, tau / (t[1]-t[0])))
    I = running_avg(y * np.cos(w*t), N)
    Q = running_avg(y * np.sin(w*t), N)
    return np.sqrt(I**2 + Q**2)

def save_time_plot(t, y, path, title, ylabel="[rad]", avgN=35):
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, y, alpha=0.9, label="signal")
    if len(y) >= avgN:
        ax.plot(t, running_avg(y, avgN), lw=1.8, label=f"running avg (N={avgN})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.4); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)

def save_psd_plot(f, P, path, f_dm=None, title="PSD", ylabel="PSD [arb.]"):
    fig, ax = plt.subplots(figsize=(8,3.6))
    if len(f) > 0:
        ax.semilogy(f, P)
    else:
        ax.text(0.5, 0.5, "PSD not computed (too few/degenerate samples)",
                ha='center', va='center')
    if f_dm is not None and len(f) > 0:
        ax.axvline(f_dm, color="red", ls="--", label=f"f_DM ≈ {f_dm:.2f} Hz"); ax.legend()
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.4, which='both'); fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)

def probe_dm_and_env(sim, V_dm_callable, V_env, f_dm):
    """Print quick checks to ensure signals are present before running GPE."""
    t0 = 0.0
    tQ = 0.25 / f_dm
    tH = 0.5  / f_dm
    center = (sim.ny//2, sim.nx//2)

    Vdm_t0 = V_dm_callable((sim.X, sim.Y), t0)
    Vdm_tQ = V_dm_callable((sim.X, sim.Y), tQ)
    Vdm_tH = V_dm_callable((sim.X, sim.Y), tH)

    stats = {
        "V_env_min": float(np.min(V_env)),
        "V_env_max": float(np.max(V_env)),
        "V_env_center": float(V_env[center]),
        "V_dm_min_t0": float(np.min(Vdm_t0)),
        "V_dm_max_t0": float(np.max(Vdm_t0)),
        "V_dm_center_t0": float(Vdm_t0[center]),
        "V_dm_center_tQ": float(Vdm_tQ[center]),
        "V_dm_center_tH": float(Vdm_tH[center]),
        "V_dm_span_t0": float(np.max(Vdm_t0) - np.min(Vdm_t0)),
    }
    print("\n[DEBUG] Environment & DM potential probes:")
    for k,v in stats.items():
        print(f"  - {k}: {v:.3e}")
    return stats

def summarize_phase(name, t, phi):
    phi = np.asarray(phi)
    dphi = np.diff(phi) if len(phi) > 1 else np.array([0.0])
    trend = np.polyfit(t, phi, 1) if len(t) >= 2 else [np.nan, np.nan]
    out = {
        "len": int(len(phi)),
        "phi_min": float(np.min(phi)) if len(phi) else np.nan,
        "phi_max": float(np.max(phi)) if len(phi) else np.nan,
        "phi_std": float(np.std(phi)) if len(phi) else np.nan,
        "dphi_std": float(np.std(dphi)) if len(dphi) else np.nan,
        "slope_rad_per_s": float(trend[0]) if len(t) >= 2 else np.nan
    }
    print(f"\n[DEBUG] {name} phase summary:")
    for k,v in out.items():
        print(f"  - {k}: {v:.6e}")
    return out

# =========================
# Single-state run
# =========================
def run_state(coupling_scale: float, tag: str, f_dm: float,
              dm_amplitude_j: float, env_scale: float):
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=T_TOTAL)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    # Build DM potential callable (time-dependent)
    V_dm = ul_dm_cosine_potential((sim.X, sim.Y),
                                  amplitude_J=dm_amplitude_j * coupling_scale,
                                  m_phi_ev=M_PHI_EV,
                                  phase0=0.0, v_dm=220e3, direction=0.0,
                                  spatial_modulation=True)

    # Environment (static array)
    V_env = create_environment_potential(sim.X, sim.Y,
                                         neutron_star_potential,
                                         center_offset=(0.0, 0.0),
                                         R_ns=1.0)

    # Remove huge DC and scale for numerical conditioning (debug mode)
    if ENV_REMOVE_DC:
        V_env = V_env - float(V_env[sim.ny//2, sim.nx//2])
    if env_scale is not None:
        V_env = V_env * float(env_scale)

    if DEBUG:
        probe_dm_and_env(sim, V_dm, V_env, f_dm)
        center = (sim.ny//2, sim.nx//2)
        print("[CHECK] V_env_center=", float(V_env[center]))
        print("[CHECK] V_dm_center@t0=", float(V_dm((sim.X, sim.Y), 0.0)[center]))

    def V_total(coords, t):
        # IMPORTANT: time-dependent V must be called each step inside BECSimulation
        return V_dm(coords, t) + V_env

    print(f"[INFO] Running state {tag} with coupling c={coupling_scale:.3f} …")
    res = sim.run(V_function=V_total, snapshot_interval=0)

    # Save raw series
    np.savez_compressed(RESULTS_TS / f"two_state_{tag}.npz",
                        times=res.times,
                        center_phases=res.center_phases,
                        ref_phases=res.ref_phases,
                        delta_phi=res.delta_phi)
    return res

# =========================
# Main
# =========================
def main():
    f_dm = (M_PHI_EV * EV2J / HBAR) / (2*np.pi)

    # run fresh (debug)
    res1 = run_state(COUPLING_1, "c1", f_dm, DM_AMPLITUDE, ENV_SCALE)
    res2 = run_state(COUPLING_2, "c2", f_dm, DM_AMPLITUDE, ENV_SCALE)

    t    = np.asarray(res1.times)
    phi1 = np.unwrap(np.asarray(res1.center_phases, dtype=float))
    phi2 = np.unwrap(np.asarray(res2.center_phases, dtype=float))

    # Raw plots (see if runs diverge)
    save_time_plot(t, phi1, PLOTS_DIR / "phi1_raw.png", "φ1 center phase (raw)")
    save_time_plot(t, phi2, PLOTS_DIR / "phi2_raw.png", "φ2 center phase (raw)")

    dphi_raw = phi1 - phi2
    save_time_plot(t, dphi_raw, PLOTS_DIR / "dphi_raw.png", "Δφ_int (raw)")

    # Summaries
    s1 = summarize_phase("phi1", t, phi1)
    s2 = summarize_phase("phi2", t, phi2)
    sR = summarize_phase("dphi_raw", t, dphi_raw)

    # Detrend & analyze
    dphi_detr, trend = detrend_linear(t, dphi_raw)
    save_time_plot(t, dphi_detr, PLOTS_DIR / "two_state_detrended.png",
                   title="Two-state differential phase (detrended)",
                   ylabel="Δφ_int detrended [rad]")

    f, P = compute_psd(t, dphi_detr)
    save_psd_plot(f, P, PLOTS_DIR / "two_state_psd.png",
                  f_dm=f_dm, title="PSD of two-state Δφ_int")

    amp = lock_in_amplitude(t, dphi_detr, f_dm, tau=0.2)
    save_time_plot(t, amp, PLOTS_DIR / "two_state_lockin.png",
                   title=f"Two-state lock-in amplitude at f_DM≈{f_dm:.2f} Hz",
                   ylabel="Lock-in amplitude [rad]")

    summary = {
        "grid": {"nx": NX, "ny": NY, "dx": DX, "dy": DY},
        "sim":  {"dt": DT, "t_total": T_TOTAL, "n_steps": int(round(T_TOTAL/DT))},
        "dm":   {"m_phi_ev": M_PHI_EV, "amplitude_J": DM_AMPLITUDE,
                 "coupling_1": COUPLING_1, "coupling_2": COUPLING_2,
                 "f_dm_hz": f_dm},
        "env":  {"scale": ENV_SCALE, "remove_dc": ENV_REMOVE_DC},
        "phi1": s1, "phi2": s2, "dphi_raw": sR,
        "detrend_slope_rad_per_s": float(trend[0]),
        "dphi_detr_std": float(np.std(dphi_detr)),
        "psd_points": int(len(f)),
        "notes": "Opposite couplings and scaled/DC-removed environment to avoid numerical erasure."
    }
    with open(RESULTS_TS / "debug_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[DEBUG] Summary:")
    print(json.dumps(summary, indent=2))
    print("\n[DONE] Plots →", PLOTS_DIR)
    print("[DONE] Debug summary →", RESULTS_TS / "debug_summary.json")

if __name__ == "__main__":
    main()
