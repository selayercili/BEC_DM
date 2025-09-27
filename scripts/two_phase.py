#!/usr/bin/env python3
# scripts/two_phase.py

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d

# ---- project paths (works from scripts/) ------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]                 # repo root
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.bec_simulation import BECSimulation
from src.environment import create_environment_potential, neutron_star_potential
from src.dark_matter import ul_dm_cosine_potential

# ---- outputs ----------------------------------------------------------------
RESULTS_TS = PROJECT_ROOT / "results" / "two_state" / "time_series"
PLOTS_DIR  = PROJECT_ROOT / "results" / "two_state" / "plots"
RESULTS_TS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- constants / params ------------------------------------------------------
HBAR = 1.054571817e-34
EV2J = 1.602176634e-19

# DM & sim settings (choose visible-but-sane demo values)
M_PHI_EV    = 1e-12        # ULDM “mass” (eV) ⇒ f_DM ≈ (m*eV2J/ħ)/(2π)
AMPLITUDE_J = 1e-18        # base DM potential amplitude (J)
COUPLING_1  = 1.00         # different couplings to avoid perfect cancellation
COUPLING_2  = 0.60

NX = NY = 128
DX = DY = 1.0
DT = 1e-3
T_TOTAL = 7.0
G  = 1e-52
M_PARTICLE = 1.6726219e-27

FORCE_RERUN = True

# ---- helpers ----------------------------------------------------------------
def running_avg(x, N): return uniform_filter1d(np.asarray(x), size=max(3, int(N)), mode="nearest")

def detrend_linear(t, y):
    p = np.polyfit(t, y, 1)
    return y - np.polyval(p, t), p

def compute_psd(times, y):
    dt = float(times[1] - times[0])
    fs = 1.0 / dt
    f, Pxx = welch(y, fs=fs, nperseg=min(2048, len(y)))
    return f, Pxx

def lock_in_amplitude(times, y, f0, tau=0.2):
    t = np.asarray(times); y = np.asarray(y)
    w = 2*np.pi*f0
    I = running_avg(y * np.cos(w*t), int(max(3, tau / (t[1]-t[0]))))
    Q = running_avg(y * np.sin(w*t), int(max(3, tau / (t[1]-t[0]))))
    return np.sqrt(I**2 + Q**2)

def save_time_plot(t, y, path, title, ylabel="[rad]", avgN=35):
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, y, alpha=0.85, label="signal")
    ax.plot(t, running_avg(y, avgN), lw=1.8, label=f"running avg (N={avgN})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.4); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)

def save_psd_plot(f, P, path, f_dm=None, title="PSD", ylabel="PSD [arb.]"):
    fig, ax = plt.subplots(figsize=(8,3.6))
    ax.semilogy(f, P)
    if f_dm is not None:
        ax.axvline(f_dm, color="red", ls="--", label=f"f_DM ≈ {f_dm:.2f} Hz"); ax.legend()
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.4, which='both'); fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)

# ---- single-state runner -----------------------------------------------------
def run_state(coupling_scale, tag):
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=T_TOTAL)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    # DM potential: your function returns a callable V_dm((X,Y), t) -> ndarray
    V_dm = ul_dm_cosine_potential((sim.X, sim.Y),
                                  amplitude_J=AMPLITUDE_J * coupling_scale,
                                  m_phi_ev=M_PHI_EV,
                                  phase0=0.0, v_dm=220e3, direction=0.0,
                                  spatial_modulation=True)

    # Environment (static array): V_env has shape (ny, nx)
    # NOTE: R_ns kept small (1.0 in grid units) to avoid singularity on this domain
    V_env = create_environment_potential(sim.X, sim.Y,
                                         neutron_star_potential,
                                         center_offset=(0.0, 0.0),
                                         R_ns=1.0)

    def V_total(coords, t):
        # BECSimulation.run calls V_function with (coords, t); we sum DM (callable) + static env
        return V_dm(coords, t) + V_env

    print(f"[INFO] Running state {tag} with coupling c={coupling_scale:.3f} …")
    res = sim.run(V_function=V_total, snapshot_interval=0)

    np.savez_compressed(RESULTS_TS / f"two_state_{tag}.npz",
                        times=res.times,
                        center_phases=res.center_phases,
                        ref_phases=res.ref_phases,
                        delta_phi=res.delta_phi)
    return res

# ---- main: run both states, build Δφ, plot/PSD/lock-in ----------------------
def main():
    f1 = RESULTS_TS / "two_state_c1.npz"
    f2 = RESULTS_TS / "two_state_c2.npz"

    # f_DM from parameters (your ul_dm function doesn't return ω)
    f_dm = (M_PHI_EV * EV2J / HBAR) / (2*np.pi)

    if FORCE_RERUN or (not f1.exists()) or (not f2.exists()):
        res1 = run_state(COUPLING_1, "c1")
        res2 = run_state(COUPLING_2, "c2")
        t    = res1.times
        phi1 = np.unwrap(np.array(res1.center_phases))
        phi2 = np.unwrap(np.array(res2.center_phases))
    else:
        d1 = np.load(f1); d2 = np.load(f2)
        t    = d1["times"]
        phi1 = np.unwrap(np.array(d1["center_phases"]))
        phi2 = np.unwrap(np.array(d2["center_phases"]))

    dphi_int = phi1 - phi2
    dphi_int_detr, _ = detrend_linear(t, dphi_int)

    # A) time series (detrended)
    save_time_plot(t, dphi_int_detr, PLOTS_DIR / "two_state_detrended.png",
                   title="Two-state differential phase (detrended)",
                   ylabel="Δφ_int detrended [rad]")

    # B) PSD with f_DM marker
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
