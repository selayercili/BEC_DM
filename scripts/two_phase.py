#!/usr/bin/env python3
"""
Two-state differential DM detection (works with src/ + scripts/ layout)

Runs two TD-GPE simulations with different DM couplings c1≠c2.
Signal: Δφ_int(t) = φ_center(c1) - φ_center(c2).
Plots: (A) detrended Δφ_int, (B) PSD with f_DM marker, (C) lock-in amplitude.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d

# ── Project paths (src/ + scripts/) ──────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]            # …/project/
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.bec_simulation import BECSimulation   # your class

# Optional: use your own environment/dark_matter if present
HAVE_ENV = (SRC_DIR / "environment.py").exists()
HAVE_DM  = (SRC_DIR / "dark_matter.py").exists()
if HAVE_ENV:
    try:
        from src.environment import create_environment_potential, neutron_star_potential
    except Exception:
        HAVE_ENV = False
if HAVE_DM:
    try:
        from src.dark_matter import ul_dm_cosine_potential
    except Exception:
        HAVE_DM = False

# ── Output dirs ──────────────────────────────────────────────────────────────
RESULTS_TS = PROJECT_ROOT / "results" / "two_state" / "time_series"
PLOTS_DIR  = PROJECT_ROOT / "results" / "two_state" / "plots"
RESULTS_TS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants / demo params ──────────────────────────────────────────────────
HBAR = 1.054571817e-34
EV2J = 1.602176634e-19
M_PHI_EV    = 1e-12        # ULDM mass (eV) → f_DM ≈ (m_ev*EV2J/ħ)/(2π) ≈ 242 Hz
AMPLITUDE_J = 1e-18        # base DM potential amplitude (J)
COUPLING_1  = 1.0
COUPLING_2  = 0.6

NX = NY = 128
DX = DY = 1.0
DT = 1e-3
T_TOTAL = 7.0
G  = 1e-52
M_PARTICLE = 1.6726219e-27

FORCE_RERUN = True

# ── Fallback potentials if your modules are absent ───────────────────────────
def _harmonic_trap(X, Y, k_trap=1e-33):
    return 0.5 * k_trap * (X*X + Y*Y)

def _make_uldm_cosine_factory(X, Y, amplitude_J, m_phi_ev, phase0=0.0,
                              v_dm=220e3, direction=0.0, spatial_modulation=True):
    omega = (m_phi_ev * EV2J) / HBAR
    # tiny spatial phase so the two states don't perfectly common-mode cancel
    c = 299792458.0
    k_eff = (omega * v_dm) / (c*c)
    kx, ky = k_eff*np.cos(direction), k_eff*np.sin(direction)
    phi_xy = (kx*X + ky*Y) if spatial_modulation else 0.0
    def V_dm(coords, t):
        return amplitude_J * np.cos(omega*t + phi_xy + phase0)
    return V_dm, omega

# ── Small helpers ────────────────────────────────────────────────────────────
def running_avg(x, N): return uniform_filter1d(np.asarray(x), size=max(3,int(N)), mode="nearest")
def detrend_linear(t,y):
    p = np.polyfit(t,y,1)
    return y - np.polyval(p,t), p
def compute_psd(times,y):
    dt = float(times[1]-times[0]); fs = 1.0/dt
    f,Pxx = welch(y, fs=fs, nperseg=min(2048, len(y)))
    return f,Pxx
def lock_in_amplitude(times,y,f0,tau=0.2):
    t = np.asarray(times); y=np.asarray(y); w=2*np.pi*f0
    I = running_avg(y*np.cos(w*t), int(tau/(t[1]-t[0])))
    Q = running_avg(y*np.sin(w*t), int(tau/(t[1]-t[0])))
    return np.sqrt(I**2 + Q**2)
def save_time_plot(t,y,path,title,ylabel="[rad]",avgN=35):
    fig,ax=plt.subplots(figsize=(8,3))
    ax.plot(t,y,alpha=0.85,label="signal")
    ax.plot(t,running_avg(y,avgN),lw=1.8,label=f"running avg (N={avgN})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True,ls='--',alpha=0.4); ax.legend(); fig.tight_layout()
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)
def save_psd_plot(f,P,path,f_dm=None,title="PSD",ylabel="PSD [arb.]"):
    fig,ax=plt.subplots(figsize=(8,3.6))
    ax.semilogy(f,P)
    if f_dm is not None:
        ax.axvline(f_dm,color="red",ls="--",label=f"f_DM ≈ {f_dm:.2f} Hz"); ax.legend()
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True,ls='--',alpha=0.4,which='both'); fig.tight_layout()
    fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)

# ── Single-state run ─────────────────────────────────────────────────────────
def run_state(coupling_scale, tag):
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=T_TOTAL)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    # Environment potential
    if HAVE_ENV:
        V_env = create_environment_potential(sim.X, sim.Y, kind=neutron_star_potential)
    else:
        V_env = _harmonic_trap(sim.X, sim.Y, k_trap=1e-33)

    # DM potential
    if HAVE_DM:
        V_dm_callable, omega = ul_dm_cosine_potential(sim.X, sim.Y,
                              amplitude_J=AMPLITUDE_J * coupling_scale,
                              m_phi_ev=M_PHI_EV, phase0=0.0,
                              v_dm=220e3, direction=0.0, spatial_modulation=True)
    else:
        V_dm_callable, omega = _make_uldm_cosine_factory(sim.X, sim.Y,
                              amplitude_J=AMPLITUDE_J * coupling_scale,
                              m_phi_ev=M_PHI_EV, phase0=0.0,
                              v_dm=220e3, direction=0.0, spatial_modulation=True)

    def V_total(coords, t):
        return V_env + V_dm_callable(coords, t)

    print(f"[INFO] Running state {tag} with coupling c={coupling_scale:.3f} …")
    res = sim.run(V_function=V_total, snapshot_interval=0)

    np.savez_compressed(RESULTS_TS / f"two_state_{tag}.npz",
                        times=res.times,
                        center_phases=res.center_phases,
                        ref_phases=res.ref_phases,
                        delta_phi=res.delta_phi)
    return res, omega

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    f1 = RESULTS_TS / "two_state_c1.npz"
    f2 = RESULTS_TS / "two_state_c2.npz"

    if FORCE_RERUN or (not f1.exists()) or (not f2.exists()):
        res1, omega = run_state(COUPLING_1, "c1")
        res2, _     = run_state(COUPLING_2, "c2")
        t    = res1.times
        phi1 = np.unwrap(np.array(res1.center_phases))
        phi2 = np.unwrap(np.array(res2.center_phases))
    else:
        d1 = np.load(f1); d2 = np.load(f2)
        t    = d1["times"]
        phi1 = np.unwrap(np.array(d1["center_phases"]))
        phi2 = np.unwrap(np.array(d2["center_phases"]))
        omega = (M_PHI_EV * EV2J) / HBAR

    # Differential internal-state phase and detrend (remove slow drift)
    dphi_int = phi1 - phi2
    dphi_int_detr, _ = detrend_linear(t, dphi_int)

    # A) Time series
    save_time_plot(t, dphi_int_detr, PLOTS_DIR / "two_state_detrended.png",
                   title="Two-state differential phase (detrended)",
                   ylabel="Δφ_int detrended [rad]")

    # B) PSD with f_DM marker
    f, P = compute_psd(t, dphi_int_detr)
    f_dm = omega / (2*np.pi)
    save_psd_plot(f, P, PLOTS_DIR / "two_state_psd.png",
                  f_dm=f_dm, title="PSD of two-state Δφ_int")

    # C) Lock-in amplitude at f_DM
    amp = lock_in_amplitude(t, dphi_int_detr, f_dm, tau=0.2)
    save_time_plot(t, amp, PLOTS_DIR / "two_state_lockin.png",
                   title=f"Two-state lock-in amplitude at f_DM≈{f_dm:.2f} Hz",
                   ylabel="Lock-in amplitude [rad]")

    print("[DONE] Plots →", PLOTS_DIR)

if __name__ == "__main__":
    main()
