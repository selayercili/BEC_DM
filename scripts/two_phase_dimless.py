#!/usr/bin/env python3
# scripts/two_phase_dimless.py
# Physically scaled two-state differential phase runner (dimensionless trap mapping)

from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt

# ── project paths ─────────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
SRC  = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.bec_simulation import BECSimulation

# ── outputs ──────────────────────────────────────────────────────────────────
RESULTS_TS = ROOT / "results" / "two_state_dimless" / "time_series"
PLOTS_DIR  = ROOT / "results" / "two_state_dimless" / "plots"
RESULTS_TS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── physical constants ───────────────────────────────────────────────────────
HBAR = 1.054_571_817e-34     # J*s
EV2J = 1.602_176_634e-19     # J/eV

# ── dimensionless mapping choices (lab-like) ─────────────────────────────────
# Trap frequency (choose a typical cold-atom value). Sets the energy scale E0 = ħ ω_tr.
F_TR_HZ   = 50.0                         # trap frequency in Hz
OMEGA_TR  = 2*np.pi*F_TR_HZ              # rad/s
E0        = HBAR * OMEGA_TR              # J

# Particle mass in the solver (doesn't change mapping). Keep your existing value.
M_PARTICLE = 1.6726219e-27               # kg (proton used in your solver)

# Grid / solver settings (kept close to your working setup)
NX = NY = 128
DX = DY = 1.0            # treated as units of the harmonic oscillator length (dimensionless X,Y)
DT = 1e-3                # seconds (solver runs in SI time; we feed it V in Joules)
T_TOTAL = 30.0           # seconds (longer run → narrower PSD bins)
G  = 1e-52               # your existing nonlinearity

# Two-state coupling (opposite signs; physically corresponds to states with opposite effective coupling)
COUPLING_1 = +1.0
COUPLING_2 = +1.0

# Dark-matter parameters (physically small; no boosting)
M_PHI_EV   = 1e-12                       # eV (ULDM "mass")
OMEGA_DM   = (M_PHI_EV * EV2J) / HBAR    # rad/s
OMEGA_BAR  = OMEGA_DM / OMEGA_TR         # dimensionless DM frequency Ω = ω_DM/ω_tr
EPSILON    = 1e-5                        # dimensionless DM amplitude ε ≪ 1 (small perturbation)

# tiny spatial structure so the drive isn't perfectly uniform (breaks common-mode)
SPATIAL_EPS   = 1e-3                     # dimensionless modulation strength ≪ 1
SPATIAL_THETA = 0.0                      # direction of weak gradient

# Analysis settings
HPF_CUTOFF_HZ = 50.0                     # remove slow drift (well below f_DM≈242 Hz)
LOCKIN_TAU_S  = 0.02
DEBUG = True

# ── helpers ──────────────────────────────────────────────────────────────────
def highpass(y, fs, fc=50.0, order=2):
    b, a = butter(order, fc/(0.5*fs), btype='highpass')
    return filtfilt(b, a, y)

def running_avg(y, N):
    if N < 3: return y
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(np.asarray(y), size=int(N), mode="nearest")

def save_time_plot(t, y, path, title, ylabel, avgN=35):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, y, lw=1.1, label="signal")
    if len(y) >= avgN:
        ax.plot(t, running_avg(y, avgN), lw=1.6, label=f"running avg (N={avgN})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.35); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)

def compute_psd(y, fs):
    nper = min(8192, len(y))
    f, P = welch(y, fs=fs, window='hann', nperseg=nper, noverlap=nper//2)
    return f, P

def save_psd(f, P, f_dm, path, title="PSD"):
    fig, ax = plt.subplots(figsize=(9,3.6))
    ax.semilogy(f, P, lw=1.1)
    ax.axvline(f_dm, color="red", ls="--", label=f"f_DM ≈ {f_dm:.2f} Hz")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD [arb.]"); ax.set_title(title)
    ax.grid(True, ls='--', which='both', alpha=0.35); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)

def save_psd_zoom(f, P, f_dm, span=25.0, path=None):
    m = (f > f_dm-span) & (f < f_dm+span)
    fig, ax = plt.subplots(figsize=(8,3.2))
    ax.semilogy(f[m], P[m], lw=1.1)
    ax.axvline(f_dm, color="red", ls="--", label=f"f_DM")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD [arb.]"); ax.set_title(f"PSD around f_DM ±{span} Hz")
    ax.grid(True, ls='--', which='both', alpha=0.35); ax.legend(); fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def lock_in(t, y, f0, tau=0.02):
    w = 2*np.pi*f0
    I = y*np.cos(w*t); Q = y*np.sin(w*t)
    N = max(3, int(round(tau / (t[1]-t[0]))))
    return np.sqrt(running_avg(I, N)**2 + running_avg(Q, N)**2)

# ── dimensionless potentials mapped back to Joules ───────────────────────────
def build_V_total_callable(X, Y, epsilon, c_state, omega_bar, spatial_eps, spatial_theta):
    """
    We construct V/J = E0 * [ 0.5 r^2 + c_state * ε * (1 + spatial_eps * grad·r) * cos(Ω τ) ],
    with τ = ω_tr * t, so V(t) in Joules is returned to the solver.
    X,Y are treated as dimensionless coordinates in oscillator units (L/L0).
    """
    R2 = X*X + Y*Y
    # weak linear spatial term to break perfect uniformity (dimensionless)
    gx = spatial_eps * np.cos(spatial_theta)
    gy = spatial_eps * np.sin(spatial_theta)
    GdotR = gx*X + gy*Y

    def V_total(coords, t):
        # map physical time t [s] to dimensionless τ = ω_tr * t
        tau = OMEGA_TR * t
        # dimensionless potential inside [...]
        V_dimless = 0.5*R2 + c_state * epsilon * (1.0 + GdotR) * np.cos(omega_bar * tau)
        return E0 * V_dimless

    return V_total

# ── single-state run ─────────────────────────────────────────────────────────
def run_state(tag, c_state, epsilon, omega_bar, spatial_eps, spatial_theta):
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=T_TOTAL)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    V_total = build_V_total_callable(sim.X, sim.Y,
                                     epsilon=epsilon,
                                     c_state=c_state,
                                     omega_bar=omega_bar,
                                     spatial_eps=spatial_eps,
                                     spatial_theta=spatial_theta)

    if DEBUG:
        center = (sim.ny//2, sim.nx//2)
        v0  = V_total((sim.X, sim.Y), 0.0)[center]
        vQ  = V_total((sim.X, sim.Y), 0.25*(2*np.pi/OMEGA_DM))[center]  # quarter period at physical ω_DM
        vH  = V_total((sim.X, sim.Y), 0.5 *(2*np.pi/OMEGA_DM))[center]
        print(f"[{tag}] V_center@t0={v0:.3e} J, @tQ={vQ:.3e} J, @tH={vH:.3e} J (should flip sign)")

    print(f"[INFO] Running state {tag} with c_state={c_state:+.1f}, ε={epsilon:g}, Ω={omega_bar:.3f} …")
    res = sim.run(V_function=V_total, snapshot_interval=0)

    np.savez_compressed(RESULTS_TS / f"dimless_{tag}.npz",
                        times=res.times,
                        center_phases=res.center_phases,
                        ref_phases=res.ref_phases,
                        delta_phi=res.delta_phi)
    return res

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # Physical frequencies
    f_dm = OMEGA_DM / (2*np.pi)     # ≈ 242 Hz
    fs   = 1.0/DT

    # Run the two states (opposite couplings)
    r1 = run_state("c1",  COUPLING_1, EPSILON, OMEGA_BAR, SPATIAL_EPS, SPATIAL_THETA)
    r2 = run_state("c2", COUPLING_2, EPSILON, OMEGA_BAR, SPATIAL_EPS, SPATIAL_THETA)

    t    = np.asarray(r1.times)
    phi1 = np.unwrap(np.asarray(r1.center_phases, dtype=float))
    phi2 = np.unwrap(np.asarray(r2.center_phases, dtype=float))

    # Raw & differential
    from pathlib import Path
    save_time_plot(t, phi1, PLOTS_DIR / "phi1_raw.png", "φ₁ (raw center phase)", "[rad]")
    save_time_plot(t, phi2, PLOTS_DIR / "phi2_raw.png", "φ₂ (raw center phase)", "[rad]")
    dphi = phi1 - phi2
    save_time_plot(t, dphi, PLOTS_DIR / "dphi_raw.png", "Δφ_int (raw)", "[rad]")

    # High-pass to remove slow drift, then detrend any residual linear slope
    dphi_hp = highpass(dphi, fs=fs, fc=HPF_CUTOFF_HZ, order=2)
    p = np.polyfit(t, dphi_hp, 1)
    dphi_detr = dphi_hp - np.polyval(p, t)
    save_time_plot(t, dphi_detr, PLOTS_DIR / "dphi_detrended.png",
                   "Δφ_int (HPF + detrended)", "[rad]")

    # PSD (full and zoom)
    f, P = compute_psd(dphi_detr, fs=fs)
    save_psd(f, P, f_dm, PLOTS_DIR / "psd_full.png", "PSD of Δφ_int (HPF+detr)")
    save_psd_zoom(f, P, f_dm, span=25.0, path=PLOTS_DIR / "psd_zoom_fdm.png")

    # Local SNR metric near f_DM
    band = (f > f_dm-2.0) & (f < f_dm+2.0)
    nbhd = ((f > f_dm-20) & (f < f_dm-5)) | ((f > f_dm+5) & (f < f_dm+20))
    signal_power = float(P[band].max()) if band.any() else 0.0
    noise_floor  = float(np.median(P[nbhd])) if nbhd.any() else 1e-12
    snr = signal_power / noise_floor

    # Lock-in detector at f_DM
    amp = lock_in(t, dphi_detr, f_dm, tau=LOCKIN_TAU_S)
    save_time_plot(t, amp, PLOTS_DIR / "lockin_amp.png",
                   f"Lock-in amplitude at f_DM≈{f_dm:.2f} Hz", "[rad]")

    # Summaries
    s1 = {"len": len(phi1), "std": float(np.std(phi1))}
    s2 = {"len": len(phi2), "std": float(np.std(phi2))}
    sd = {"len": len(dphi), "std": float(np.std(dphi)), "std_detr": float(np.std(dphi_detr))}
    summary = {
        "trap": {"f_tr_hz": F_TR_HZ, "omega_tr_rad_s": OMEGA_TR, "E0_J": E0},
        "dm":   {"m_phi_ev": M_PHI_EV, "omega_dm_rad_s": OMEGA_DM,
                 "f_dm_hz": f_dm, "Omega=omega_dm/omega_tr": OMEGA_BAR,
                 "epsilon_dimless": EPSILON, "spatial_eps": SPATIAL_EPS},
        "solver": {"dt_s": DT, "t_total_s": T_TOTAL, "fs_hz": fs, "nx": NX, "ny": NY},
        "phi1": s1, "phi2": s2, "dphi": sd,
        "analysis": {"hpf_cutoff_hz": HPF_CUTOFF_HZ,
                     "lockin_tau_s": LOCKIN_TAU_S,
                     "local_snr_near_fdm": snr}
    }
    with open(RESULTS_TS / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2))
    print("\n[OUTPUT] Plots →", PLOTS_DIR)
    print("[OUTPUT] Data  →", RESULTS_TS)

if __name__ == "__main__":
    main()
