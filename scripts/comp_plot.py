#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d

# Paths
BASE_DIR   = Path(__file__).resolve().parents[1]
RESULTS_TS = BASE_DIR / "results" / "time_series"
PLOTS_DIR  = BASE_DIR / "results" / "plots"
RESULTS_TS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.bec_simulation import BECSimulation
from src.environment   import create_environment_potential, neutron_star_potential
from src.dark_matter   import ul_dm_cosine_potential

# constants
HBAR = 1.054571817e-34
EV2J = 1.602176634e-19

# Visual test params (unchanged physics engine)
TEST_MPHI_EV     = 1e-12      # eV → f_DM ≈ 242 Hz
TEST_AMPLITUDE_J = 1e-18      # J (boosted so you can see it in seconds)
TEST_T_TOTAL     = 7.0        # s

NX = NY = 128
DX = DY = 1.0
DT = 1e-3
G = 1e-52
M_PARTICLE = 1.6726219e-27

FORCE_RERUN = True   # <-- set True once so we get center_phases in NPZ

# helpers
def running_avg_safe(x, N):
    N = max(3, int(N))
    return uniform_filter1d(np.asarray(x), size=N, mode="nearest")

def detrend_linear(t, y):
    p = np.polyfit(t, y, 1)
    return y - np.polyval(p, t), p

def compute_psd(times, y):
    dt = float(times[1]-times[0])
    fs = 1.0/dt
    nper = min(2048, len(y))
    f, Pxx = welch(y, fs=fs, nperseg=nper)
    return f, Pxx

def lock_in_amplitude(times, y, f0, tau=0.2):
    t = np.asarray(times); y = np.asarray(y)
    w = 2*np.pi*f0
    I = y*np.cos(w*t)
    Q = y*np.sin(w*t)
    N = max(3, int(round(tau/(t[1]-t[0]))))
    I_f = running_avg_safe(I, N); Q_f = running_avg_safe(Q, N)
    return np.sqrt(I_f**2 + Q_f**2)

def save_time_plot(t, y, path, title, y_limits=None, avgN=35, ylabel="Δφ [rad]"):
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(t, y, alpha=0.8, label="signal")
    ax.plot(t, running_avg_safe(y, avgN), lw=1.8, label=f"running avg (N={avgN})")
    if y_limits is not None:
        ax.set_ylim(*y_limits); ax.set_autoscale_on(False)
    else:
        ymin, ymax = np.min(y), np.max(y)
        pad = 0.1*(ymax-ymin) if ymax>ymin else 1.0
        ax.set_ylim(ymin-pad, ymax+pad)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.4); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)

# sims
def run_with_dm():
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=TEST_T_TOTAL)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    V_dm  = ul_dm_cosine_potential((sim.X, sim.Y),
                                   amplitude_J=TEST_AMPLITUDE_J,
                                   m_phi_ev=TEST_MPHI_EV,
                                   phase0=0.0, v_dm=220e3, direction=0.0,
                                   spatial_modulation=True)
    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential,
                                         center_offset=(0.0,0.0), R_ns=1.0)
    def V_total(coords, t): return V_dm(coords,t) + V_env

    print("[INFO] Running WITH-DM…")
    res = sim.run(V_function=V_total, snapshot_interval=0)
    np.savez_compressed(RESULTS_TS / "delta_phi_with_dm.npz",
                        times=res.times, delta_phi=res.delta_phi,
                        center_phases=res.center_phases, ref_phases=res.ref_phases)
    return res

def run_no_dm():
    sim = BECSimulation(nx=NX, ny=NY, dx=DX, dy=DY,
                        m_particle=M_PARTICLE, g=G, dt=DT, t_total=TEST_T_TOTAL)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)
    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential)
    def V_total(coords, t): return V_env

    print("[INFO] Running NO-DM…")
    res = sim.run(V_function=V_total, snapshot_interval=0)
    np.savez_compressed(RESULTS_TS / "delta_phi_no_dm.npz",
                        times=res.times, delta_phi=res.delta_phi,
                        center_phases=res.center_phases, ref_phases=res.ref_phases)
    return res

def main():
    f_dm = (TEST_MPHI_EV*EV2J/HBAR)/(2*np.pi)

    f_with = RESULTS_TS / "delta_phi_with_dm.npz"
    f_nodm = RESULTS_TS / "delta_phi_no_dm.npz"

    if FORCE_RERUN or (not f_with.exists()):
        res_dm = run_with_dm()
    else:
        d = np.load(f_with)
        res_dm = type("R",(),{})()
        res_dm.times = d["times"]; res_dm.delta_phi = d["delta_phi"]
        res_dm.center_phases = d["center_phases"]; res_dm.ref_phases = d["ref_phases"]
        print("[INFO] Loaded WITH-DM NPZ.")

    if FORCE_RERUN or (not f_nodm.exists()):
        res_no = run_no_dm()
    else:
        d = np.load(f_nodm)
        res_no = type("R",(),{})()
        res_no.times = d["times"]; res_no.delta_phi = d["delta_phi"]
        res_no.center_phases = d["center_phases"]; res_no.ref_phases = d["ref_phases"]
        print("[INFO] Loaded NO-DM NPZ.")

    # === A) RAW Δφ, shared y (context)
    mn = float(min(res_dm.delta_phi.min(), res_no.delta_phi.min()))
    mx = float(max(res_dm.delta_phi.max(), res_no.delta_phi.max()))
    pad = 0.1*(mx-mn) if mx>mn else 1.0
    ylims = (mn-pad, mx+pad)
    save_time_plot(res_dm.times, res_dm.delta_phi,
                   PLOTS_DIR/"phase_with_dm_raw_shared.png",
                   "Δφ(t) — With DM (raw, shared y)", y_limits=ylims)
    save_time_plot(res_no.times, res_no.delta_phi,
                   PLOTS_DIR/"phase_no_dm_raw_shared.png",
                   "Δφ(t) — No DM (raw, shared y)", y_limits=ylims)

    # === B) DETRENDED Δφ overlay + residual
    detr_dm, _ = detrend_linear(res_dm.times, res_dm.delta_phi)
    detr_no, _ = detrend_linear(res_no.times, res_no.delta_phi)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(res_dm.times, running_avg_safe(detr_dm,35), label="With DM (detrended)")
    ax.plot(res_no.times, running_avg_safe(detr_no,35), label="No DM (detrended)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Δφ detrended [rad]")
    ax.set_title("Detrended Δφ(t): With DM vs No DM"); ax.grid(True, ls='--', alpha=0.4)
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS_DIR/"phase_detrended_overlay.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # residual
    if len(res_dm.times)==len(res_no.times) and np.allclose(res_dm.times, res_no.times):
        times_res = res_dm.times; resid = detr_dm - detr_no
    else:
        tmin = max(res_dm.times.min(), res_no.times.min())
        tmax = min(res_dm.times.max(), res_no.times.max())
        times_res = np.linspace(tmin, tmax, min(len(res_dm.times), len(res_no.times)))
        resid = np.interp(times_res, res_dm.times, detr_dm) - \
                np.interp(times_res, res_no.times, detr_no)
    save_time_plot(times_res, running_avg_safe(resid,35),
                   PLOTS_DIR/"phase_residual_detrended.png",
                   "Residual (With DM detrended − No DM detrended)",
                   y_limits=None, ylabel="Residual [rad]")

    # === C) COMMON-MODE: center phase (this is the DM-sensitive channel)
    c_dm_detr, _ = detrend_linear(res_dm.times, res_dm.center_phases)
    c_no_detr, _ = detrend_linear(res_no.times, res_no.center_phases)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(res_dm.times, running_avg_safe(c_dm_detr,35), label="With DM (center, detrended)")
    ax.plot(res_no.times, running_avg_safe(c_no_detr,35), label="No DM (center, detrended)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Center phase detrended [rad]")
    ax.set_title("Detrended center phase (common-mode)"); ax.grid(True, ls='--', alpha=0.4)
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS_DIR/"center_phase_detrended_overlay.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # PSD of center phase (detrended) with f_DM marked
    fC_dm, P_dm = compute_psd(res_dm.times, c_dm_detr)
    fC_no, P_no = compute_psd(res_no.times, c_no_detr)
    fig, ax = plt.subplots(figsize=(8,3.6))
    ax.semilogy(fC_dm, P_dm, label="With DM"); ax.semilogy(fC_no, P_no, label="No DM")
    ax.axvline(f_dm, color='red', ls='--', label=f"f_DM ≈ {f_dm:.2f} Hz")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD [arb.]")
    ax.set_title("PSD of center phase (detrended)"); ax.grid(True, ls='--', alpha=0.4, which='both')
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS_DIR/"center_phase_psd.png", dpi=150, bbox_inches='tight'); plt.close(fig)

    # === D) Lock-in amplitude at f_DM for center phase
    amp_dm = lock_in_amplitude(res_dm.times, c_dm_detr, f_dm, tau=0.2)
    amp_no = lock_in_amplitude(res_no.times, c_no_detr, f_dm, tau=0.2)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(res_dm.times, amp_dm, label="With DM")
    ax.plot(res_no.times, amp_no, label="No DM")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Lock-in amplitude [rad]")
    ax.set_title(f"Lock-in at f_DM ≈ {f_dm:.2f} Hz (center phase)")
    ax.grid(True, ls='--', alpha=0.4); ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS_DIR/"center_phase_lockin.png", dpi=150, bbox_inches='tight'); plt.close(fig)

    # Δφ PSDs (keep for completeness; less decisive than common-mode)
    f_dmphi, P_dmphi = compute_psd(res_dm.times, res_dm.delta_phi)
    f_nophi, P_nophi = compute_psd(res_no.times, res_no.delta_phi)
    fig, ax = plt.subplots(figsize=(8,3.6))
    ax.semilogy(f_dmphi, P_dmphi, label="With DM"); ax.semilogy(f_nophi, P_nophi, label="No DM")
    ax.axvline(f_dm, color='red', ls='--', label=f"f_DM ≈ {f_dm:.2f} Hz")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD [arb.]")
    ax.set_title("PSD of Δφ (detrended not applied here)"); ax.grid(True, ls='--', alpha=0.4, which='both')
    ax.legend(); fig.tight_layout()
    fig.savefig(PLOTS_DIR/"delta_phi_psd.png", dpi=150, bbox_inches='tight'); plt.close(fig)

    print("[DONE] Wrote plots to", PLOTS_DIR)

if __name__ == "__main__":
    main()
