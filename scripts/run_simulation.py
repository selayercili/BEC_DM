#!/usr/bin/env python3
# scripts/run_simulation.py
"""
Top-level script: builds potentials robustly, runs a BECSimulation, saves results and PSD.
This version is defensive: it accepts different signatures from ul_dm_cosine_potential
and makes sure both DM and environment potentials are full (ny,nx) arrays.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# project root and sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# imports
from src.bec_simulation import BECSimulation
from src.environment import create_environment_potential, neutron_star_potential
try:
    from src.dark_matter import ul_dm_cosine_potential
except Exception:
    ul_dm_cosine_potential = None

# try to import RESULTS dirs if user has src.utils, otherwise fallback
try:
    from src.utils import RESULTS_DIR, SPECTRA_DIR
except Exception:
    RESULTS_DIR = project_root / "results"
    SPECTRA_DIR = RESULTS_DIR / "spectra"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(SPECTRA_DIR).mkdir(parents=True, exist_ok=True)

# physical constants
hbar = 1.054571817e-34
eV_to_J = 1.602176634e-19
c_light = 299792458.0

def make_Vdm_wrapper(base_callable, sim, amplitude_J, m_phi_ev, phase0=0.0, v_dm=0.0, direction=0.0, spatial_modulation=False):
    """
    Given whatever base_callable you have (from src.dark_matter.ul_dm_cosine_potential),
    return a robust function V_dm(coords, t) that ALWAYS returns an array shaped like sim.X.
    This wrapper:
      - tries to call base_callable((X,Y), t)
      - if that raises TypeError, tries base_callable(t)
      - if base returns a scalar, broadcasts to the grid
      - if spatial_modulation requested but base doesn't do it, we add a simple k·x term
    """
    X, Y = sim.X, sim.Y

    # compute phys quantities for optional spatial modulation
    mass_energy_J = m_phi_ev * eV_to_J          # m c^2 [J]
    omega = mass_energy_J / hbar                # angular freq [rad/s]
    # mass in kg:
    m_kg = mass_energy_J / (c_light**2)         # m [kg]
    # de Broglie k = m v / ħ
    kx_dm = (m_kg * v_dm * np.cos(direction)) / hbar if spatial_modulation else 0.0
    ky_dm = (m_kg * v_dm * np.sin(direction)) / hbar if spatial_modulation else 0.0

    def V_dm(coords, t):
        Xc, Yc = coords
        # Try calling base in the usual (coords,t) way first
        V = None
        if base_callable is not None:
            try:
                V = base_callable((Xc, Yc), t)
            except TypeError:
                try:
                    V = base_callable(t)
                except Exception:
                    V = None
            except Exception:
                V = None

        # If base failed or returned None, construct a safe cosine model
        if V is None:
            V = amplitude_J * np.cos(omega * t + kx_dm * Xc + ky_dm * Yc + phase0)
            return V

        V = np.asarray(V)
        if V.shape == ():  # scalar -> broadcast
            V = np.ones_like(Xc) * float(V)
            return V
        # If we get shape mismatch, try broadcasting or inserting spatial modulation
        if V.shape != Xc.shape:
            try:
                V = np.broadcast_to(V, Xc.shape)
            except Exception:
                # fallback to simple analytic model
                V = amplitude_J * np.cos(omega * t + kx_dm * Xc + ky_dm * Yc + phase0)
        return V

    return V_dm

def main():
    # --- simulation parameters (you can adjust) ---
    nx = ny = 128
    dx = dy = 1.0
    m_particle = 1.6726219e-27
    g = 1e-52
    dt = 1e-3
    t_total = 7.0  # debug run

    # --- ULDM params (debug-friendly defaults; scale to realistic later) ---
    amplitude_J = 1e-24
    m_phi_ev = 1e-14
    phase0 = 0.0
    v_dm = 220e3
    direction = 0.0
    spatial_modulation = True  # try to include spatial k·x term

    # build sim
    sim = BECSimulation(nx=nx, ny=ny, dx=dx, dy=dy,
                        m_particle=m_particle, g=g, dt=dt, t_total=t_total)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    # build base ULDM callable if available
    base_callable = None
    if ul_dm_cosine_potential is not None:
        try:
            base_callable = ul_dm_cosine_potential((sim.X, sim.Y),
                                                  amplitude_J=amplitude_J,
                                                  m_phi_ev=m_phi_ev,
                                                  phase0=phase0,
                                                  v_dm=v_dm,
                                                  direction=direction,
                                                  spatial_modulation=spatial_modulation)
        except Exception:
            # if that fails, we still continue with wrapper which constructs model
            base_callable = None

    V_dm = make_Vdm_wrapper(base_callable, sim,
                            amplitude_J=amplitude_J,
                            m_phi_ev=m_phi_ev,
                            phase0=phase0,
                            v_dm=v_dm,
                            direction=direction,
                            spatial_modulation=spatial_modulation)

    # environment potential array
    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential)
    # Debug prints
    Vdm0 = V_dm((sim.X, sim.Y), 0.0)
    Vdm1 = V_dm((sim.X, sim.Y), 0.1)
    print("[DEBUG] V_dm(t=0) shape, min, max, std:", Vdm0.shape, Vdm0.min(), Vdm0.max(), Vdm0.std())
    print("[DEBUG] V_dm(t=0.1) min/max diff:", (Vdm1 - Vdm0).min(), (Vdm1 - Vdm0).max())
    print("[DEBUG] V_env shape, min, max, std:", V_env.shape, V_env.min(), V_env.max(), V_env.std())

    # combined potential function for solver: returns array (ny,nx)
    def total_potential(coords, t):
        Xc, Yc = coords
        return V_dm((Xc, Yc), t) + V_env

    # run
    print("Running simulation...")
    result = sim.run(V_function=total_potential, snapshot_interval=0)

    # save time series
    out_ts_dir = RESULTS_DIR / "time_series"
    out_ts_dir.mkdir(parents=True, exist_ok=True)
    out_ts_file = out_ts_dir / "delta_phi_test.npz"
    np.savez_compressed(out_ts_file, times=result.times, delta_phi=result.delta_phi)
    print("Saved timeseries to:", out_ts_file)

    # make phase plot using BECSimulation helper
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sim.plot_delta_phi(result, filename="delta_phi_test.png", out_dir=plots_dir)
    print("Saved phase plot to:", plots_dir / "delta_phi_test.png")

    # PSD
    fs = 1.0 / dt
    f, Pxx = welch(result.delta_phi, fs=fs, nperseg=min(256, len(result.delta_phi)))
    np.savez_compressed(SPECTRA_DIR / "delta_phi_psd.npz", f=f, Pxx=Pxx)
    # Plot PSD
    fig, ax = plt.subplots(figsize=(6,4))
    ax.loglog(f[1:], Pxx[1:])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("PSD of Δφ(t)")
    fig.savefig(SPECTRA_DIR / "delta_phi_psd.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved PSD to:", SPECTRA_DIR / "delta_phi_psd.png")

    # print expected DM frequency (physically consistent)
    mass_energy_J = m_phi_ev * eV_to_J
    omega_dm = mass_energy_J / hbar
    f_dm = omega_dm / (2 * np.pi)
    print(f"[INFO] expected DM frequency from m_phi={m_phi_ev:.1e} eV: {f_dm:.3e} Hz (period {(1/f_dm):.1f} s)")

if __name__ == "__main__":
    main()
