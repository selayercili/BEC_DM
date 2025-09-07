#!/usr/bin/env python3
# scripts/run_simulation.py
"""
Run a 2D TD-GPE simulation with ULDM perturbation and neutron star environment.
Saves Δφ(t) and PSD.
"""
import sys
from pathlib import Path

# ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from src.bec_simulation import BECSimulation
from src.dark_matter import ul_dm_cosine_potential
from src.utils import RESULTS_DIR, SPECTRA_DIR
from src.environment import neutron_star_potential, create_environment_potential

def main():
    # --- simulation parameters ---
    nx = ny = 128
    dx = dy = 1.0  # meters per pixel (toy units)
    m_particle = 1.6726219e-27  # kg
    g = 1e-52
    dt = 1e-3
    t_total = 20  # seconds (toy run; scale later in Colab)

    # --- dark matter parameters ---
    amplitude_J = 1e-24
    m_phi_ev = 1e-14
    phase0 = 0.0
    v_dm = 220e3
    direction = 0.0

    # --- build simulation ---
    sim = BECSimulation(nx=nx, ny=ny, dx=dx, dy=dy,
                        m_particle=m_particle, g=g, dt=dt, t_total=t_total)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    # --- DM potential ---
    V_dm = ul_dm_cosine_potential((sim.X, sim.Y),
                                  amplitude_J=amplitude_J,
                                  m_phi_ev=m_phi_ev,
                                  phase0=phase0,
                                  v_dm=v_dm,
                                  direction=direction,
                                  spatial_modulation=False)

    # --- Neutron star environment potential ---
    V_env = create_environment_potential(sim.X, sim.Y, neutron_star_potential)

    # --- Diagnostic prints: check potentials and shapes ---
    # If V_dm is a function of t (baked-in grid) call it at t=0, t=0.1, etc.
    try:
        V0 = V_dm(0.0)      # if V_dm expects t only
        V1 = V_dm(0.1)
    except TypeError:
        # if V_dm expects (grid, t)
        V0 = V_dm((sim.X, sim.Y), 0.0)
        V1 = V_dm((sim.X, sim.Y), 0.1)

    print("V_dm shape:", np.shape(V0), "min/max:", V0.min(), V0.max())
    print("V_dm(t=0.1) min/max:", V1.min(), V1.max())

    print("V_env shape:", np.shape(V_env), "min/max:", V_env.min(), V_env.max())

    tot0 = V0 + V_env
    print("Total potential at t=0: min/max:", tot0.min(), tot0.max())

    # --- Combine potentials ---
    def total_potential(grid_coords, t):
        """
        Combined potential function.
        
        Args:
            grid_coords: (X, Y) coordinate tuple
            t: time in seconds
            
        Returns:
            Combined potential array
        """

        X, Y = grid_coords
        # V_dm may be t-only or space+time; handle both
        try:
            Vd = V_dm(t)
        except TypeError:
            Vd = V_dm((X, Y), t)
        Vtot = Vd + V_env  # V_env must be array shaped (nx,ny)
        # Diagnostic
        if not isinstance(Vtot, np.ndarray):
            Vtot = np.array(Vtot)
        assert Vtot.shape == X.shape, f"Vtot shape {Vtot.shape} != grid shape {X.shape}"
        return Vtot
        

    # --- Run simulation ---
    print("Running simulation with neutron star environment...")
    result = sim.run(V_function=total_potential, snapshot_interval=0)

    # --- Save results ---
    sim.save_time_series(result, filename="delta_phi_test.npz")
    sim.plot_delta_phi(result, filename="delta_phi_test.png")

    # --- PSD computation ---
    fs = 1.0 / dt
    f, Pxx = welch(result.delta_phi, fs=fs, nperseg=min(256, len(result.delta_phi)))
    np.savez_compressed(SPECTRA_DIR / "delta_phi_psd.npz", f=f, Pxx=Pxx)

    # --- Plot PSD ---
    fig, ax = plt.subplots(figsize=(6,4))
    ax.loglog(f[1:], Pxx[1:])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("PSD of Δφ(t)")
    fig.savefig(SPECTRA_DIR / "delta_phi_psd.png", bbox_inches="tight")
    plt.close(fig)

    print("Simulation complete. Results saved in results/ folder.")

if __name__ == "__main__":
    main()