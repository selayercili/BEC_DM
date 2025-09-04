#!/usr/bin/env python3
# scripts/run_simulation.py
"""
Run a simple test simulation: 2D TD-GPE with a cosine ULDM perturbation.
Saves Δφ(t) and a PSD plot.
"""
import sys
from pathlib import Path

# ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import welch

from src.bec_simulation import BECSimulation
from src.dark_matter import ul_dm_cosine_potential
from src.utils import RESULTS_DIR, SPECTRA_DIR

def main():
    # --- simulation parameters (tiny, fast for prototyping) ---
    nx = ny = 128
    dx = dy = 1.0  # meters per pixel (toy units)
    m_particle = 1.6726219e-27  # kg (proton mass)
    g = 1e-52  # interaction strength (tweak as needed)
    dt = 1e-3
    t_total = 0.5  # seconds (short test)

    # --- dark-matter parameters ---
    amplitude_J = 1e-30  # tiny potential amplitude (J)
    m_phi_ev = 1e-18     # ULDM mass in eV (choose small to get low freq)
    phase0 = 0.0
    v_dm = 220e3
    direction = 0.0

    # Build simulation
    sim = BECSimulation(nx=nx, ny=ny, dx=dx, dy=dy,
                        m_particle=m_particle, g=g, dt=dt, t_total=t_total)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)

    # Build DM potential function
    V_fn = ul_dm_cosine_potential((sim.X, sim.Y),
                                  amplitude_J=amplitude_J,
                                  m_phi_ev=m_phi_ev,
                                  phase0=phase0,
                                  v_dm=v_dm,
                                  direction=direction,
                                  spatial_modulation=False)  # start simple

    # Run simulation
    print("Running simulation...")
    result = sim.run(V_function=V_fn, snapshot_interval=0)

    # Save
    sim.save_time_series(result, filename="delta_phi_test.npz")
    sim.plot_delta_phi(result, filename="delta_phi_test.png")

    # Compute PSD of delta_phi
    fs = 1.0 / dt
    f, Pxx = welch(result.delta_phi, fs=fs, nperseg=min(256, len(result.delta_phi)))
    out = SPECTRA_DIR / "delta_phi_psd.npz"
    np.savez_compressed(out, f=f, Pxx=Pxx)
    print(f"Saved PSD to {out}")

    # Plot PSD
    fig, ax = plt.subplots(figsize=(6,4))
    ax.loglog(f[1:], Pxx[1:])  # skip DC
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("PSD of Δφ(t)")
    fig.savefig(SPECTRA_DIR / "delta_phi_psd.png", bbox_inches="tight")
    plt.close(fig)
    print("Done. Results are in the results/ folder.")

if __name__ == "__main__":
    main()
