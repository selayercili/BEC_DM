#!/usr/bin/env python3
"""
Plot relative phase shift vs time from simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
project_root = Path("/content/BEC_DM")
results_dir = project_root / "results"
ts_file = results_dir / "time_series/delta_phi_test.npz"

def plot_phase_shift(times, delta_phi, outdir):
    """
    Plot relative phase shift vs time with diagnostics.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Main plot with running average ---
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(times, delta_phi, label="Δφ(t)", alpha=0.7)

    # running average over ~1% of total points
    N = max(1, len(delta_phi)//100)
    avg = np.convolve(delta_phi, np.ones(N)/N, mode="same")
    ax.plot(times, avg, 'r', lw=2, label=f"Running avg (N={N})")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Relative Phase Δφ [rad]")
    ax.set_title("Phase Shift vs Time")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    fig.savefig(outdir / "phase_vs_time.png", bbox_inches="tight")
    plt.close(fig)

    # --- Zoom in on first second ---
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(times, delta_phi, alpha=0.6)
    ax.set_xlim(0, min(1.0, times[-1]))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Δφ [rad]")
    ax.set_title("Zoom: First second")
    ax.grid(True, ls="--", alpha=0.5)
    fig.savefig(outdir / "phase_zoom.png", bbox_inches="tight")
    plt.close(fig)

    # --- Histogram of values ---
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(delta_phi, bins=50, alpha=0.7)
    ax.set_xlabel("Δφ [rad]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Phase Shift Values")
    fig.savefig(outdir / "phase_hist.png", bbox_inches="tight")
    plt.close(fig)

def main():
    # --- Load data ---
    data = np.load(ts_file)
    times, delta_phi = data["times"], data["delta_phi"]

    print(f"Loaded {len(times)} timesteps from {ts_file}")
    print(f"Δφ range: {delta_phi.min():.3e} to {delta_phi.max():.3e} rad")

    # --- Plot ---
    plot_phase_shift(times, delta_phi, results_dir)
    print("Plots saved to results/ folder.")

if __name__ == "__main__":
    main()
