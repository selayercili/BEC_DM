import numpy as np
import matplotlib.pyplot as plt

def plot_phase_shift(times, delta_phi, outpath=None):
    """
    Plot relative phase shift vs time with extra diagnostics.
    """

    # --- Basic line plot ---
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(times, delta_phi, label="Δφ(t)", alpha=0.7)
    
    # Running average (smooth)
    N = max(1, len(delta_phi)//100)  # ~1% smoothing window
    avg = np.convolve(delta_phi, np.ones(N)/N, mode="same")
    ax.plot(times, avg, 'r', lw=2, label=f"Running avg (N={N})")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Relative Phase Δφ [rad]")
    ax.set_title("Phase Shift vs Time")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    if outpath:
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    # --- Extra: Zoom in on first second ---
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(times, delta_phi, label="Δφ(t)", alpha=0.6)
    ax.set_xlim(0, min(1.0, times[-1]))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Δφ [rad]")
    ax.set_title("Zoom: First second")
    ax.grid(True, ls="--", alpha=0.5)
    plt.show()

    # --- Extra: Histogram of phase values ---
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(delta_phi, bins=50, alpha=0.7)
    ax.set_xlabel("Δφ [rad]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Phase Shift Values")
    plt.show()
