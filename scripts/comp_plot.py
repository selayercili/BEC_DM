# comp_plot.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

# Import your simulation class
from src.bec_simulation import BECSimulation
from src.environment import neutron_star_environment
from src.potentials import ul_dm_cosine_potential

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "time_series"
PLOTS_DIR = BASE_DIR / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_psd(times, signal):
    fs = 1.0 / (times[1] - times[0])  # sampling frequency
    freqs, psd = welch(signal, fs=fs, nperseg=2048)
    return freqs, psd

def run_no_dm_sim():
    """Run the simulation without DM potential (V_dm=0)."""
    sim = BECSimulation(grid_size=64, dx=0.1, dt=0.01, t_max=20.0)
    V_env = neutron_star_environment(sim.X, sim.Y)

    def total_potential(coords, t):
        return V_env  # no DM contribution

    result = sim.run(V_function=total_potential, snapshot_interval=0)
    out_file = RESULTS_DIR / "delta_phi_no_dm.npz"
    np.savez(out_file, times=result["times"], delta_phi=result["delta_phi"])
    print("Saved no-DM results to:", out_file)
    return result

def main():
    # --- Load DM results ---
    dm_data = np.load(RESULTS_DIR / "delta_phi_test.npz")
    times = dm_data["times"]
    delta_phi_dm = dm_data["delta_phi"]

    freqs_dm, psd_dm = compute_psd(times, delta_phi_dm)

    # --- Run or load no-DM simulation ---
    no_dm_file = RESULTS_DIR / "delta_phi_no_dm.npz"
    if no_dm_file.exists():
        no_dm_data = np.load(no_dm_file)
        delta_phi_no_dm = no_dm_data["delta_phi"]
    else:
        no_dm_result = run_no_dm_sim()
        times = no_dm_result["times"]
        delta_phi_no_dm = no_dm_result["delta_phi"]

    freqs_no_dm, psd_no_dm = compute_psd(times, delta_phi_no_dm)

    # --- Plot comparison ---
    plt.figure(figsize=(10,4))
    plt.loglog(freqs_dm, psd_dm, label="With DM", color="blue")
    plt.loglog(freqs_no_dm, psd_no_dm, label="No DM", color="orange", linestyle="--")

    f_dm = 2.418e-4  # Hz for m_phi = 1e-18 eV
    plt.axvline(f_dm, color="red", linestyle="--", label=f"Expected DM freq = {f_dm:.2e} Hz")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.title("PSD Comparison: With vs Without DM Effects")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    out_file = PLOTS_DIR / "psd_comparison.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print("Saved comparison plot to:", out_file)

if __name__ == "__main__":
    main()
