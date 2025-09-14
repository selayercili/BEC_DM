# src/bec_simulation.py
"""
BECSimulation: small 2D TD-GPE solver (split-step Fourier)
- Uses Strang (half potential - full kinetic - half potential) split-step
- Good for prototyping and debugging. Uses complex128 for precision.

Main interface:
    sim = BECSimulation(nx, ny, dx, dy, m_particle, g, dt, t_total)
    sim.initialize_wavefunction(kind="gaussian", width=8.0)
    result = sim.run(V_function=some_callable, snapshot_interval=100)

Result object contains:
    times (1D array), delta_phi (1D array), psi_snapshots (list of 2D arrays, optional)
"""
import numpy as np
from types import SimpleNamespace

# physical constants
hbar = 1.054571817e-34
eV_to_J = 1.602176634e-19
c_light = 299792458.0

class BECSimulation:
    def __init__(self, nx=128, ny=128, dx=1.0, dy=1.0,
                 m_particle=1.6726219e-27, g=1e-52, dt=1e-3, t_total=1.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.m = m_particle
        self.g = g
        self.dt = dt
        self.t_total = t_total

        # create grids
        x = (np.arange(nx) - nx//2) * dx
        y = (np.arange(ny) - ny//2) * dy
        self.X, self.Y = np.meshgrid(x, y, indexing='xy')

        # prepare FFT k-grid (angular wavenumbers)
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing='xy')
        self.k2 = KX**2 + KY**2  # k^2 for kinetic term

        # initial psi placeholder
        self.psi = np.zeros((ny, nx), dtype=np.complex128)  # note: (ny,nx)
        self._initial_norm = None

        # derived
        self.nsteps = int(np.floor(t_total / dt))
        self.times = np.arange(0, self.nsteps) * dt

    def initialize_wavefunction(self, kind="gaussian", width=8.0):
        """
        Initialize psi on the grid. width in grid units (same units as dx/dy)
        """
        if kind == "gaussian":
            sigma = width
            R2 = (self.X**2 + self.Y**2)
            psi0 = np.exp(-0.5 * R2 / (sigma**2)).astype(np.complex128)
        elif kind == "random":
            psi0 = (np.random.normal(size=self.X.shape) + 1j * np.random.normal(size=self.X.shape)).astype(np.complex128)
        else:
            raise ValueError("Unknown wavefunction kind")

        # normalize such that integral |psi|^2 dA = 1 (arbitrary normalization)
        norm = np.sqrt(np.sum(np.abs(psi0)**2) * self.dx * self.dy)
        psi0 /= norm
        self.psi = psi0
        self._initial_norm = np.sum(np.abs(self.psi)**2) * self.dx * self.dy

    def _kinetic_propagator(self, dt):
        """
        Return kinetic propagator array: exp(-i * (ħ k^2)/(2 m) * dt)
        """
        omega_k = (hbar * self.k2) / (2.0 * self.m)  # [J·s / kg?] units: (ħ k^2)/(2m)
        # Actually omega_k has units of [J·s]*[1/m^2]/kg -> reduces to [1/s] when assembled correctly
        return np.exp(-1j * omega_k * dt / hbar)  # exp(-i ω_k dt)

    def _apply_potential_step(self, psi, V, half_dt):
        """
        Multiply by potential operator exp(-i (V + g|psi|^2) dt / ħ)
        V should be grid-shaped array.
        """
        nonlinear = self.g * np.abs(psi)**2
        phase = np.exp(-1j * (V + nonlinear) * half_dt / hbar)
        return psi * phase

    def run(self, V_function, snapshot_interval=0):
        """
        Run the simulation.

        V_function: callable that accepts either:
            - (coords, t) -> array shape (ny,nx)
            - or (t) -> scalar or array
        snapshot_interval: if >0, number of steps between saved psi snapshots
        """
        # prepare result containers
        psi_snapshots = [] if snapshot_interval else None
        center_phases = []
        ref_phases = []

        # Precompute kinetic propagator for dt
        Kprop_dt = self._kinetic_propagator(self.dt)
        half_dt = self.dt / 2.0

        # helper to call V_function robustly
        def call_V(coords, t):
            """
            Robustly call V_function with either signature:
            - V_function(coords, t)
            - V_function(t)
            Always return a numpy array shaped like self.X
            """
            # Try to inspect signature if possible
            try:
                sig = inspect.signature(V_function)
                n_params = len(sig.parameters)
            except Exception:
                n_params = None

            V = None
            # Prefer coords,t when possible
            if n_params is None or n_params >= 2:
                try:
                    V = V_function(coords, t)
                except TypeError:
                    V = None
            if V is None:
                # try time-only call
                try:
                    V = V_function(t)
                except Exception:
                    # last resort: try calling with no args
                    try:
                        V = V_function()
                    except Exception as e:
                        raise RuntimeError(f"V_function is not callable with (coords,t) or (t): {e}")

            V = np.asarray(V)
            if V.shape == ():
                V = np.ones_like(self.X) * float(V)
            if V.shape != self.X.shape:
                try:
                    V = np.broadcast_to(V, self.X.shape)
                except Exception as e:
                    raise ValueError(f"V_function returned shape {V.shape} but expected {self.X.shape}") from e
            return V

        # region masks for phase measurement: small center disk and outer reference ring
        rr = np.sqrt(self.X**2 + self.Y**2)
        center_mask = (rr <= max(1.0, min(self.X.shape) * 0.05)).astype(np.float64)
        ref_mask = ((rr >= (0.45 * rr.max())) & (rr <= (0.5 * rr.max()))).astype(np.float64)
        # avoid zero division
        if center_mask.sum() == 0:
            center_mask[self.X.shape[0]//2, self.X.shape[1]//2] = 1.0
        if ref_mask.sum() == 0:
            ref_mask[0, 0] = 1.0

        # running loop
        psi = self.psi.copy()
        norm_target = self._initial_norm if self._initial_norm is not None else np.sum(np.abs(psi)**2) * self.dx * self.dy

        for step in range(self.nsteps):
            t = step * self.dt

            # get potential array for this step
            Vt = call_V((self.X, self.Y), t)

            # Strang-split: half-potential, full-kinetic (via FFT), half-potential
            psi = self._apply_potential_step(psi, Vt, half_dt)

            # kinetic: FFT -> multiply -> iFFT
            psi_k = np.fft.fft2(psi)
            psi_k *= Kprop_dt
            psi = np.fft.ifft2(psi_k)

            # second half potential (use same Vt)
            psi = self._apply_potential_step(psi, Vt, half_dt)

            # renormalize to avoid drift
            current_norm = np.sum(np.abs(psi)**2) * self.dx * self.dy
            if current_norm != 0:
                psi *= np.sqrt(norm_target / current_norm)

            # diagnostics: measure center and reference phases
            # complex average inside mask -> angle
            c_center = np.sum(psi * center_mask) / (np.sum(center_mask) + 1e-30)
            c_ref = np.sum(psi * ref_mask) / (np.sum(ref_mask) + 1e-30)

            center_phases.append(np.angle(c_center))
            ref_phases.append(np.angle(c_ref))

            # snapshot
            if snapshot_interval and (step % snapshot_interval == 0):
                if psi_snapshots is not None:
                    psi_snapshots.append(psi.copy())

        # post-process phases: unwrap then compute delta_phi
        center_phases = np.unwrap(np.array(center_phases))
        ref_phases = np.unwrap(np.array(ref_phases))
        delta_phi = center_phases - ref_phases

        result = SimpleNamespace()
        result.times = np.arange(0, self.nsteps) * self.dt
        result.delta_phi = delta_phi
        result.center_phases = center_phases
        result.ref_phases = ref_phases
        result.psi_snapshots = psi_snapshots
        result.params = dict(nx=self.nx, ny=self.ny, dx=self.dx, dy=self.dy,
                             m=self.m, g=self.g, dt=self.dt, t_total=self.t_total)

        return result

    # helpers to save and plot
    def save_time_series(self, result, filename="delta_phi_test.npz", out_dir=None):
        from pathlib import Path
        if out_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            out_dir = project_root / "results" / "time_series"
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / filename
        np.savez_compressed(out, times=result.times, delta_phi=result.delta_phi)
        return out

    def plot_delta_phi(self, result, filename="delta_phi_test.png", out_dir=None):
        import matplotlib.pyplot as plt
        from pathlib import Path
        if out_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            out_dir = project_root / "results" / "plots"
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(result.times, result.delta_phi, alpha=0.8)
        # running average
        N = max(1, len(result.delta_phi) // 200)
        avg = np.convolve(result.delta_phi, np.ones(N)/N, mode='same')
        ax.plot(result.times, avg, color='red', lw=2, label=f"running avg (N={N})")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Δφ [rad]")
        ax.set_title("Relative Phase Δφ(t)")
        ax.grid(True, ls="--")
        ax.legend()
        out = out_dir / filename
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        return out
