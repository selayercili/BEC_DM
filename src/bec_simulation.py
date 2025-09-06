# src/bec_simulation.py
"""
Minimal 2D split-step Fourier solver for the time-dependent Gross-Pitaevskii equation (TD-GPE).

Equation (in convenient units):
    i ħ ∂Ψ/∂t = [- (ħ^2 / 2m) ∇^2 + V_ext(r, t) + g |Ψ|^2 ] Ψ

This implementation uses:
 - constant particle mass `m`
 - periodic boundary conditions via FFT
 - split-step method: half kinetic -> potential+nonlinear -> half kinetic
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from typing import Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
import json

from .utils import grid_coords, TIME_SERIES_DIR, FIGURES_DIR
import matplotlib.pyplot as plt

@dataclass
class SimulationResult:
    times: np.ndarray
    delta_phi: np.ndarray
    psi_snapshots: np.ndarray  # optional, can be empty

class BECSimulation:
    def __init__(self,
                 nx: int = 128,
                 ny: int = 128,
                 dx: float = 1.0,
                 dy: float = 1.0,
                 m_particle: float = 1.6726219e-27,
                 g: float = 1e-51,
                 dt: float = 1e-3,
                 t_total: float = 1.0):
        """
        Initialize simulation grid and physical parameters.

        Args:
            nx, ny: grid resolution
            dx, dy: grid spacing in meters
            m_particle: mass of the condensed particle (kg)
            g: interaction strength (J*m^2) -- tune as needed for scale
            dt: time step (s)
            t_total: total simulation time (s)
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.m = m_particle
        self.g = g
        self.dt = dt
        self.t_total = t_total

        # grids
        self.X, self.Y = grid_coords(nx, ny, dx, dy)
        self._init_kgrid()

    def _init_kgrid(self):
        """Precompute k^2 grid for kinetic evolution operator."""
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        self.k2 = KX**2 + KY**2

        # kinetic evolution factor for half step
        hbar = 1.054571817e-34
        self._K_half = np.exp(-1j * (self.dt / 2.0) * (hbar / (2.0 * self.m)) * self.k2)

    def initialize_wavefunction(self, kind: str = "gaussian", width: float = 10.0):
        """Initialize Ψ on the grid."""
        if kind == "gaussian":
            r2 = self.X**2 + self.Y**2
            sigma2 = width**2
            psi0 = np.exp(-r2 / (2 * sigma2))
            # normalize
            norm = np.sqrt(np.sum(np.abs(psi0)**2) * self.dx * self.dy)
            self.psi = psi0 / norm
        elif kind == "uniform":
            psi0 = np.ones((self.nx, self.ny), dtype=np.complex128)
            norm = np.sqrt(np.sum(np.abs(psi0)**2) * self.dx * self.dy)
            self.psi = psi0 / norm
        else:
            raise ValueError("Unknown initialization kind")

    def _apply_kinetic_half(self, psi: np.ndarray) -> np.ndarray:
        """Apply half step kinetic evolution using FFT."""
        psi_k = fft2(psi)
        psi_k *= self._K_half
        return ifft2(psi_k)

    def _apply_potential_and_nonlinear(self, psi: np.ndarray, V_ext: np.ndarray) -> np.ndarray:
        """Apply full-step potential + nonlinear evolution: exp(-i dt (V + g|ψ|^2)/ħ)"""
        hbar = 1.054571817e-34
        nonlinear = self.g * (np.abs(psi)**2)
        phase = np.exp(-1j * (self.dt / hbar) * (V_ext + nonlinear))
        return psi * phase

    def run(self, V_function: Callable[[Tuple[np.ndarray, np.ndarray], float], np.ndarray], 
            readout_mask: np.ndarray = None,
            snapshot_interval: int = 0) -> SimulationResult:
        """
        Run the TD-GPE solver.

        Args:
            V_function: function of ((X, Y), t) -> 2D potential (same shape as grid)
            readout_mask: boolean mask selecting region A (if None, will use left half vs right half)
            snapshot_interval: save psi snapshot every N steps (0 => none)

        Returns:
            SimulationResult with times and Δφ(t).
        """
        n_steps = int(np.ceil(self.t_total / self.dt))
        times = np.arange(n_steps) * self.dt
        delta_phi_ts = np.zeros(n_steps, dtype=np.float64)

        if readout_mask is None:
            # define two regions: left half vs right half for relative phase
            readout_mask = np.zeros((self.nx, self.ny), dtype=bool)
            readout_mask[:, : self.ny // 2] = True  # left half True

        psi_snapshots = [] if snapshot_interval else None

        # main integration loop
        for i, t in enumerate(times):
            # half kinetic
            self.psi = self._apply_kinetic_half(self.psi)

            # potential + nonlinear full step
            Vt = V_function((self.X, self.Y), t)
            self.psi = self._apply_potential_and_nonlinear(self.psi, Vt)

            # half kinetic
            self.psi = self._apply_kinetic_half(self.psi)

            # normalize (to reduce numerical drift)
            norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx * self.dy)
            if norm == 0:
                raise RuntimeError("Wavefunction vanished (norm=0)")
            self.psi /= norm

            # compute relative phase between left and right halves
            left = self.psi[readout_mask]
            right = self.psi[~readout_mask]

            # compute average complex phase (argument of mean value)
            phi_left = np.angle(np.mean(left))
            phi_right = np.angle(np.mean(right))
            # store relative phase wrapped to [-pi, pi]
            delta = np.angle(np.exp(1j * (phi_left - phi_right)))
            delta_phi_ts[i] = delta

            # snapshots
            if snapshot_interval and (i % snapshot_interval == 0):
                psi_snapshots.append(self.psi.copy())

        psi_snapshots_arr = np.array(psi_snapshots) if psi_snapshots is not None else np.empty((0,))
        return SimulationResult(times=times, delta_phi=delta_phi_ts, psi_snapshots=psi_snapshots_arr)

    def save_time_series(self, result: SimulationResult, filename: str = "delta_phi.npz"):
        out = TIME_SERIES_DIR / filename
        np.savez_compressed(out, times=result.times, delta_phi=result.delta_phi)
        print(f"Saved time series to {out}")

    def plot_delta_phi(self, result: SimulationResult, filename: str = "delta_phi.png"):
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(result.times, result.delta_phi)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative phase (rad)")
        ax.set_title("Δφ(t) between left/right halves")
        fig.savefig(Path(FIGURES_DIR) / filename, bbox_inches="tight")
        plt.close(fig)