#!/usr/bin/env python3
# scripts/bec_visual.py
# Visualize BEC density from saved solver outputs (physically accurate |psi|^2)

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── project paths ─────────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]  # repo root
RESULTS_ROOT = ROOT / "results" / "two_state_dimless"
FIELDS_DIR   = RESULTS_ROOT / "fields"
VIS_DIR      = RESULTS_ROOT / "visuals"
VIS_DIR.mkdir(parents=True, exist_ok=True)

def load_state(tag: str):
    """Load final wavefunction for state tag (e.g., 'c1' or 'c2')."""
    f = FIELDS_DIR / f"dimless_{tag}_final.npz"
    if not f.exists():
        raise FileNotFoundError(
            f"Missing {f}. Run scripts/two_phase_dimless.py first to generate it."
        )
    data = np.load(f)
    psi = data["psi"]
    x   = data["x"]
    y   = data["y"]
    return psi, x, y

def plot_heatmap(psi, x, y, out_path, title="BEC density (|ψ|²)"):
    """2D density heatmap."""
    density = np.abs(psi)**2
    Xmin, Xmax = x.min(), x.max()
    Ymin, Ymax = y.min(), y.max()

    fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=140)
    im = ax.imshow(density,
                   extent=[Xmin, Xmax, Ymin, Ymax],
                   origin="lower", cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|\psi(x,y)|^2$")
    ax.set_xlabel("x (trap units)")
    ax.set_ylabel("y (trap units)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_scatter(psi, x, y, out_path, title="BEC density (scatter)"):
    """
    Colored scatter view of |ψ|² (decimates points to keep size reasonable).
    """
    density = np.abs(psi)**2
    # Build grid coordinates
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Decimate if grid is large
    stride = 2 if (psi.shape[0] * psi.shape[1] > 64_000) else 1
    xs = X[::stride, ::stride].ravel()
    ys = Y[::stride, ::stride].ravel()
    cs = density[::stride, ::stride].ravel()

    fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=140)
    sc = ax.scatter(xs, ys, c=cs, s=6, marker='s', cmap="viridis")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$|\psi(x,y)|^2$")
    ax.set_xlabel("x (trap units)")
    ax.set_ylabel("y (trap units)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def side_by_side(psi_left, x, y, psi_right, out_path,
                 title_left="c1", title_right="c2"):
    """Side-by-side heatmaps for quick comparison."""
    density_L = np.abs(psi_left)**2
    density_R = np.abs(psi_right)**2
    Xmin, Xmax = x.min(), x.max()
    Ymin, Ymax = y.min(), y.max()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=140)
    for ax, dens, ttl in zip(axes, (density_L, density_R), (title_left, title_right)):
        im = ax.imshow(dens, extent=[Xmin, Xmax, Ymin, Ymax],
                       origin="lower", cmap="viridis", aspect="auto")
        ax.set_xlabel("x (trap units)")
        ax.set_ylabel("y (trap units)")
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|\psi|^2$")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    # Load both states if available
    psi1, x, y = load_state("c1")
    psi2, _, _ = load_state("c2")

    # Individual visuals (heatmap + scatter)
    plot_heatmap(psi1, x, y, VIS_DIR / "bec_density_heatmap_c1.png", title="BEC density |ψ|² (state c1)")
    plot_scatter(psi1, x, y, VIS_DIR / "bec_density_scatter_c1.png", title="BEC density (scatter, c1)")

    plot_heatmap(psi2, x, y, VIS_DIR / "bec_density_heatmap_c2.png", title="BEC density |ψ|² (state c2)")
    plot_scatter(psi2, x, y, VIS_DIR / "bec_density_scatter_c2.png", title="BEC density (scatter, c2)")

    # Comparison figure
    side_by_side(psi1, x, y, psi2, VIS_DIR / "bec_density_heatmap_side_by_side.png",
                 title_left="c1 (|ψ|²)", title_right="c2 (|ψ|²)")

    print("[OUTPUT] Visuals →", VIS_DIR)

if __name__ == "__main__":
    main()
