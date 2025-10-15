# BEC_DM: Astrophysical BEC Simulation for Ultralight Dark Matter Detection

> **Status:** Active research project (v0.1, preprint-ready)  
> **Keywords:** Bose–Einstein condensate, ultralight dark matter, phase shift sensing, numerical simulation, Python

## Overview

This repository contains a **dimensionless Python simulation** of an astrophysical **Bose–Einstein Condensate (BEC)** designed to explore **phase-shift signatures** induced by **ultralight dark matter (UDM)**. It includes:
- A modular simulation core (BEC dynamics, environment, DM potential)
- Comparative methods (e.g., **two-phase** and **two-state** approaches)
- Utilities for diagnostics (**PSD**, phase traces, summary metrics)
- A clean **visualization script** for 2D density/phase plots

> This is an evolving research codebase. A citable Zenodo DOI will be added upon first release.

## Features

- Dimensionless BEC evolution with configurable potentials & noise  
- Dark-matter coupling as an external oscillatory perturbation  
- **Two-phase** and **Two-state** detection strategies  
- Built-in plotting: PSD curves, phase-time traces, comparison panels  
- Reproducible runs with seeds + config files  
- Ready for **Google Colab** and local execution

## Repository Structure

```
BEC_DM/
├─ src/
│  ├─ bec_simulation.py          # Core BEC simulation class
│  ├─ environment.py             # Astrophysical-like environment / noise
│  ├─ dm_potential.py            # Ultralight DM coupling models
│  ├─ two_phase_dimless.py       # Two-phase method
│  ├─ two_state.py               # Two-state method (comparison)
│  └─ __init__.py
├─ scripts/
│  ├─ run_simulation.py          # CLI entry point
│  └─ bec_visual.py              # 2D visualization utilities (scatter/heatmap)
├─ configs/
│  ├─ default.yaml               # Baseline config (dimensionless)
│  └─ examples/                  # Additional presets
├─ figures/                      # Auto-generated plots
├─ results/                      # Metrics, logs, artifacts
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## Quick Start

### Option A — Local

```bash
git clone https://github.com/<YOUR_USERNAME>/BEC_DM.git
cd BEC_DM
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the baseline simulation:

```bash
python -m scripts.run_simulation --config configs/default.yaml
```

Generate visualizations (from saved outputs):

```bash
python -m scripts.bec_visual --input results/latest_run/ --out figures/
```

### Option B — Google Colab

```python
# In Colab
!git clone https://github.com/<YOUR_USERNAME>/BEC_DM.git
%cd BEC_DM
!pip install -r requirements.txt

# Add project to sys.path if needed:
import sys, os
sys.path.append(os.getcwd())

# Run
!python -m scripts.run_simulation --config configs/default.yaml
# Visualize
!python -m scripts.bec_visual --input results/latest_run/ --out figures/
```

## Configuration

All key parameters live in YAML configs:
- **simulation:** grid size, dt, total_time, seed  
- **dm:** coupling strength, frequency, on/off window  
- **environment:** noise level, background potential switches  
- **outputs:** save paths, figure toggles  

Example (`configs/default.yaml`):

```yaml
simulation:
  grid: [256, 256]
  dt: 1e-3
  total_time: 15.0
  seed: 5655

dm:
  enable: true
  g_dm: 2.5e-4
  omega_dm: 0.75
  window: [3.0, 9.0]   # when DM perturbation is active (dimensionless time)

environment:
  noise_level: 0.02
  trap_strength: 0.1

outputs:
  save_dir: "results/default"
  save_every: 50
  make_figures: true
```

## What You Get (Outputs)

- `results/<run_id>/metrics.json` → phase offsets, PSD peaks, SNR, run metadata  
- `figures/` →  
  - `phase_trace.png` — net phase vs. time (DM on/off highlighted)  
  - `psd.png` — power spectral density with DM peak annotation  
  - `density2d.png` — 2D density snapshot (from `bec_visual.py`)  
  - `comparison.png` — two-phase vs two-state summary (if both run)


## Reproducibility Checklist

- Set `seed` in config (and any RNG in `environment.py`)  
- Commit the exact `configs/*.yaml` used for each figure  
- Record git commit hash (stored in `metrics.json` if available)  
- Keep `requirements.txt` pinned (e.g., `numpy==...`, `matplotlib==...`)

## Methods (Short)

- **Two-phase:** track the global (or spatially averaged) phase; DM coupling induces a measurable phase shift against baseline evolution.  
- **Two-state:** prepare/propagate two slightly different internal states; measure relative phase/density differences to enhance DM contrast.

## Roadmap

- [ ] Parameter sweep tool (grid over `g_dm`, `omega_dm`, noise)  
- [ ] Automated figure panels for the paper (1-click export)  
- [ ] Configurable traps (harmonic, box, rotating)  
- [ ] Add unit mapping helpers for astrophysical scaling  
- [ ] Preprint & **Zenodo v1** (DOI)  
- [ ] Student-journal submission (v2)

## Contributing

PRs and issues welcome! Please:
1. Open an issue describing your change  
2. Keep functions documented and tests minimal but meaningful  
3. Add/update a config in `configs/examples/` for new experiments

## License

**MIT** (permissive). See `LICENSE`.  

## Cite This Work

A **Zenodo DOI** will be added after the first release. For now, you can cite as a preprint:

**APA (temporary):**  
Ercili, S. (2025). *Simulation of Astrophysical Bose–Einstein Condensates for Ultralight Dark Matter Detection* (v0.1). GitHub. https://github.com/selayercili/BEC_DM

**BibTeX (temporary):**
```bibtex
@misc{Ercili_BEC_DM_2025,
  author       = {Selay Ercili},
  title        = {Simulation of Astrophysical Bose–Einstein Condensates for Ultralight Dark Matter Detection},
  year         = {2025},
  howpublished = {\url{https://github.com/selayercili/BEC_DM}},
  note         = {Version 0.1, preprint}
}
```

## Contact

Questions or collaboration ideas?  
**Selay Ercili** — open an issue or reach me via the email on my GitHub profile.
