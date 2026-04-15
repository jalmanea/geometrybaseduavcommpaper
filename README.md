# Geometry-Based UAV Communication Deployment for Coverage and Interference Management

Code and data accompanying the paper:

> **Geometry-Based UAV Communication Deployment for Coverage and Interference Management**
> Abdulrahman Almanea, Balsam Alkouz, and Basem Shihada
> King Abdullah University of Science and Technology (KAUST)

---

## Overview

This repository contains the simulation framework, experiment scripts, and analysis notebooks for evaluating an analytic geometry-based heuristic for UAV base station deployment. The heuristic derives complete UAV swarm deployments (positions, altitudes, and antenna orientations) from closed-form geometric relationships at negligible computational cost, without iterative optimization.

Two deployment methods are compared:

| Method | Positions | Altitude | UL Orientation | BH Orientation |
|--------|-----------|----------|----------------|----------------|
| **K-means baseline** | K-means centroids | Fixed 150 m | Nadir | MST |
| **Analytic heuristic** | K-means centroids | Beam-footprint matched to cluster spread | Nadir | MST |

The key differentiator is per-drone altitude adaptation: each drone flies at the height where its −3 dB beam footprint matches the spatial spread of its assigned user cluster, providing vertical beam separation that suppresses co-channel interference as fleet size grows.

---

## Repository Structure

```
.
├── dronecomm/                  # Core simulation library
│   ├── antenna.py              # Parametric 3GPP TR 38.901 antenna model
│   ├── channel.py              # Probabilistic air-to-ground LoS/NLoS channel
│   ├── config.py               # Dataclass configuration with YAML loading
│   ├── geometry.py             # 3D vector math and coordinate transforms
│   ├── heuristic.py            # Analytic heuristic and K-means baseline
│   ├── interference.py         # Inter-drone and user interference computation
│   ├── metrics.py              # Coverage, throughput, and SINR metrics
│   ├── scenario.py             # User distribution generation (clustered/uniform/hotspot)
│   ├── simulation.py           # Monte Carlo simulation runner
│   ├── sinr.py                 # SINR computation for ground users
│   └── visualization.py        # Scenario plots, SINR heatmaps, antenna patterns
│
├── scripts/
│   ├── run_heuristic_experiment.py     # Synthetic configuration sweep (main experiment)
│   ├── run_telecom_statistical.py      # Real-world Telecom Italia experiment
│   ├── run_per_drone_angle_sweep.py    # UL orientation validation sweep
│   ├── extract_telecom_snapshots.py    # Batch-extract balanced telecom snapshots
│   ├── extract_telecom_users.py        # Extract user positions from a single snapshot
│   └── plot_environment_figure.py      # Generate Fig. 1 (system illustration)
│
├── notebooks/
│   ├── analyze_heuristic.ipynb             # Figs. 2 & 3: synthetic sweep results
│   ├── analyze_telecom_experiment.ipynb    # Exploratory telecom analysis
│   ├── analyze_telecom_statistical.ipynb   # Fig. 4: telecom coverage vs. fleet size
│   └── analyze_per_drone_angle_sweep.ipynb # Table 2: orientation validation
│
├── paper/
│   ├── heuristic_paper_v2.tex  # Paper source
│   ├── references.bib          # Bibliography
│   └── figures/                # Compiled figures (PDF + PNG)
│
├── configs/
│   └── default.yaml            # Default simulation parameters
├── environment.yml             # Conda environment specification
└── pyproject.toml              # Python package definition
```

---

## Installation

### 1. Clone and create the environment

```bash
git clone <repo-url>
cd geometrybaseduavcommpaper

conda env create -f environment.yml
conda activate dronecomm
pip install -e .
```

### 2. Obtain the Telecom Italia dataset

The real-world experiments use the **Telecom Italia Big Data Challenge — Trentino** dataset:

> Barlacchi, G. et al. (2015). *A multi-source dataset of urban life in the city of Milan and the Province of Trentino*. Scientific Data, 2, 150055.
> Dataset DOI: [10.7910/DVN/QLCABU](https://doi.org/10.7910/DVN/QLCABU) (Harvard Dataverse)

Download the `sms-call-internet-tn-*.txt` files and place them in `dataverse_files/` at the repository root.

---

## Running the Experiments

### Synthetic configuration sweep

Evaluates both methods over 1,500 configurations:
- Fleet sizes: N ∈ {1, …, 50}
- User counts: M ∈ {100, 200, …, 1000}
- Distributions: clustered, hotspot, uniform
- 20 Monte Carlo seeds per configuration

```bash
# Quick smoke-test (a few configs, reduced seeds)
python scripts/run_heuristic_experiment.py --quick

# Single configuration
python scripts/run_heuristic_experiment.py --n-drones 10 --target-users 400

# List all tasks in the sweep grid
python scripts/run_heuristic_experiment.py --list-tasks

# Run one task by index (for job array parallelism)
python scripts/run_heuristic_experiment.py --task-id 0 --output results/heuristic/
```

Results are written as JSON files (one per task) to the output directory.

---

### Real-world Telecom Italia experiment

Evaluates deployment methods on real-world user distributions extracted from the Telecom Italia dataset. Requires the dataset in `dataverse_files/`.

**Step 1 — extract snapshots:**

```bash
# Extract all balanced snapshots (800 users each, saved as JSON)
python scripts/extract_telecom_snapshots.py --out-dir results/telecom/snapshots

# Preview the snapshot design without extracting
python scripts/extract_telecom_snapshots.py --list

# Extract a single snapshot for inspection
python scripts/extract_telecom_users.py --date 2013-11-04 --hour 10 --visualize
```

**Step 2 — run the experiment:**

```bash
# Quick smoke-test
python scripts/run_telecom_statistical.py --quick

# List all tasks
python scripts/run_telecom_statistical.py --phase main --list-tasks

# Run one task by index
python scripts/run_telecom_statistical.py --phase main --task-id 0 --output results/telecom/
```

---

### UL orientation validation

Verifies that nadir-pointing is near-optimal for the analytic heuristic by sweeping each drone's UL tilt (0°–60°, 1° steps) and azimuth (0°–355°, 5° steps) independently across 36 configurations.

```bash
# Quick test
python scripts/run_per_drone_angle_sweep.py --quick

# Single configuration
python scripts/run_per_drone_angle_sweep.py --n-drones 10 --target-users 200

# List all tasks
python scripts/run_per_drone_angle_sweep.py --list-tasks

# Run one task by index
python scripts/run_per_drone_angle_sweep.py --task-id 0 --output results/angle_sweep/
```

---

### System illustration figure (Fig. 1)

```bash
python scripts/plot_environment_figure.py
# Output: paper/figures/environment_figure.pdf
```

---

## Analysis Notebooks

Once results are generated, open the notebooks to reproduce the paper figures:

| Notebook | Produces | Description |
|----------|----------|-------------|
| `analyze_heuristic.ipynb` | Figs. 2 & 3 | Coverage by distribution; coverage vs. fleet size on hotspot |
| `analyze_telecom_statistical.ipynb` | Fig. 4 | Coverage vs. fleet size over 24 Telecom Italia snapshots |
| `analyze_telecom_experiment.ipynb` | — | Exploratory per-snapshot and per-hour telecom analysis |
| `analyze_per_drone_angle_sweep.ipynb` | Table 2 | Mean per-drone coverage gain from orientation optimization |

---

## Simulation Parameters

Key parameters (see `configs/default.yaml` for the full set):

| Parameter | Value |
|-----------|-------|
| Area size | 2000 × 2000 m |
| Carrier frequency | 2.0 GHz |
| Bandwidth | 10 MHz |
| UL transmit power | 30 dBm |
| BH transmit power | 27 dBm |
| Noise PSD | −174 dBm/Hz |
| SINR threshold | 3 dB |
| K-means baseline altitude | 150 m |
| Analytic altitude range | [50, 300] m |
| UL antenna (G_max, θ_3dB, A_SLA) | 18 dBi, 60°, 20 dB |
| BH antenna (G_max, θ_3dB, A_SLA) | 12 dBi, 30°, 25 dB |
| Channel model | Al-Hourani urban probabilistic LoS |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{almanea2025geometry,
  title     = {Geometry-Based {UAV} Communication Deployment for Coverage and Interference Management},
  author    = {Almanea, Abdulrahman and Alkouz, Balsam and Shihada, Basem},
  booktitle = {},
  year      = {2025},
}
```
