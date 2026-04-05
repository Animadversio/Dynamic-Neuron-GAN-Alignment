# Dynamic Neuron–GAN Alignment

Code and data for:

> **Neuronal tuning aligns dynamically with object and texture manifolds across the visual hierarchy**  
> Binxu Wang & Carlos R. Ponce  
> *Nature Neuroscience*, 2026  
> DOI: [to be added upon publication]

---

## Overview

We evolved images to maximally activate individual neurons in primate visual cortex (V1, V4, IT) using two contrasting generative spaces:

- **DeePSim** (fc6-GAN) — texture-based, 4,096-dimensional latent space
- **BigGAN-deep-256** — object-based, photorealistic, 256-dimensional latent space

Optimization used CMA-ES running in parallel threads, one per GAN, with each neuron's firing rate as the fitness signal. Key findings:

- V1 and V4 align more strongly with the texture space; PIT aligns comparably with both
- PIT neurons respond to locally similar features across globally distinct images (local compositional code)
- Object-space features preferentially drive *late* temporal responses in PIT (>80 ms), while texture dominates early responses

---

## Repository Structure

```
Dynamic-Neuron-GAN-Alignment/
├── figure_reproduction/          # Scripts to reproduce each paper figure
│   ├── Figure2_reproduction.py
│   ├── Figure3_reproduction.py
│   ├── Figure4_reproduction.py
│   ├── Figure5_reproduction.py
│   ├── Figure6_reproduction.py
│   ├── FigureExt4_reproduction.py
│   └── FigureExt5_reproduction.py
├── source_data/                  # Pre-computed data files (CSV/PKL)
│   ├── Fig2/   Fig3/   Fig4/   Fig5/   Fig6/
│   ├── ExtendedFig4/
│   └── ExtendedFig5/
├── source_data_export/           # Notebooks showing how source data was generated
│   └── Figure*_source_data_export.ipynb
├── core/utils/                   # Reusable utilities
│   ├── GAN_utils.py              # Load BigGAN and DeePSim (upconvGAN)
│   ├── CNN_scorers.py            # Activation scoring via forward hooks
│   ├── Optimizers.py             # CMA-ES variants (CholeskyCMAES, HessCMAES)
│   ├── layer_hook_utils.py       # Dynamic layer name parsing and hook registration
│   ├── grad_RF_estim.py          # Receptive field estimation via backprop
│   ├── plot_utils.py             # Visualization helpers
│   ├── montage_utils.py          # Image montage creation
│   └── stats_utils.py            # Statistical utilities
├── neuro_data_analysis/          # Neural data loading library
│   ├── neural_data_lib.py        # Load .mat / .pkl neural recordings
│   ├── neural_data_utils.py      # Map electrode channels to visual areas
│   ├── image_comparison_lib.py   # Image similarity metrics
│   └── neural_tuning_analysis_lib.py
└── requirements.txt
```

---

## Reproducing the Figures

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain source data

Source data (CSV and PKL files) are distributed separately. Download from OSF:  
**https://osf.io/pre96**

Place the files under `source_data/` following the existing subdirectory structure (`Fig2/`, `Fig3/`, ..., `ExtendedFig5/`).

### 3. Run reproduction scripts

Each script is standalone — it reads from `source_data/` and displays figures:

```bash
# From the repo root:
python figure_reproduction/Figure2_reproduction.py   # Fig 2B, 2D
python figure_reproduction/Figure3_reproduction.py   # Fig 3G
python figure_reproduction/Figure4_reproduction.py   # Fig 4A–4D
python figure_reproduction/Figure5_reproduction.py   # Fig 5A, 5C
python figure_reproduction/Figure6_reproduction.py   # Fig 6C–6E
python figure_reproduction/FigureExt4_reproduction.py  # Ext Fig 4B, 4C
python figure_reproduction/FigureExt5_reproduction.py  # Ext Fig 5C–5E
```

---

## Figure Map

| Figure | Script | What it shows |
|--------|--------|---------------|
| 2B, 2D | `Figure2_reproduction.py` | Example paired evolution (PIT site, Exp 155): trajectory and stacked PSTHs |
| 3G | `Figure3_reproduction.py` | Image similarity (ResNet/LPIPS) vs. PSTH difference across visual areas |
| 4A–4D | `Figure4_reproduction.py` | Success rates, initial/final activation, trajectories, convergence time constants by area |
| 5A, 5C | `Figure5_reproduction.py` | Population PSTHs and time-binned (10 ms) trajectories across V4/IT |
| 6C–6E | `Figure6_reproduction.py` | Example tuning curve, tuning heatmap, peak-location and shape-type distributions |
| Ext 4B–C | `FigureExt4_reproduction.py` | Temporal attribution and time-binned trajectories (multiple window sizes) |
| Ext 5C–E | `FigureExt5_reproduction.py` | RealNVP depth/dimension optimization; latent code linearity in Caffenet |

---

## Source Data Export Notebooks

The `source_data_export/` notebooks document exactly how each source data file was generated from the raw neural recordings. They **require access to the raw `.mat` recording files** (not publicly distributed; see Data Availability section of the paper).

Set the `MATROOT` environment variable to the directory containing the `.mat` files before running:

```bash
export MATROOT=/path/to/Mat_Statistics
jupyter notebook source_data_export/Figure2_source_data_export.ipynb
```

---

## Core Utilities

The `core/utils/` module provides reusable components for activation maximization experiments:

- **`GAN_utils.py`** — `upconvGAN(name)` loads DeePSim; `BigGAN` loaded via `pytorch_pretrained_biggan`
- **`CNN_scorers.py`** — `TorchScorer(model, layer, unit)` extracts activations via forward hooks
- **`Optimizers.py`** — `CholeskyCMAES`, `HessCMAES`: CMA-ES variants with optional Hessian preconditioning
- **`layer_hook_utils.py`** — `get_module_names(model)`, `register_hook_by_module_name(model, layer_name)`

---

## Citation

```bibtex
@article{wang2026neuronal,
  title   = {Neuronal tuning aligns dynamically with object and texture manifolds across the visual hierarchy},
  author  = {Wang, Binxu and Ponce, Carlos R.},
  journal = {Nature Neuroscience},
  year    = {2026},
}
```

---

## Data Availability

Raw neural recording data are available at **OSF: https://osf.io/pre96** and upon request from C.R. Ponce.

Pre-trained GAN weights are downloaded automatically on first use:
- DeePSim: `binxu/DeePSim_DosovitskiyBrox2016` (HuggingFace)
- BigGAN: `pytorch_pretrained_biggan`

## License

See [LICENSE](LICENSE).
