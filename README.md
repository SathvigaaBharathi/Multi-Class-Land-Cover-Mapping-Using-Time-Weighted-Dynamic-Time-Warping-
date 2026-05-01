# TWDTW State-Scale LULC Pipeline

A Python-native pipeline for multi-class Land Use / Land Cover (LULC) mapping using **Time-Weighted Dynamic Time Warping (TWDTW)** on fused Sentinel-1/2 time series. Applied to Maharashtra, India (~308,000 km²) for the Rabi 2023–24 season.

---

## Pipeline

| Script | What it does |
|---|---|
| `01_download_s2_composites.py` | Download monthly Sentinel-2 tiles from MPC STAC |
| `01b_download_s1_composites.py` | Download monthly Sentinel-1 GRD tiles from MPC STAC |
| `02_preprocess_stack.py` | Fuse S1 + S2, apply physical conversions, build 56-band stack |
| `03_classify_twdtw.py` | Extract signatures, classify via TWDTW, merge final mosaic |

Run scripts in order. Stages 1 and 1b can run in parallel.

---

## Installation

```bash
conda env create -f environment.yml
conda activate twdtw_env
```

Or with pip:
```bash
pip install -r requirements.txt
```

---

## Quick Start

1. Set your boundary file and output paths in the `SETTINGS` block of each script.
2. Set your training polygon file and class field name in `03_classify_twdtw.py`.
3. Run:

```bash
python 01_download_s2_composites.py
python 01b_download_s1_composites.py
python 02_preprocess_stack.py
python 03_classify_twdtw.py
```

---

## Output Classes

Defined by your training polygon labels. For Maharashtra Rabi 2023–24:

| Code | Class |
|---|---|
| 1 | Cropland |
| 2 | Forest |
| 3 | Water |
| 4 | Settlements |
| 5 | Barrenland |

---

## Citation

```
Sathvigaa Bharathi (2026). State-Scale Multi-Class Land Cover Mapping Using
Time-Weighted Dynamic Time Warping on Fused Sentinel-1/2 Time Series:
A Python-Native Scalable Pipeline. Amrita Vishwa Vidyapeetham, Amaravati, India.
```

---

## Acknowledgements

- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) for open STAC access to Sentinel archives
- Maus et al. (2016) for the original TWDTW formulation
- Computational resources provided by Amrita Vishwa Vidyapeetham, Amaravati, India

---

## License

MIT License. See `LICENSE` for details.
