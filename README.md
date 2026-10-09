# Skeleton-based losses versus threshold calibration for wheat stem segmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23257262.svg)](https://doi.org/10.5281/zenodo.23257262)

Code, data splits, experiment specifications and per-image results for:

> Pratama, Y., Idris, M.Y., Mohd Sidik, M.K., Toscany, A.N., Rasywir, E. *Skeleton-based losses versus threshold calibration for wheat stem segmentation across ten institutions.* (manuscript submitted)

The study compares two skeleton-based connectivity losses, **clDice** and **Skeleton Recall**, added to the stem channel of a cross-entropy + Dice objective, on the [GWFSS v1.0](https://huggingface.co/datasets/GlobalWheat/GWFSS_v1.0) wheat organ segmentation dataset: four architectures (DeepLabV3+ R101, SegFormer-B1/B2, UPerNet ConvNeXt-T), three seeds, a random split, a held-out institution, doubled resolution, a leave-one-institution-out (LOIO) experiment, decision-threshold calibration and image-derived stem traits.

## Contents

| Folder | Content |
|---|---|
| `src/gwfss_stem/` | Dataset preparation and splits (`data.py`), models (`models.py`), losses (`losses.py`), metrics (`metrics.py`), training/evaluation/threshold sweeps (`engine.py`), statistics (`analysis.py`), run plans (`plan.py`) |
| `notebooks/` | Google Colab notebooks used to run everything (00 setup and data audit → 01 training → 02 evaluation → 03 analysis → 04 threshold control → 05a–c LOIO → 06a–b revision runs) |
| `splits/splits.json` | All splits: `random`, `region`, `loio_<institution>` |
| `specifications/` | Time-stamped specifications written before the LOIO and revision runs, with SHA-256 hashes |
| `results/` | One folder per run (`<split>/<architecture>/<loss>/scale<s>/seed<n>/`): `config.json` (settings, library and GPU versions), `train_log.csv`, `metrics_<tag>.csv` (per-image metrics), `summary_<tag>.json`, and for threshold analyses `sweep_val.csv` / `sweep_test.csv` (per-image confusion counts and stem traits for every stem-logit offset), `metrics_thr.csv`, `offset_val.csv` |
| `analysis/` | Scripts and outputs: `main/` (paired tests, consistency, threshold control), `loio/` (H1–H5), `review/` (analyses added in peer review) |
| `figures/` | Figures and the scripts that generate them |

Model weights and the GWFSS images are not included. All statistics in the paper can be recomputed from the per-image tables in `results/` without retraining.

## Reproducing

```bash
pip install -r requirements.txt
```

Training and evaluation were run on Google Colab (see `notebooks/`); set `PROJECT` in the setup cell to a folder on your Google Drive that contains `src/` and `splits/`. Locally, the analysis scripts are run from the repository root, e.g.

```bash
python analysis/loio/loio_h1_h3.py
python analysis/review/rev_block1.py
python figures/scripts/fig3_loio_area_ratio.py .
```

Scripts that read the reference masks (trait exclusions, stem width) expect the dataset converted with `gwfss_stem.data.prepare_dataset("data_cache/gw")`, which downloads the labelled parquet shards from the Hugging Face Hub.

Hardware differed between experiments (Colab T4, A100 and RTX PRO 6000; see each `config.json`); within an experiment, all losses and seeds of an architecture share the same GPU type and number of workers.

## Licences

- Code: MIT (see `LICENSE`).
- Result tables, specifications and figures: CC BY 4.0.
- GWFSS images and masks: CC BY 4.0, © the GWFSS authors (Wang et al., 2025, *Plant Phenomics* 7, 100084, https://doi.org/10.1016/j.plaphe.2025.100084); not redistributed here.

## Citation

See `CITATION.cff`. Archived on Zenodo: https://doi.org/10.5281/zenodo.23257262 (all versions).
