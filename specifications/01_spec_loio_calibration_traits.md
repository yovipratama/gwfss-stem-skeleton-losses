# Pre-specification: leave-one-institution-out (LOIO), target-site threshold calibration, image-derived stem traits

Written 2026-10-08, before any LOIO model was trained and before any trait or few-label calibration number was computed. Approved scope (author, 2026-10-08): "kuota colab masih banyak, aman" — full LOIO with two architectures.

What is already known when this is written (and therefore not a prediction): all SQ1–SQ3 results, the descriptive area-ratio analysis (Table 10) and the validation-tuned threshold control (Section 4.7, Table 11). The hypotheses below are derived from those results and are tested on new data (eight institutions never used as test sets: Arvalis, CIMMYT, ETHZ, INRAE, NJAU, RRES, ULiege, USASK, UTokyo; UQ is re-tested with a different training set).

## E1. LOIO experiment (exp-LOIO)
- Splits: `loio_<institution>` in the frozen `splits/splits.json` (created at Stage 1, 2026-10-05): test = one institution (108–110 images); validation = the next institution in alphabetical order (wrapping around); training = the remaining eight institutions (876–878 images).
- Runs: 10 folds × {SegFormer-B2, UPerNet (ConvNeXt-T)} × {base, clDice, Skeleton Recall} × seed 0 = 60 runs. Hyperparameters identical to the main study (engine.DEFAULTS), native resolution, final iterate.
- Choice of architectures: the two strongest in-distribution (Table 3), one transformer and one convolutional encoder. Fixed before running.
- One seed per run: seed variation is small relative to between-institution variation (main study); the unit of replication is the institution.

## E2. Hypotheses
Let, for each fold × architecture unit u (20 units): A_u = baseline predicted / reference stem area on the test institution (dataset-level sums); Δ_u(L) = mean per-image change in stem IoU of loss L relative to baseline.

- **H1 (primary, predictive).** For each loss L, Δ_u(L) is negatively associated with A_u: Spearman ρ < 0 across the 20 units, one-sided p < 0.05 (permutation test, 10,000 permutations). Reported also per architecture (10 units each, descriptive).
- **H2.** For each loss, Δ_u(L) > 0 for most units with A_u < 0.7, and Δ_u(L) ≤ 0 for most units with A_u ≥ 0.85 (descriptive; counts reported). The thresholds 0.7 and 0.85 bracket the values observed so far (held-out 0.33–0.53; in-distribution 0.84–0.94).
- **H3 (per fold, secondary).** Per fold and architecture, each loss vs baseline: paired per-image Wilcoxon test, bootstrap 95% CI (2,000 resamples), Holm correction within each metric over all 40 LOIO comparisons. Metrics as in the main study (stem IoU, stem clDice, skeleton recall, Betti-0 error; head, leaf, background IoU as secondary). Reported as counts of folds with improvement / degradation.

## E3. Threshold controls
For every LOIO run (60) and every region-split run (36), a sweep stores per-image confusion counts and stem trait values for the offset grid b ∈ {−2.0, −1.5, …, +8.0} on the validation (LOIO only) and test sets. All selections below are done offline from these sweeps.
- **T-val**: offset chosen on the fold's validation institution (as in Section 4.7).
- **T-k (few-label target calibration)**: offset chosen on k labelled images drawn at random from the test institution, k ∈ {5, 10}; the selection maximises stem IoU summed over the k images (ties → smallest |b|); evaluation on the remaining test images of that institution. 20 random draws per run (draw seeds 0–19, shared by all conditions of the same fold so that every condition is evaluated on identical images); results are averaged over draws.
- **H4.** With k = 10, the target-calibrated baseline reaches at least the stem IoU of the losses without an offset in most units (descriptive; mean difference and count of units reported). This is the practical alternative to retraining.
- Comparisons for T-k: base+T-k vs loss (no offset) and vs loss+T-k, on the same held-out images, per unit; summarised across units (mean, range, count). No per-image test, because images differ between draws.

## E4. Image-derived stem traits (all conditions with per-image results: random, region, LOIO; native resolution; no offset, T-val, T-k)
- Traits per image, computed from prediction and reference in the same way:
  - visible stem length: number of pixels of the 8-connected skeleton of the stem mask (pixels, uncalibrated);
  - stem area fraction: stem pixels / labelled pixels;
  - stem fragments: number of 8-connected stem components.
- Images with reference stem area < 500 px are excluded from relative errors (reported count).
- Agreement per condition: signed relative error (pred − ref)/ref, summarised by the median (bias); absolute relative error; Lin's concordance correlation coefficient (CCC) across images.
- **H5.** On test institutions where the baseline under-segments (A_u < 0.7), the baseline underestimates visible stem length and stem area fraction (median signed relative error < 0), and each loss reduces the absolute relative error (paired per-image Wilcoxon, Holm within trait). In-distribution (random split), the losses do not reduce it.
- Stem count per plant and calibrated length (cm) are not estimable from GWFSS (semantic, uncalibrated masks) and will not be claimed.

## E5. Reporting rules
- All results reported whatever their direction; LOIO, T-k and trait analyses are labelled as added after the Stage 2.5 integrity review, with this file cited as their specification.
- Any deviation from this file is logged in the decision log with date and reason before the affected analysis is run.
