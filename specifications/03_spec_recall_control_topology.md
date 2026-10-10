# Pre-specification: recall-weighted control, spatial topology analysis and loss-weight sensitivity

Written 2026-10-10, before any of the runs or analyses below. Author decision (verbatim): "saya pilih B" (option B: text revision + R4 spatial connectivity + R2 recall-weighted control, R3 if time). Source: second external pre-submission review (stage4/response_external_review_2.md), items R2, R3, R4.

All runs use the LOIO protocol of the seed-0 LOIO runs (8,000 iterations, batch 8, 512-pixel crops, AdamW, same augmentation, final iterate, seed 0, two data-loading workers), with SegFormer-B2 on an NVIDIA A100 and UPerNet (ConvNeXt-T) on an NVIDIA RTX PRO 6000, as for their seed-0 counterparts. Nothing else is changed.

## Block A (R2): recall-weighted loss without a skeleton term

- Loss "tversky": L = L_CE + L_Dice + λ · L_T with λ = 1, where L_T = 1 − (TP + s)/(TP + α·FP + β·FN + s) is computed on the softmax stem probability p and the binary stem mask y over the whole mini-batch (TP = Σ p·y, FP = Σ p·(1 − y), FN = Σ (1 − p)·y, ignored pixels excluded), α = 0.3, β = 0.7, s = 1 (Salehi et al., 2017). Not tuned, as for the skeleton losses.
- Runs: 10 LOIO folds × {SegFormer-B2, UPerNet} × seed 0 = 20 runs, each followed by evaluation (metrics_ss) and offset sweeps on the validation and test sets.
- Unit-level quantities as in the LOIO analysis: Δ_T = mean per-image change in stem IoU of Tversky vs the seed-0 baseline; area ratio A from the seed-0 baseline.
- H6a: Δ_T is negatively associated with A (Spearman ρ over the 20 units, one-sided permutation test, 10,000 permutations; also over the ten institution means). Reported whatever the direction.
- H6b: for each skeleton loss, D = Δ_skel − Δ_T per unit; mean D over the 20 units with a 95% bootstrap CI resampling institutions (both architectures kept together, 10,000 resamples), as for Table 5. Decision rule: the skeleton loss is "better than the recall-weighted control" if the CI lies above 0, "worse" if below 0, otherwise "not distinguishable"; the two are called "equivalent" only if the CI lies within ±1 point.
- Secondary (descriptive, same procedures as before): Tversky stem precision, recall, area ratio, head/leaf/background IoU vs baseline; Tversky with the ten-image offset (T10, the same 20 draws as before) vs baseline + T10; length bias and absolute length error (Table 6 procedure).

## Block B (R4): spatial topology of the stem prediction (inference only)

- New per-image measures, computed for every offset of the existing grid (−2.0 … +8.0, step 0.5) on the test set of every seed-0 LOIO model (baseline, clDice, Skeleton Recall, Tversky; 80 models). Components are 8-connected components of the stem class within valid pixels; a predicted component P and a reference component G are linked if they share at least one pixel.
  - n_pred, n_ref: numbers of components;
  - merges: number of P linked to ≥ 2 G; excess_merge = Σ over P of max(0, links − 1);
  - splits: number of G linked to ≥ 2 P; excess_split = Σ over G of max(0, links − 1);
  - spurious: P with no link; missed: G with no link.
- Comparisons at matched predicted area (procedure of the matched-area analysis, Section 4.4): the comparison model is evaluated at the grid offset whose total predicted stem area over the test set is closest to that of the treatment model at offset 0.
- H7 (structure specific to the skeleton term): at matched area, each skeleton loss has fewer predicted components per image and a lower excess_split than Tversky. Unit-level differences over 20 units, mean with 95% institution-cluster bootstrap CI; supported if the CI lies below 0.
- H8 (fragmentation vs over-merging): for each loss (clDice, Skeleton Recall, Tversky) vs the area-matched baseline, report the mean unit-level change in excess_merge, excess_split, spurious and missed. Decision rule: the reduction in fragmentation is described as attributable to over-merging if the mean increase in excess_merge is at least as large as the mean decrease in excess_split; otherwise as reduced splitting of reference stems. Caveat stated in advance: semantic reference components already merge crossing stems, so merges are counted relative to the reference, not to individual plants.

## Block C (R3, run only if scheduled): loss-weight sensitivity

- Runs: SegFormer-B2, seed 0, λ ∈ {0.25, 0.5} × {clDice, Skeleton Recall} × four LOIO folds = 16 runs. Folds chosen by a rule fixed before the runs from the existing seed-0 SegFormer-B2 baseline area ratios: the two lowest (UTokyo 0.23, USASK 0.40) and the two highest (INRAE 0.83, NJAU 0.90).
- Analysis (descriptive): per fold and loss, stem IoU, area ratio, stem precision/recall and leaf/background IoU as a function of λ ∈ {0.25, 0.5, 1}, with paired per-image CIs vs the baseline. Expectation stated in advance: smaller λ gives a smaller change in predicted area and smaller side effects on leaf and background, and smaller gains where A is low.

## Reporting

All results are reported whatever their direction. Blocks A–C are described as added after review; Block B results at offset 0 and at matched area are both reported. Deviations from this specification will be recorded in the specifications README, not by editing this file.
