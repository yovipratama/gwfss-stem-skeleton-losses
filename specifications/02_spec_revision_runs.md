# Pre-specification: additional runs requested in Stage 3 review (round 1)

Written 2026-10-09, before any of these runs. Author decision (verbatim): "1a, just fix it, training k40 + seed kedua". Source items: Revision Roadmap REV-27 (clDice soft-skeleton depth) and REV-17 (training-run variance).

## R-K40: clDice at doubled resolution with a deeper soft skeleton
- Runs: random split, scale 2.0, loss clDice, skel_iters = 40, {SegFormer-B1, DeepLabV3+ R101} × seeds {0, 1, 2} = 6 runs. All other settings identical to the SQ3 runs. GPU: NVIDIA RTX PRO 6000 (the GPU of all SQ3 runs).
- Rationale: Shit et al. (2021) require the number of soft-skeleton iterations to be at least the largest structure radius; at ×2, 37–39% of stem skeleton pixels exceed radius 10 (reviewer R2).
- Analysis: same paired per-image procedure as SQ3 (seed-averaged per-image Wilcoxon, bootstrap 95% CI, Holm within metric over the comparisons of this analysis): clDice-k40 vs baseline (×2) and clDice-k40 vs clDice-k10 (×2), per architecture; primary metric stem IoU, plus clDice, skeleton recall, Betti-0 error, head/leaf/background IoU.
- Reporting rule: the ×2 conclusion ("adding either loss lowered stem IoU") is kept for clDice only if clDice-k40 is also lower than the ×2 baseline in both architectures under the consistency criterion; otherwise it is revised. Reported whatever the direction.

## R-S1: second seed for the LOIO baselines
- Runs: 10 LOIO folds × {SegFormer-B2, UPerNet} × baseline loss × seed 1 = 20 runs; SegFormer-B2 on A100 and UPerNet on RTX PRO 6000 (as for seed 0). Followed by offset sweeps (validation and test) for these runs.
- Uses: (i) re-test H1 with the area ratio A taken from the seed-1 baseline, so that A is estimated from a run independent of the seed-0 runs used for Δ; Δ is unchanged (seed-0 losses vs seed-0 baseline), and additionally Δ' = seed-0 loss vs seed-1 baseline is reported; (ii) report the seed-to-seed spread of baseline stem IoU and A per unit as an estimate of training-run variance in the LOIO setting.
- Same Spearman / one-sided permutation test (10,000) as H1; reported whatever the direction.
