# Experiment specifications

These files were written before the experiments they describe were run, and they are released unedited so that their content can be checked against the hashes below. They therefore contain the working notes of the time (including short quotations of the first author's decisions, partly in Indonesian) and refer to internal file names.

| File | Written | Describes | SHA-256 |
|---|---|---|---|
| `01_spec_loio_calibration_traits.md` | 2026-10-08, before any LOIO model was trained | Leave-one-institution-out experiment (H1–H3), threshold calibration (H4) and image-derived stem traits (H5) | `ed1804018e5ba7979a8b238a295ea64d828ca0a1fd3632f9bf07dfa1e25e4fee` |
| `02_spec_revision_runs.md` | 2026-10-09, before the runs | clDice with k = 40 at doubled resolution (6 runs); second-seed LOIO baselines (20 runs) | `38afc7e756feeb3e0014fcda7a69bb4fca1f66153e0b6d0280bc3288d5dedd1b` |
| `03_spec_recall_control_topology.md` | 2026-10-10, before the runs and analyses | Recall-weighted Tversky control (20 runs; H6a/H6b), spatial topology at matched predicted area (H7/H8), loss-weight sensitivity (16 runs) | `f160ac1f1b8d57ab1330cb0b34dcfe7b2ba36bd80a68544bf9f92145b693d490` |

The ordering of specification and runs can also be checked against the `started` time stamps in each run's `results/**/config.json`.

**Errata and deviations (recorded here instead of editing the files):**
- `01_spec…`, section E2: "eight institutions never used as test sets" should read "nine" (the nine institutions are listed in the same sentence).
- The offset grid in `01_spec…` (−2.0 … +4.0) was extended to −2.0 … +8.0 for the threshold analyses after a code test on six validation images and before any test result was computed; two region-split models selected the +8.0 bound (manuscript, Section 3.8).
- The season-stratified calibration (Supplementary Table S15) and the matched-area comparison (Supplementary Table S14) were added in review and are not part of these specifications; they are reported as exploratory and post hoc, respectively.
- `03_spec…` refers to itself by its internal name (`06_spec…`) and to internal paths; the analysis is `analysis/review2/rev_block6.py`.
- `03_spec…`, Block B: the Betti-0 error per offset (|n_pred − n_ref|) was computed as an additional descriptive measure; it is not used for the H7/H8 decisions.
- `03_spec…`, Block A: institution-cluster bootstrap intervals for the mean change per loss (manuscript Table 7) were added as descriptive statistics.
- `03_spec…`, Block C was run after the Block A/B results had been written up; the outputs follow the specification (stem precision and recall included in `analysis/review2/lambda_sensitivity.csv`). Block B results at offset 0 are reported in Supplementary Table S21.

The pre-declared analysis of the main experiments (sub-questions SQ1–SQ4, primary outcomes and the consistency criterion) is described in the manuscript, Sections 1 and 3.9.
