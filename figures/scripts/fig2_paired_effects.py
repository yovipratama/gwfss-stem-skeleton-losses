"""Fig. 2: paired per-image differences (loss - baseline) with 95% bootstrap CIs.
Row 1: random split (SQ1/SQ4). Row 2: held-out institution, region split (SQ2).
Filled markers: difference meets the pre-declared criterion (CI excludes 0 and Holm p < 0.05)."""
import sys, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P = sys.argv[1]
ROWS = [("Random split", f"{P}/analysis/main/paired_tests_sq1.csv"),
        ("Held-out institution (UQ)", f"{P}/analysis/main/paired_region_s1.0.csv")]
BB = [("deeplabv3plus_r101", "DeepLabV3+"), ("segformer_b1", "SegFormer-B1"), ("segformer_b2", "SegFormer-B2"), ("upernet_convnext_t", "UPerNet")]
MET = [("stem_iou", "Stem IoU\n(pp)", 100, "right"), ("stem_cldice", "Stem clDice\n(pp)", 100, "right"),
       ("stem_skel_recall", "Stem skeleton\nrecall (pp)", 100, "right"), ("stem_betti0_err", "Betti-0 error\n(components)", 1, "left"),
       ("head_iou", "Head IoU\n(pp)", 100, "right"), ("leaf_iou", "Leaf IoU\n(pp)", 100, "right"), ("background_iou", "Background\nIoU (pp)", 100, "right")]
LOSS = [("cldice", "clDice", "#2a78d6", "o", 0.14), ("skelrecall", "Skeleton Recall", "#eb6834", "s", -0.14)]  # validated pair
TXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": TXT,
                     "xtick.color": MUTED, "ytick.color": TXT, "axes.linewidth": 0.6})
fig, axes = plt.subplots(2, len(MET), figsize=(190 / 25.4, 120 / 25.4), sharey=True, constrained_layout=True)
y = np.arange(len(BB))[::-1]
tabs = [pd.read_csv(f) for _, f in ROWS]
for c, (m, label, k, better) in enumerate(MET):          # shared x-range per metric across rows
    lo = min(t[t.metric == m].ci_low.min() for t in tabs) * k; hi = max(t[t.metric == m].ci_high.max() for t in tabs) * k
    pad = 0.08 * (hi - lo); lo, hi = min(lo - pad, -pad), max(hi + pad, pad)
    for r, ((rowname, _), t) in enumerate(zip(ROWS, tabs)):
        ax = axes[r, c]; ax.set_xlim(lo, hi)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        for loss, lname, col, mk, off in LOSS:
            for yi, (bb, _) in zip(y, BB):
                row = t[(t.loss == loss) & (t.metric == m) & (t.backbone == bb)].iloc[0]
                sig = bool(row.improves) or bool(row.degrades)
                ax.plot([row.ci_low * k, row.ci_high * k], [yi + off] * 2, color=col, lw=1.2, solid_capstyle="round", zorder=2)
                ax.plot(row.mean_diff * k, yi + off, marker=mk, ms=4.2, mfc=col if sig else "white", mec=col, mew=1.0, zorder=3)
        if r == 0:
            ax.set_title(label, fontsize=7, color=TXT, pad=10)
            ax.text(0.5, 1.01, ("better →" if better == "right" else "← better"), transform=ax.transAxes, ha="center", va="bottom", fontsize=6, color=MUTED)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for r, (rowname, _) in enumerate(ROWS):
        axes[r, 0].set_yticks(y, [b for _, b in BB])
for r, (rowname, _) in enumerate(ROWS):
    axes[r, 0].annotate(f"({'ab'[r]}) {rowname}", xy=(0, 1.0), xycoords="axes fraction", xytext=(-62, 4), textcoords="offset points",
                        fontsize=7.5, fontweight="bold", color=TXT, ha="left", va="bottom", annotation_clip=False)
h = [Line2D([], [], color=c, marker=mk, ms=4.2, lw=1.2, mfc=c, label=n) for _, n, c, mk, _ in LOSS]
h += [Line2D([], [], color=MUTED, marker="o", ms=4.2, lw=0, mfc=MUTED, label="meets criterion"),
      Line2D([], [], color=MUTED, marker="o", ms=4.2, lw=0, mfc="white", label="does not meet criterion")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))
for ext in ("pdf", "png"):
    fig.savefig(f"{P}/figures/fig2_paired_effects.{ext}", dpi=600, bbox_inches="tight")
print("ok")
