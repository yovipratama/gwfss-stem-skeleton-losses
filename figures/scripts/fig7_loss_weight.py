"""Fig. 7: loss weight (SegFormer-B2, seed 0, four LOIO folds chosen by the baseline area ratio). Predicted-to-reference
stem area ratio, and change in stem and leaf IoU relative to the baseline (mean per image, 95% bootstrap interval over
images), against λ. Numbers: Supplementary Table S22 (analysis/review2/lambda_sensitivity.csv)."""
import sys
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = sys.argv[1]
L = pd.read_csv(f"{P}/analysis/review2/lambda_sensitivity.csv")
TXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
LOSS = [("cldice", "clDice", "#2a78d6", "o", -0.012), ("skelrecall", "Skeleton Recall", "#eb6834", "s", 0.012)]
FOLDS = ["UTokyo", "USASK", "INRAE", "NJAU"]
LAMS = [0.25, 0.5, 1.0]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": TXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
fig, axes = plt.subplots(3, 4, figsize=(190 / 25.4, 130 / 25.4), sharex=True, sharey="row", constrained_layout=True)
for c, f in enumerate(FOLDS):
    base = L[L.institute == f].iloc[0].area_ratio_base
    axes[0, c].set_title(f"{f} (baseline A = {base:.2f})", fontsize=7)
    axes[0, c].axhline(1, color=MUTED, lw=0.8); axes[0, c].axhline(base, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    for r in (1, 2):
        axes[r, c].axhline(0, color=MUTED, lw=0.8)
    for l, name, col, mk, dx in LOSS:
        d = L[(L.institute == f) & (L.loss == l)].sort_values("lam")
        x = [{0.25: 0, 0.5: 1, 1.0: 2}[v] + 8 * dx for v in d.lam]
        axes[0, c].plot(x, d.area_ratio, color=col, marker=mk, ms=3.5, lw=1.1)
        for r, m in ((1, "d_stem_iou"), (2, "d_leaf_iou")):
            axes[r, c].errorbar(x, d[m], yerr=[d[m] - d[f"{m}_lo"], d[f"{m}_hi"] - d[m]], color=col, marker=mk, ms=3.5,
                                lw=1.1, capsize=1.5, elinewidth=0.8)
for a in axes.flat:
    a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.set_xticks(range(3)); a.set_xticklabels(["0.25", "0.5", "1"]); a.set_xlim(-0.4, 2.4)
for a in axes[2]:
    a.set_xlabel("Loss weight λ")
axes[0, 0].set_ylabel("Predicted / reference\nstem area A")
axes[1, 0].set_ylabel("Change in stem IoU\nvs baseline (points)")
axes[2, 0].set_ylabel("Change in leaf IoU\nvs baseline (points)")
for r, t in enumerate("abc"):
    axes[r, 0].annotate(f"({t})", xy=(0, 1), xycoords="axes fraction", xytext=(-38, 4), textcoords="offset points", fontsize=8, fontweight="bold")
h = [Line2D([], [], color=c, marker=m, ms=3.5, lw=1.1, label=n) for _, n, c, m, _ in LOSS]
h += [Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), label="baseline area ratio"), Line2D([], [], color=MUTED, lw=0.8, label="reference area (A = 1) / no change")]
fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0), handlelength=1.8)
for ext in ("pdf", "png"):
    fig.savefig(f"{P}/figures/fig7_loss_weight.{ext}", dpi=600, bbox_inches="tight")
print("ok")
