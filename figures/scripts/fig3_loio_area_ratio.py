"""Fig. 3: LOIO (10 held-out institutions x 2 architectures = 20 units).
(a) Mean per-image change in stem IoU (loss - baseline) vs the baseline's predicted/reference stem area on the held-out institution.
(b) Median signed relative error of visible stem length (skeleton length) per unit, baseline vs losses."""
import sys, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P = sys.argv[1]
U = pd.read_csv(f"{P}/analysis/loio/units.csv"); H1 = pd.read_csv(f"{P}/analysis/loio/h1_spearman.csv")
S = pd.read_csv(f"{P}/analysis/loio/traits_loio_summary.csv")
TXT, MUTED, GRID, BASEC = "#0b0b0b", "#52514e", "#d9d8d4", "#52514e"
LOSS = [("cldice", "clDice", "#2a78d6"), ("skelrecall", "Skeleton Recall", "#eb6834")]
MK = {"segformer_b2": "o", "upernet_convnext_t": "s"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": TXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 2, figsize=(190 / 25.4, 78 / 25.4), constrained_layout=True)
for a in ax:
    a.axvspan(0, 0.7, color="#f1f0ec", zorder=0, lw=0); a.axhline(0, color=MUTED, lw=0.8, zorder=1)
    a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.set_xlim(0.15, 0.95); a.set_xlabel("Baseline predicted / reference stem area on the held-out institution")
# (a)
for l, name, col in LOSS:
    for b, mk in MK.items():
        d = U[U.backbone == b]
        ax[0].scatter(d.area_ratio_base, d[f"delta_{l}"], marker=mk, s=22, facecolor=col, edgecolor="white", linewidth=0.8, zorder=3)
    r = H1[(H1.loss == l) & (H1.scope == "all 20")].iloc[0]
    ax[0].text(0.93, 9.8 - (0 if l == "cldice" else 1.1), f"{name}: ρ = {r.rho:.2f}, p = {r.p_one_sided:.3f}", ha="right", va="top", fontsize=6.5, color=TXT)
ax[0].set_ylabel("Change in stem IoU vs baseline\n(mean per image, percentage points)")
ax[0].text(0.17, -3.3, "under-segmented\n(ratio < 0.7)", fontsize=6, color=MUTED, va="bottom")
# (b)
for l, name, col in [("base", "Baseline", BASEC)] + LOSS:
    for b, mk in MK.items():
        d = S[(S.condition == l) & (S.backbone == b)]
        ax[1].scatter(d.area_ratio_base, 100 * d.len_bias, marker=mk, s=22, facecolor=col if l != "base" else "white",
                      edgecolor=col if l == "base" else "white", linewidth=0.9, zorder=3 if l != "base" else 4)
ax[1].set_ylabel("Visible stem-class skeleton length:\nmedian signed relative error (%)")
for a, t in zip(ax, "ab"):
    a.annotate(f"({t})", xy=(0, 1), xycoords="axes fraction", xytext=(-30, 6), textcoords="offset points", fontsize=8, fontweight="bold", color=TXT)
h = [Line2D([], [], lw=0, marker="o", ms=5, mfc=c, mec="white", label=n) for _, n, c in LOSS]
h += [Line2D([], [], lw=0, marker="o", ms=5, mfc="white", mec=BASEC, label="Baseline (b)"),
      Line2D([], [], lw=0, marker="o", ms=5, mfc=MUTED, mec=MUTED, label="SegFormer-B2"),
      Line2D([], [], lw=0, marker="s", ms=5, mfc=MUTED, mec=MUTED, label="UPerNet (ConvNeXt-T)")]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.1))
for ext in ("pdf", "png"):
    fig.savefig(f"{P}/figures/fig3_loio_area_ratio.{ext}", dpi=600, bbox_inches="tight")
print("ok")
