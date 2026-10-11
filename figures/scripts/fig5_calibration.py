"""Fig. 5: threshold calibration on the 20 LOIO units (replaces the former Table 5; the full table, including the
region-split rows, is Supplementary Table S23). Per-unit differences in pooled stem IoU (dots) with their mean and the
95% bootstrap interval over institutions (5,000 resamples; analysis/review/table12_cluster_bootstrap.csv)."""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = sys.argv[1]
E = pd.read_csv(f"{P}/analysis/loio/e3_thresholds.csv"); E = E[E.split.str.startswith("loio")]
CB = pd.read_csv(f"{P}/analysis/review/table12_cluster_bootstrap.csv")
TXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
GREY, BLUE, ORANGE = "#52514e", "#2a78d6", "#eb6834"
MK = {"segformer_b2": "o", "upernet_convnext_t": "s"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": TXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
ROWS = [("base+off vs base", "Calibrated baseline\n− baseline", "base+{t}", "base", GREY, True),
        ("clDice vs base+off", "clDice\n− calibrated baseline", "cldice", "base+{t}", BLUE, True),
        ("SR vs base+off", "Skeleton Recall\n− calibrated baseline", "skelrecall", "base+{t}", ORANGE, True),
        ("clDice+off vs base+off", "clDice + offset\n− calibrated baseline", "cldice+{t}", "base+{t}", BLUE, False),
        ("SR+off vs base+off", "Skeleton Recall + offset\n− calibrated baseline", "skelrecall+{t}", "base+{t}", ORANGE, False)]
PANELS = [(0, "Tval", "Offset chosen on the\nvalidation institution"), (5, "T5", "Offset chosen on\n5 labelled target images"),
          (10, "T10", "Offset chosen on\n10 labelled target images")]
fig, axes = plt.subplots(1, 3, figsize=(190 / 25.4, 88 / 25.4), sharey=True, sharex=True, constrained_layout=True)
g = np.random.default_rng(0)
for ax, (k, t, ttl) in zip(axes, PANELS):
    Q = E[E.k == k].pivot_table(index=["institute", "backbone"], columns="condition", values="stem_iou").reset_index()
    for j, (key, lab, a, b, col, filled) in enumerate(ROWS):
        y = len(ROWS) - 1 - j
        d = Q[a.format(t=t)] - Q[b.format(t=t)]
        r = CB[(CB.offset == t) & (CB.contrast == key)].iloc[0]
        assert abs(d.mean() - r["mean"]) < 1e-9 and int((d > 0).sum()) == r.n_higher
        jit = g.uniform(-0.17, 0.17, len(d))
        for bb, mk in MK.items():
            m = (Q.backbone == bb).to_numpy()
            ax.scatter(d[m], y + 0.12 + jit[m], marker=mk, s=9, facecolor=col if filled else "white", edgecolor=col,
                       linewidth=0.6, alpha=0.65, zorder=3)
        ax.plot([r.ci_low, r.ci_high], [y - 0.22] * 2, color=TXT, lw=1.3, zorder=4, solid_capstyle="butt")
        ax.scatter([r["mean"]], [y - 0.22], marker="D", s=20, facecolor=col if filled else "white", edgecolor=TXT, linewidth=0.8, zorder=5)
        ax.text(1.0, y - 0.22, f"{r['mean']:+.1f} ({r.n_higher}/20)".replace("-", "−"), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6, color=TXT)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_title(ttl, fontsize=7.2)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("Difference in stem IoU (points)")
axes[0].set_yticks(range(len(ROWS))); axes[0].set_yticklabels([r[1] for r in ROWS][::-1], fontsize=6.5)
axes[0].tick_params(axis="y", length=0)
axes[0].set_xlim(-6.5, 13.5); axes[0].set_ylim(-0.6, len(ROWS) - 0.45)
for a, t in zip(axes, "abc"):
    a.annotate(f"({t})", xy=(0, 1), xycoords="axes fraction", xytext=(-4, 14), textcoords="offset points", fontsize=8, fontweight="bold", ha="right")
h = [Line2D([], [], lw=0, marker="o", ms=4, mfc=MUTED, mec=MUTED, alpha=0.7, label="unit, SegFormer-B2"),
     Line2D([], [], lw=0, marker="s", ms=4, mfc=MUTED, mec=MUTED, alpha=0.7, label="unit, UPerNet (ConvNeXt-T)"),
     Line2D([], [], color=TXT, lw=1.3, marker="D", ms=4, mfc="white", mec=TXT, label="mean and 95% interval over institutions"),
     Line2D([], [], lw=0, marker="o", ms=4, mfc="white", mec=MUTED, label="open: loss with its own offset")]
fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.0), handlelength=1.6)
for ext in ("pdf", "png"):
    fig.savefig(f"{P}/figures/fig5_calibration.{ext}", dpi=600, bbox_inches="tight")
print("ok")
