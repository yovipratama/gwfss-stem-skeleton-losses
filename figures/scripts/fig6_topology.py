"""Fig. 6: spatial structure of the stem predictions on the 20 LOIO units at matched predicted area (replaces the
former Table 8; the numbers are Supplementary Table S24). Per-unit differences per image (dots) between the first model
without an offset and the second model at the offset that matches its predicted stem area, with the mean and the 95%
bootstrap interval over institutions (10,000 resamples, as in analysis/review2/rev_block6_tables.py)."""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = sys.argv[1]
TP = pd.read_csv(f"{P}/analysis/review2/topology_matched.csv")
INST = ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UQ", "USASK", "UTokyo"]


def cluster_boot(diff, inst, n=10000, seed=0):  # as in rev_block6.py
    g = np.random.default_rng(seed); boots = []
    for _ in range(n):
        s = g.choice(INST, len(INST)); boots.append(np.mean(np.concatenate([diff[inst == j] for j in s])))
    return np.percentile(boots, [2.5, 97.5])


TXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3a8f4f"
MK = {"segformer_b2": "o", "upernet_convnext_t": "s"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": TXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
ROWS = [("cldice", "base", "clDice − baseline", BLUE, True), ("skelrecall", "base", "Skeleton Recall − baseline", ORANGE, True),
        ("tversky", "base", "Tversky − baseline", GREEN, True), ("cldice", "tversky", "clDice − Tversky", BLUE, False),
        ("skelrecall", "tversky", "Skeleton Recall − Tversky", ORANGE, False)]
METRICS = [("n_pred", "Predicted\ncomponents"), ("excess_merge", "Excess\nmerges"), ("excess_split", "Excess\nsplits"),
           ("spurious", "Spurious\ncomponents"), ("missed", "Missed reference\ncomponents")]
fig, axes = plt.subplots(1, 5, figsize=(190 / 25.4, 78 / 25.4), sharey=True, constrained_layout=True)
g = np.random.default_rng(1)
out = []
for ax, (m, ttl) in zip(axes, METRICS):
    for j, (tr, cp, lab, col, filled) in enumerate(ROWS):
        y = len(ROWS) - 1 - j
        x = TP[(TP.treat == tr) & (TP.comp == cp)]
        d = (x[f"{m}_treat"] - x[f"{m}_comp_matched"]).to_numpy(); lo, hi = cluster_boot(d, x.institute.to_numpy())
        out.append(dict(comparison=lab, metric=m, mean=d.mean(), lo=lo, hi=hi))
        jit = g.uniform(-0.15, 0.15, len(d))
        for bb, mk in MK.items():
            k = (x.backbone == bb).to_numpy()
            ax.scatter(d[k], y + 0.13 + jit[k], marker=mk, s=7, facecolor=col if filled else "white", edgecolor=col, lw=0.5, alpha=0.6, zorder=3)
        ax.plot([lo, hi], [y - 0.22] * 2, color=TXT, lw=1.3, zorder=4, solid_capstyle="butt")
        ax.scatter([d.mean()], [y - 0.22], marker="D", s=16, facecolor=col if filled else "white", edgecolor=TXT, lw=0.8, zorder=5)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_title(ttl, fontsize=7)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(ROWS))); axes[0].set_yticklabels([r[2] for r in ROWS][::-1], fontsize=6.5); axes[0].tick_params(axis="y", length=0)
axes[0].set_ylim(-0.6, len(ROWS) - 0.5)
fig.supxlabel("Difference per image at matched predicted stem area (first model without offset − second model)", fontsize=7)
h = [Line2D([], [], lw=0, marker="o", ms=3.5, mfc=MUTED, mec=MUTED, alpha=0.7, label="unit, SegFormer-B2"),
     Line2D([], [], lw=0, marker="s", ms=3.5, mfc=MUTED, mec=MUTED, alpha=0.7, label="unit, UPerNet (ConvNeXt-T)"),
     Line2D([], [], color=TXT, lw=1.3, marker="D", ms=3.5, mfc="white", mec=TXT, label="mean and 95% interval over institutions"),
     Line2D([], [], lw=0, marker="o", ms=3.5, mfc="white", mec=MUTED, label="open: comparison with the Tversky control")]
fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=6.3, bbox_to_anchor=(0.55, -0.0), handlelength=1.5)
for ext in ("pdf", "png"):
    fig.savefig(f"{P}/figures/fig6_topology.{ext}", dpi=600, bbox_inches="tight")
print(pd.DataFrame(out).round(2).to_string(index=False))
