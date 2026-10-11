"""Fig. 2: GWFSS labelled set by institution. (a) One image and its annotation per institution, chosen by a fixed
rule (the image whose stem fraction is the institution's median among images with >= 500 stem pixels; ties by
name), so that the examples are typical rather than selected. (b) Per-image stem fraction by institution."""
import sys
import numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

P = sys.argv[1]
GW = f"{P}/data_cache/gw"
M = pd.read_csv(f"{GW}/manifest.csv")
M["px"] = M[["bg_px", "head_px", "stem_px", "leaf_px"]].sum(axis=1)
M["stem_frac"] = 100 * M.stem_px / M.px
INST = sorted(M.institute.unique())
assert len(INST) == 10 and len(M) == 1096
COL = np.array([(0, 0, 0), (50, 255, 132), (50, 132, 255), (214, 255, 50)], np.uint8)
TXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": TXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})


def pick(i):
    d = M[(M.institute == i) & (M.stem_px >= 500)].sort_values(["stem_frac", "name"]).reset_index(drop=True)
    med = d.stem_frac.median()
    return d.iloc[(d.stem_frac - med).abs().argsort(kind="stable").iloc[0]]["name"]


def load(name):
    img = Image.open(f"{GW}/images/{name}").convert("RGB")
    m = np.array(Image.open(f"{GW}/masks/{name}"))
    if img.size[0] != 512:
        img = img.resize((512, 512), Image.BILINEAR)
        m = np.array(Image.fromarray(m).resize((512, 512), Image.NEAREST))
    rgb = np.full(m.shape + (3,), 128, np.uint8); ok = m < 4; rgb[ok] = COL[m[ok]]
    return np.array(img), rgb


fig = plt.figure(figsize=(190 / 25.4, 200 / 25.4))
gs = fig.add_gridspec(7, 5, height_ratios=[1, 1, 0.13, 1, 1, 0.42, 1.25], hspace=0.05, wspace=0.04, left=0.07, right=0.995, top=0.975, bottom=0.06)
picked = {}
for k, inst in enumerate(INST):
    name = picked[inst] = pick(inst)
    img, ann = load(name)
    r, c = 3 * (k // 5), k % 5
    for rr, panel in ((r, img), (r + 1, ann)):
        a = fig.add_subplot(gs[rr, c]); a.imshow(panel); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
        if rr == r:
            a.set_title(inst, fontsize=7.5, pad=2.5)
        if c == 0:
            a.set_ylabel("RGB" if rr == r else "Annotation", fontsize=7)
fig.legend(handles=[Patch(color=COL[i] / 255, label=l) for i, l in enumerate(["background", "head", "stem", "leaf"])],
           loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.53, gs[4, :].get_position(fig).y0 - 0.002), handlelength=1.2)
# (b)
ax = fig.add_subplot(gs[6, :])
g = np.random.default_rng(0)
for k, inst in enumerate(INST):
    d = M[M.institute == inst]
    x = k + g.uniform(-0.28, 0.28, len(d))
    ax.scatter(x, d.stem_frac, s=3.5, color="#3284ff", alpha=0.55, lw=0, zorder=3)
    ax.plot([k - 0.34, k + 0.34], [d.stem_frac.median()] * 2, color=TXT, lw=1.2, zorder=4)
    ax.text(k, -1.4, inst, ha="center", va="top", fontsize=7, color=TXT)
    ax.text(k, -3.6, f"n = {len(d)}\n{(d.stem_px > 0).sum()} with stem", ha="center", va="top", fontsize=5.5, color=MUTED)
ax.set_xticks([])
ax.set_xlim(-0.6, 9.6); ax.set_ylim(-0.5, None)
ax.set_ylabel("Stem pixels per image (%)")
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.005, 0.985, "(a)", fontsize=8, fontweight="bold", va="top")
ax.annotate("(b)", xy=(0, 1), xycoords="axes fraction", xytext=(-38, 8), textcoords="offset points", fontsize=8, fontweight="bold", va="bottom")
for ext in ("pdf", "png"):
    fig.savefig(f"{P}/figures/fig2_dataset_overview.{ext}", dpi=400, bbox_inches="tight")
pd.Series(picked).to_csv(f"{P}/figures/fig2_dataset_overview_images.csv", header=["image"])
print(picked)
