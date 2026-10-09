"""Fig. 4: qualitative stem predictions for four run units (one image each, chosen by analysis/review/select_fig4.py).
Prediction panels: stem true positives (blue), false positives (orange), false negatives (yellow) over the darkened image."""
import sys, numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
P = sys.argv[1]; D = f"{P}/analysis/review/fig4"
CK = pd.read_csv(f"{D}/fig4_check.csv")
TP, FP, FN, GTC = (42, 120, 214), (235, 104, 52), (245, 211, 63), (42, 120, 214)
COLS = [("rgb", "RGB image"), ("gt", "Reference"), ("base", "Baseline"), ("cldice", "+ clDice"),
        ("skelrecall", "+ Skeleton Recall"), ("base_cal", "Baseline + calibrated\nthreshold")]
ROWS = ["Random split\nSegFormer-B2\n(INRAE)", "Held out: USASK\nSegFormer-B2", "Held out: UTokyo\nSegFormer-B2", "Held out: UQ\nUPerNet"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
fig, ax = plt.subplots(4, 6, figsize=(190 / 25.4, 128 / 25.4), gridspec_kw=dict(wspace=0.04, hspace=0.1))
for i in range(4):
    rgb = np.array(Image.open(f"{D}/row{i}_rgb.png").convert("RGB")).astype(float)
    gt = np.array(Image.open(f"{D}/row{i}_gt.png")); valid = gt != 255; g = (gt == 2) & valid
    dark = rgb * 0.35
    for j, (c, title) in enumerate(COLS):
        a = ax[i, j]; a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_linewidth(0.4); sp.set_color("#52514e")
        if c == "rgb":
            img = rgb
        elif c == "gt":
            img = dark.copy(); img[g] = GTC
        else:
            p = (np.array(Image.open(f"{D}/row{i}_{c}.png")) == 2) & valid
            img = dark.copy(); img[p & g] = TP; img[p & ~g] = FP; img[~p & g] = FN
            r = CK[(CK.row == i) & (CK.cond == c)].iloc[0]
            lab = f"IoU {r.stem_iou:.1f}" + (f" (b = +{r.offset:.1f})" if c == "base_cal" else "")
            a.text(0.03, 0.03, lab, transform=a.transAxes, fontsize=5.8, color="white", va="bottom",
                   bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6))
        a.imshow(img.astype(np.uint8))
        if i == 0: a.set_title(title, fontsize=7, pad=3)
        if j == 0: a.set_ylabel(ROWS[i], fontsize=6.5, labelpad=3)
h = [Patch(fc=np.array(TP) / 255, label="Stem, correct (true positive)"),
     Patch(fc=np.array(FP) / 255, label="Predicted stem outside reference (false positive)"),
     Patch(fc=np.array(FN) / 255, label="Reference stem missed (false negative)")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.02))
for ext in ("pdf", "png"):
    fig.savefig(f"{P}/figures/fig4_qualitative.{ext}", dpi=600 if ext == "png" else 300, bbox_inches="tight")
print("ok")
