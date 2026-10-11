"""Fig. 1: study design. Data and splits -> training with the four loss variants -> inference with an optional
stem-logit offset -> evaluation. Run counts per experiment sum to the 212 training runs of Section 3.10."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = sys.argv[1]
TXT, MUTED = "#0b0b0b", "#52514e"
GREY_BG, GREY = "#f2f1ee", "#52514e"
ORANGE_BG, ORANGE = "#fbe6dc", "#c4541f"
BLUE_BG, BLUE = "#dde9f8", "#2a6fc4"
GREEN_BG, GREEN = "#e3f1e4", "#2f7d3b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "mathtext.fontset": "dejavusans"})
W, H = 190, 100  # one axis unit = 1 mm
fig = plt.figure(figsize=(W / 25.4, H / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=lw))


def title(x, y, s, col=TXT):
    ax.text(x, y, s, fontsize=7.6, fontweight="bold", color=col, va="top")


def body(x, y, s, size=6.4, col=TXT):
    ax.text(x, y, s, fontsize=size, color=col, va="top", linespacing=1.4)


def arrow(x0, y0, x1, y1, col=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=col, lw=1.0))


TOP = H - 2
# Data
box(2, 60, 42, TOP - 60, GREY_BG, GREY)
title(4.5, TOP - 2.5, "Data (Section 3.1)")
body(4.5, TOP - 8.5, "GWFSS v1.0, labelled set:\n1,096 RGB images from\n10 institutions\nClasses: background, head,\nstem, leaf\nStem: 3.8% of pixels;\nmodal, semantic labels")
# Splits
box(2, 2, 42, 54, GREY_BG, GREY)
title(4.5, 53.5, "Splits (Section 3.2)")
body(4.5, 47.5, "Random split: 70 / 10 / 20%\nper institution (in-distribution)\n\nRegion split: 7 institutions\nfor training, UTokyo for\nvalidation, UQ for testing\n\nLeave one institution out\n(LOIO): 10 folds, each\ninstitution tested once")
arrow(23, 60, 23, 56.4)

# Training
X0, X1 = 50, 120
box(X0, 2, X1 - X0, TOP - 2, ORANGE_BG, ORANGE, lw=1.2)
title(X0 + 2.5, TOP - 2.5, "Training (Sections 3.3–3.5)", ORANGE)
ax.text((X0 + X1) / 2, TOP - 9, r"$\mathcal{L}=\mathcal{L}_{\mathrm{CE}}+\mathcal{L}_{\mathrm{Dice}}+\lambda\,\mathcal{L}_{\mathrm{conn}}$   Eq. (1)",
        fontsize=7.6, ha="center", va="top")
body(X0 + 2.5, TOP - 16, r"$\mathcal{L}_{\mathrm{conn}}$ acts on the stem probability only:")
rows = [("none", "baseline"), ("clDice", "soft skeletons, Eqs. (2)–(3)"), ("Skeleton Recall", "reference skeleton tube, Eq. (4)"),
        ("Tversky control", "recall-weighted, no skeleton, Eq. (5)")]
for i, (a, b) in enumerate(rows):
    y = TOP - 21.5 - 4.4 * i
    ax.text(X0 + 3, y, "•", fontsize=7, va="top", color=ORANGE)
    ax.text(X0 + 5.5, y, a, fontsize=6.3, fontweight="bold", va="top")
    ax.text(X0 + 28, y, b, fontsize=6.2, va="top", color=MUTED)
body(X0 + 2.5, TOP - 41, "Architectures: DeepLabV3+ (ResNet-101), SegFormer-\nB1 and -B2, UPerNet (ConvNeXt-T); 8,000 iterations;\nfinal iterate; λ = 1 unless stated", 6.2)
yh = TOP - 55
ax.text(X0 + 2.5, yh, "Experiment", fontsize=6.2, fontweight="bold", va="top")
ax.text(X0 + 52, yh, "Runs", fontsize=6.2, fontweight="bold", va="top", ha="right")
ax.text(X0 + 64, yh, "Seeds", fontsize=6.2, fontweight="bold", va="top", ha="right")
ax.plot([X0 + 2.5, X1 - 2.5], [yh - 3.4, yh - 3.4], color=ORANGE, lw=0.5)
runs = [("Random split, 4 arch. × 3 losses", "36", "3"), ("Region split, 4 arch. × 3 losses", "36", "3"),
        ("×2 resolution, 2 arch. (+ clDice k = 40)", "18 + 6", "3"), ("LOIO, 2 arch. × 3 losses", "60", "1"),
        ("LOIO baselines, second seed", "20", "1"), ("LOIO Tversky control, 2 arch.", "20", "1"),
        ("LOIO λ = 0.25, 0.5, SegFormer-B2, 4 folds", "16", "1")]
for i, (a, n, s) in enumerate(runs):
    y = yh - 4.6 - 3.0 * i
    ax.text(X0 + 2.5, y, a, fontsize=5.9, va="top"); ax.text(X0 + 52, y, n, fontsize=5.9, va="top", ha="right")
    ax.text(X0 + 64, y, s, fontsize=5.9, va="top", ha="right")
y = yh - 4.6 - 3.0 * len(runs) - 0.6
ax.plot([X0 + 2.5, X1 - 2.5], [y, y], color=ORANGE, lw=0.5)
ax.text(X0 + 2.5, y - 1.2, "Total", fontsize=5.9, fontweight="bold", va="top")
ax.text(X0 + 52, y - 1.2, "212", fontsize=5.9, fontweight="bold", va="top", ha="right")
arrow(44.4, 30, 49.6, 30)

# Inference and calibration
Y0 = 62
box(126, Y0, 62, TOP - Y0, BLUE_BG, BLUE, lw=1.2)
title(128.5, TOP - 2.5, "Inference and calibration (3.6, 3.8)", BLUE)
body(128.5, TOP - 8.5, "Class decision: arg max over the four logits\n"
     "Threshold calibration: offset b added to the\nstem logit before the arg max, Eq. (10);\n"
     "b chosen on the validation institution or on\nk = 5 or 10 labelled target images\n(20 random draws); no retraining", 6.2)
arrow(120.4, 80, 125.6, 80)
# Evaluation
box(126, 2, 62, 56, GREEN_BG, GREEN, lw=1.2)
title(128.5, 55.5, "Evaluation (Sections 3.7–3.9)", GREEN)
body(128.5, 49.5, "Pixels: IoU per class; stem precision,\nrecall and area ratio A, Eqs. (6)–(7)\n\n"
     "Connectivity: clDice and skeleton recall\nscores, Betti-0 error; merged, split,\nspurious and missed components, Eq. (12)\n\n"
     "Traits: visible stem length, stem area,\nfragments; relative error, Eq. (8)\n\n"
     "Paired per-image tests; bootstrap intervals\nover images or institutions, Eqs. (9), (11)", 6.2)
arrow(157, Y0 - 0.4, 157, 58.4, BLUE)
for ext in ("pdf", "png"):
    fig.savefig(f"{P}/figures/fig1_study_design.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02)
print("ok")
