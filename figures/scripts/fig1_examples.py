"""Fig. 1: GWFSS examples showing modal stem annotation (stems interrupted by occluding heads/leaves)."""
import io, sys, numpy as np, pyarrow.parquet as pq
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = sys.argv[1]
WANT = ["INRAE_phenomobile_45_2022_FFAST_Clermont_2022-06-22_uplot_E1_C01_camera_2_1_RGB_WB.png", "CIMMYT_IMG_2937.png"]
LABEL = {WANT[0]: "INRAE", WANT[1]: "CIMMYT"}
found = {}
for f in [f"{P}/data_cache/shard0.parquet", f"{P}/data_cache/shard1.parquet"]:
    pf = pq.ParquetFile(f)
    for g in range(pf.num_row_groups):
        for row in pf.read_row_group(g).to_pylist():
            if row["mask"]["path"] in WANT:
                found[row["mask"]["path"]] = (np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")),
                                              np.array(Image.open(io.BytesIO(row["mask"]["bytes"])).convert("RGB")))
STEM = np.array([50, 132, 255])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
fig, axes = plt.subplots(2, 3, figsize=(190 / 25.4, 128 / 25.4), constrained_layout=True)
for r, name in enumerate(WANT):
    img, mask = found[name]
    stem = (mask == STEM).all(-1)
    overlay = (img * 0.45).astype(np.uint8); overlay[stem] = [255, 0, 255]
    for c, (panel, title) in enumerate([(img, "RGB image"), (mask, "Annotation"), (overlay, "Stem pixels (magenta)")]):
        ax = axes[r, c]; ax.imshow(panel); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if r == 0: ax.set_title(title, fontsize=8)
        ax.text(0.02, 0.97, f"({'abcdef'[r*3+c]})", transform=ax.transAxes, va="top", ha="left", fontsize=8,
                color="white", fontweight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=1.2, edgecolor="none"))
    axes[r, 0].set_ylabel(LABEL[name], fontsize=8)
# colour key for the annotation panels
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=np.array(c) / 255, label=l) for c, l in
                    [((0, 0, 0), "background"), ((50, 255, 132), "head"), ((50, 132, 255), "stem"), ((214, 255, 50), "leaf")]],
           loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
for ext in ("pdf", "png"):
    fig.savefig(f"{P}/figures/fig1_gwfss_examples.{ext}", dpi=600, bbox_inches="tight")
print("ok", list(found))
