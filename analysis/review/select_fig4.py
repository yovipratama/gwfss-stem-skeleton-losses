"""Fig. 4 image selection (rule fixed before any prediction is viewed).
Per panel row (one run unit, seed 0): test images with >= 500 reference stem pixels; per image,
d = mean of the two losses' stem IoU change vs baseline; choose the image whose d is closest to
the unit median of d (ties -> name order). Calibrated offset for the baseline: ten-image draws
(seeds 0-19, as in thresholds_e3.py), median selected offset; random split: validation offset."""
import json, numpy as np, pandas as pd
R = "results"
ROWS = [("random", "segformer_b2"), ("loio_USASK", "segformer_b2"), ("loio_UTokyo", "segformer_b2"), ("loio_UQ", "upernet_convnext_t")]

def iou(m):
    return m.inter_stem / m.union_stem.replace(0, np.nan)

def best_offset(sw, names):
    s = sw[sw.name.isin(names)].groupby("offset")[["inter_stem", "union_stem"]].sum()
    v = (s.inter_stem / s.union_stem.replace(0, np.nan)).fillna(0).round(6)
    return min(v[v == v.max()].index, key=abs)

out = []
for split, bb in ROWS:
    d = {l: pd.read_csv(f"{R}/{split}/{bb}/{l}/scale1.0/seed0/metrics_ss.csv").set_index("name") for l in ("base", "cldice", "skelrecall")}
    b = d["base"]; keep = b.index[b.gt_stem_px >= 500]
    delta = ((iou(d["cldice"]) + iou(d["skelrecall"])) / 2 - iou(b)).loc[keep]
    med = delta.median()
    name = (delta - med).abs().sort_index().idxmin()
    run = f"{split}/{bb}/base/scale1.0/seed0"
    if split.startswith("loio"):
        sw = pd.read_csv(f"{R}/{run}/sweep_test.csv"); names = sorted(sw.name.unique())
        offs = [best_offset(sw, list(np.random.default_rng(s).choice(names, 10, replace=False))) for s in range(20)]
        off, src = float(sorted(offs)[(len(offs) - 1) // 2]), "lower median of 20 ten-image draws"
    else:
        off, src = json.load(open(f"{R}/{run}/summary_thr.json"))["offset"], "validation set"
    out.append(dict(split=split, backbone=bb, name=name, unit_median_delta=round(100 * med, 2),
                    image_delta=round(100 * delta[name], 2), n_eligible=len(keep), offset=off, offset_source=src,
                    base_iou=round(100 * iou(b)[name], 1), cldice_iou=round(100 * iou(d['cldice'])[name], 1),
                    skelrecall_iou=round(100 * iou(d['skelrecall'])[name], 1)))
df = pd.DataFrame(out); df.to_csv("analysis/review/fig4_selection.csv", index=False); print(df.to_string())
