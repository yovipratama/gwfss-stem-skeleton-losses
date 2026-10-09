"""Stage 4 revision analysis, block 3 (offline; exploratory, not pre-specified): REV-35 calibration across acquisition dates.
For institutions whose file names carry a date, images are split at the median date into an early and a late half.
For each half used as evaluation set: k=10 calibration images drawn either from the same half (same-season) or from the other
half (cross-season); evaluation on the same remaining images of the evaluation half; 20 draws. LOIO seed-0 baselines."""
import re, json, numpy as np, pandas as pd
from datetime import date
OUT = "analysis/review"
def parse(n):
    i = n.split("_")[0]
    pats = {"ETHZ": r"_(20\d{2})(\d{2})(\d{2})_", "INRAE": r"_(20\d{2})-(\d{2})-(\d{2})_", "UTokyo": r"_(20\d{2})(\d{2})(\d{2})_",
            "NJAU": r"_(20\d{2})-(\d{2})(\d{2})-", "ULiege": r"_(20\d{2})_(\d{1,2})_(\d{1,2})_", "RRES": r"_(\d{2})-(\d{2})-(20\d{2})_"}
    if i not in pats: return None
    m = re.search(pats[i], n)
    if not m: return None
    g = [int(x) for x in m.groups()]
    y, mo, d = (g[2], g[0], g[1]) if i == "RRES" else g
    return date(y, mo, d)
def season_day(d):
    start = date(d.year if d.month >= 9 else d.year - 1, 9, 1)
    return (d - start).days
def best_offset(sw, names):
    s = sw[sw.name.isin(names)].groupby("offset")[["inter_stem", "union_stem"]].sum()
    iou = (s.inter_stem / s.union_stem.replace(0, np.nan)).fillna(0).round(6); top = iou[iou == iou.max()].index
    return min(top, key=abs)
def iou(sw, names, b):
    d = sw[(sw.offset == b) & sw.name.isin(names)]; return 100 * d.inter_stem.sum() / max(d.union_stem.sum(), 1)
rows = []
for inst in ["ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UTokyo"]:
    for bb in ["segformer_b2", "upernet_convnext_t"]:
        sw = pd.read_csv(f"results/loio_{inst}/{bb}/base/scale1.0/seed0/sweep_test.csv")
        names = sorted(sw.name.unique()); dts = {n: parse(n) for n in names}; names = [n for n in names if dts[n]]
        # season position: day of year within each year's campaign, split at the median (early vs late in season)
        doy = {n: season_day(dts[n]) for n in names}; med = np.median(list(doy.values()))  # days since 1 September of the season
        half = {"early": [n for n in names if doy[n] <= med], "late": [n for n in names if doy[n] > med]}
        for ev, other in [("early", "late"), ("late", "early")]:
            E, O = half[ev], half[other]
            if len(E) < 20 or len(O) < 10: continue
            z = sw[(sw.offset == 0) & sw.name.isin(E)]; area = z.pred_stem_px.sum() / max(z.gt_stem_px.sum(), 1)
            same, cross, b_same, b_cross, none = [], [], [], [], []
            for s in range(20):
                g = np.random.default_rng(s); cal_s = list(g.choice(E, 10, replace=False)); rest = [n for n in E if n not in cal_s]
                cal_o = list(g.choice(O, 10, replace=False))
                bs, bo = best_offset(sw, cal_s), best_offset(sw, cal_o)
                same.append(iou(sw, rest, bs)); cross.append(iou(sw, rest, bo)); none.append(iou(sw, rest, 0.0)); b_same.append(bs); b_cross.append(bo)
            rows.append(dict(institute=inst, backbone=bb, eval_half=ev, n_eval=len(E), area_ratio_half=area, iou_no_offset=np.mean(none),
                             iou_same_season=np.mean(same), iou_cross_season=np.mean(cross), offset_same=np.mean(b_same), offset_cross=np.mean(b_cross)))
D = pd.DataFrame(rows); D["same_minus_cross"] = D.iou_same_season - D.iou_cross_season; D["cross_minus_none"] = D.iou_cross_season - D.iou_no_offset
D.to_csv(f"{OUT}/calibration_by_season.csv", index=False)
pd.set_option("display.width", 250); print(D.round(2).to_string(index=False))
print("same-season better than cross-season in", int((D.same_minus_cross > 0).sum()), "of", len(D), "; mean diff", round(D.same_minus_cross.mean(), 2),
      "; cross-season worse than no offset in", int((D.cross_minus_none < 0).sum()))
