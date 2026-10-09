"""Round-2 revision analyses (offline). REV-17 two-level bootstrap for all Table S4/S6 metrics; REV-18 absolute error and CCC
under calibration (LOIO T-val/T-5/T-10; region T-5/T-10); REV-35 trait bias by season half; REV-03 per-unit calibration gains;
REV-34 acquisition dates from file names."""
import glob, json, re, importlib.util, numpy as np, pandas as pd
from datetime import date
spec = importlib.util.spec_from_file_location("A", "src/gwfss_stem/analysis.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
OUT = "analysis/review"; res = {}
BB4 = ["deeplabv3plus_r101", "segformer_b1", "segformer_b2", "upernet_convnext_t"]; M = A.PRIMARY + A.SECONDARY
INST = ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UQ", "USASK", "UTokyo"]
# ---- REV-17: two-level bootstrap for all metrics
old = pd.concat([pd.read_csv("analysis/main/paired_tests_sq1.csv").assign(split="random"), pd.read_csv("analysis/main/paired_region_s1.0.csv").assign(split="region")])
rows = []
for split in ["random", "region"]:
    for b in BB4:
        B = {s: A.per_image(f"results/{split}/{b}/base/scale1.0/seed{s}").set_index("name") for s in range(3)}
        for l in ["cldice", "skelrecall"]:
            L = {s: A.per_image(f"results/{split}/{b}/{l}/scale1.0/seed{s}").set_index("name") for s in range(3)}
            for m in M:
                names = sorted(B[0].index)
                X = np.stack([[L[a].loc[names, m].values - B[c].loc[names, m].values for c in range(3)] for a in range(3)])
                ok = ~np.isnan(X).any(axis=(0, 1)); X = X[:, :, ok]; n = X.shape[2]; g = np.random.default_rng(0); bt = []
                for _ in range(2000):
                    sa, sb, im = g.integers(0, 3, 3), g.integers(0, 3, 3), g.integers(0, n, n)
                    bt.append(np.mean([X[a, c][im].mean() for a, c in zip(sa, sb)]))
                lo, hi = np.percentile(bt, [2.5, 97.5])
                o = old[(old.split == split) & (old.backbone == b) & (old.loss == l) & (old.metric == m)].iloc[0]
                marked = bool(o.improves) or bool(o.degrades)
                rows.append(dict(split=split, backbone=b, loss=l, metric=m, marked=marked, two_level_low=lo, two_level_high=hi,
                                 two_level_excl0=bool(lo > 0 or hi < 0)))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/two_level_all_metrics.csv", index=False)
chg = T[T.marked & ~T.two_level_excl0]
res["two_level_marked_total"] = int(T.marked.sum()); res["two_level_marked_losing_excl0"] = chg[["split", "backbone", "loss", "metric"]].to_dict("records")
# ---- REV-18 + REV-03: calibration traits (bias, abs err, CCC) and per-unit gains
def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); return 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
def best(sw, names):
    s = sw[sw.name.isin(names)].groupby("offset")[["inter_stem", "union_stem"]].sum()
    iou = (s.inter_stem / s.union_stem.replace(0, np.nan)).fillna(0).round(6); top = iou[iou == iou.max()].index; return min(top, key=abs)
def traits(sw, names, b):
    d = sw[(sw.offset == b) & sw.name.isin(names) & (sw.gt_stem_px >= 500)]
    e = (d.skel_pred_px - d.skel_gt_px) / d.skel_gt_px
    return e.median(), e.abs().median(), ccc(d.skel_pred_px, d.skel_gt_px), 100 * d.inter_stem.sum() / max(d.union_stem.sum(), 1)
rows = []
units = [(f"results/loio_{i}/{b}/base/scale1.0/seed0", "LOIO", i, b, True) for b in ["segformer_b2", "upernet_convnext_t"] for i in INST]
units += [(d, "region", "UQ", d.split("/")[2], False) for d in sorted(glob.glob("results/region/*/base/scale1.0/seed*"))]
for d, split, inst, b, hasval in units:
    sw = pd.read_csv(f"{d}/sweep_test.csv"); names = sorted(sw.name.unique())
    r = dict(split=split, institute=inst, backbone=b, run=d)
    r["none_bias"], r["none_abs"], r["none_ccc"], r["none_iou"] = traits(sw, names, 0.0)
    if hasval:
        sv = pd.read_csv(f"{d}/sweep_val.csv"); r["Tval_bias"], r["Tval_abs"], r["Tval_ccc"], _ = traits(sw, names, best(sv, sv.name.unique()))
    for k in (5, 10):
        acc = []
        for s in range(20):
            cal = list(np.random.default_rng(s).choice(names, k, replace=False)); rest = [n for n in names if n not in cal]
            acc.append(traits(sw, rest, best(sw, cal)) + (traits(sw, rest, 0.0)[3],))
        acc = np.array(acc); r[f"T{k}_bias"], r[f"T{k}_abs"], r[f"T{k}_ccc"], r[f"T{k}_iou"], r[f"T{k}_iou_none"] = np.nanmean(acc, axis=0)
    rows.append(r)
C = pd.DataFrame(rows).groupby(["split", "institute", "backbone"]).mean(numeric_only=True).reset_index(); C.to_csv(f"{OUT}/calibration_traits.csv", index=False)
summ = {}
for sp in ["LOIO", "region"]:
    x = C[C.split == sp]; summ[sp] = {}
    for c in ["none", "Tval", "T5", "T10"]:
        if f"{c}_bias" in x and x[f"{c}_bias"].notna().any():
            summ[sp][c] = dict(bias_median=x[f"{c}_bias"].median(), abs_median=x[f"{c}_abs"].median(), ccc_median=x[f"{c}_ccc"].median())
    summ[sp]["T10_gain_range"] = [(x.T10_iou - x.T10_iou_none).min(), (x.T10_iou - x.T10_iou_none).max()]
    summ[sp]["T5_gain_range"] = [(x.T5_iou - x.T5_iou_none).min(), (x.T5_iou - x.T5_iou_none).max()]
res["calibration_traits"] = summ
# losses (no offset) abs/ccc for comparison, LOIO
S = pd.read_csv("analysis/loio/traits_loio_summary.csv")
res["loss_traits_LOIO_all20"] = S.groupby("condition")[["len_bias", "len_abs", "len_ccc"]].median().round(3).to_dict("index")
E = pd.read_csv("analysis/loio/e3_thresholds.csv"); E = E[E.split.str.startswith("loio")]
Q = E[E.k == 0].pivot_table(index=["institute", "backbone"], columns="condition", values="stem_iou"); g = Q["base+Tval"] - Q.base
res["Tval_gain_range"] = [g.min(), g.max()]
# ---- REV-35: trait bias by season half (offset 0; baseline and losses)
def parse(n):
    i = n.split("_")[0]
    pats = {"ETHZ": r"_(20\d{2})(\d{2})(\d{2})_", "INRAE": r"_(20\d{2})-(\d{2})-(\d{2})_", "UTokyo": r"_(20\d{2})(\d{2})(\d{2})_",
            "NJAU": r"_(20\d{2})-(\d{2})(\d{2})-", "ULiege": r"_(20\d{2})_(\d{1,2})_(\d{1,2})_", "RRES": r"_(\d{2})-(\d{2})-(20\d{2})_"}
    m = re.search(pats.get(i, "$^"), n)
    if not m: return None
    g = [int(x) for x in m.groups()]; y, mo, d = (g[2], g[0], g[1]) if i == "RRES" else g
    return date(y, mo, d)
rows = []
for inst in ["ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UTokyo"]:
    for b in ["segformer_b2", "upernet_convnext_t"]:
        for l in ["base", "cldice", "skelrecall"]:
            d = pd.read_csv(f"results/loio_{inst}/{b}/{l}/scale1.0/seed0/metrics_ss.csv")
            d["dt"] = d.name.map(parse); d = d[d.dt.notna()]
            doy = d.dt.map(lambda x: (x - date(x.year if x.month >= 9 else x.year - 1, 9, 1)).days); med = doy.median(); d["half"] = np.where(doy <= med, "early", "late")  # season day counted from 1 September
            for h, x in d[d.gt_stem_px >= 500].groupby("half"):
                e = (x.skel_pred_px - x.skel_gt_px) / x.skel_gt_px
                rows.append(dict(institute=inst, backbone=b, loss=l, half=h, n=len(x), len_bias=e.median(), area_ratio=x.pred_stem_px.sum() / x.gt_stem_px.sum()))
H = pd.DataFrame(rows); H.to_csv(f"{OUT}/traits_by_season.csv", index=False)
p = H.pivot_table(index=["institute", "backbone", "half"], columns="loss", values="len_bias")
res["season_bias_base"] = dict(early_median=float(p.xs("early", level="half").base.median()), late_median=float(p.xs("late", level="half").base.median()),
                               neg_units=int((p.base < 0).sum()), n=int(len(p)), max_abs_half_diff=float((p.xs("early", level="half").base - p.xs("late", level="half").base).abs().max()))
# ---- REV-34: acquisition dates per institution
man = pd.read_csv("data_cache/gw/manifest.csv"); man["dt"] = man.name.map(parse)
res["dates"] = {i: [str(min(g.dt.dropna())), str(max(g.dt.dropna())), int(g.dt.notna().sum())] for i, g in man.groupby("institute") if g.dt.notna().any()}
res["rres_stage_counts"] = man[man.institute == "RRES"].name.str.extract(r"_(tillering|booting|heading|flowering|anthesis|grainfill\w*|maturity|stem\w*|jointing|milk\w*|dough\w*|senescence)_", flags=re.I)[0].value_counts().to_dict()
json.dump(res, open(f"{OUT}/block5_results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=lambda v: round(float(v), 3)))
