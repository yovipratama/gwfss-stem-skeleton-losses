"""Stage 4 revision analyses, block 1 (offline; released files only). Run from project root.
REV-05/38/39 trait scope + exclusions; REV-18 traits under T-val/T-10; REV-36 skeleton decomposition; REV-37 leaf/head area bias;
REV-21 H1 extras; REV-13 H2 margin; REV-02 cluster bootstrap for Table 12 contrasts."""
import glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = "analysis/review"
INST = ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UQ", "USASK", "UTokyo"]
BB = ["segformer_b2", "upernet_convnext_t"]; L3 = ["base", "cldice", "skelrecall"]
R = "results/loio_{i}/{b}/{l}/scale1.0/seed0"
man = pd.read_csv("data_cache/gw/manifest.csv").set_index("name")
U = pd.read_csv("analysis/loio/units.csv")
rng0 = np.random.default_rng(0)
res = {}

# ---- REV-05/38/39: all-unit trait bias + exclusions
rows = []
for b in BB:
    for i in INST:
        for l in L3:
            d = pd.read_csv(R.format(i=i, b=b, l=l) + "/metrics_ss.csv")
            n_all = len(d); d = d[d.gt_stem_px >= 500]
            le = (d.skel_pred_px - d.skel_gt_px) / d.skel_gt_px; ae = (d.pred_stem_px - d.gt_stem_px) / d.gt_stem_px
            lin = (d.skel_pred_in_gt - d.skel_gt_px) / d.skel_gt_px            # REV-36: predicted skeleton inside reference only
            out_frac = 1 - d.skel_pred_in_gt.sum() / max(d.skel_pred_px.sum(), 1)
            rows.append(dict(institute=i, backbone=b, loss=l, n_test=n_all, n_trait=len(d), n_excluded=n_all - len(d),
                             len_bias=le.median(), area_bias=ae.median(), len_in_ref_bias=lin.median(), skel_outside_ref=out_frac))
T = pd.DataFrame(rows).merge(U[["institute", "backbone", "area_ratio_base"]], on=["institute", "backbone"])
T.to_csv(f"{OUT}/traits_all_units.csv", index=False)
b0 = T[T.loss == "base"]
res["len_bias_all20"] = dict(median=b0.len_bias.median(), min=b0.len_bias.min(), max=b0.len_bias.max(), n_neg=int((b0.len_bias < 0).sum()))
res["area_bias_all20"] = dict(median=b0.area_bias.median(), min=b0.area_bias.min(), max=b0.area_bias.max(), n_neg=int((b0.area_bias < 0).sum()))
res["len_bias_under14"] = b0[b0.area_ratio_base < 0.7].len_bias.median()
res["exclusions_per_institute"] = b0[b0.backbone == "segformer_b2"].set_index("institute")[["n_test", "n_excluded"]].to_dict("index")
res["skel_decomp_median_over_units"] = T.groupby("loss")[["len_bias", "len_in_ref_bias", "skel_outside_ref"]].median().round(3).to_dict("index")
res["skel_decomp_under"] = T[T.area_ratio_base < 0.7].groupby("loss")[["len_bias", "len_in_ref_bias", "skel_outside_ref"]].median().round(3).to_dict("index")

# ---- REV-37: leaf/head area-fraction bias (pred class px = union + inter - gt px)
rows = []
for b in BB:
    for i in INST:
        for l in L3:
            d = pd.read_csv(R.format(i=i, b=b, l=l) + "/metrics_ss.csv").set_index("name")
            r = dict(institute=i, backbone=b, loss=l)
            for c in ["head", "leaf"]:
                gt = man.loc[d.index, f"{c}_px"].values; pred = d[f"union_{c}"].values + d[f"inter_{c}"].values - gt
                r[f"{c}_area_bias_pooled"] = pred.sum() / gt.sum() - 1
            rows.append(r)
LH = pd.DataFrame(rows); LH.to_csv(f"{OUT}/leaf_head_area_bias.csv", index=False)
res["leaf_head_bias_median"] = LH.groupby("loss")[["head_area_bias_pooled", "leaf_area_bias_pooled"]].median().round(3).to_dict("index")

# ---- REV-18: traits under T-val and T-10 for LOIO (from sweeps; same selection rule and draw seeds as E3)
def best_offset(sw, names):
    s = sw[sw.name.isin(names)].groupby("offset")[["inter_stem", "union_stem"]].sum()
    iou = (s.inter_stem / s.union_stem.replace(0, np.nan)).fillna(0).round(6); top = iou[iou == iou.max()].index
    return min(top, key=abs)
def trait_bias(sw, names, b):
    d = sw[(sw.offset == b) & sw.name.isin(names) & (sw.gt_stem_px >= 500)]
    return ((d.skel_pred_px - d.skel_gt_px) / d.skel_gt_px).median(), ((d.pred_stem_px - d.gt_stem_px) / d.gt_stem_px).median()
rows = []
for b in BB:
    for i in INST:
        for l in L3:
            d = R.format(i=i, b=b, l=l); sw = pd.read_csv(f"{d}/sweep_test.csv"); sv = pd.read_csv(f"{d}/sweep_val.csv")
            names = sorted(sw.name.unique())
            bv = best_offset(sv, sv.name.unique()); lv, av = trait_bias(sw, names, bv)
            lk, ak = [], []
            for s in range(20):
                cal = list(np.random.default_rng(s).choice(names, 10, replace=False)); rest = [n for n in names if n not in cal]
                x = trait_bias(sw, rest, best_offset(sw, cal)); lk.append(x[0]); ak.append(x[1])
            rows.append(dict(institute=i, backbone=b, loss=l, len_bias_Tval=lv, area_bias_Tval=av, len_bias_T10=np.nanmean(lk), area_bias_T10=np.nanmean(ak)))
TC = pd.DataFrame(rows).merge(T[["institute", "backbone", "loss", "len_bias", "area_bias", "area_ratio_base"]], on=["institute", "backbone", "loss"])
TC.to_csv(f"{OUT}/traits_calibrated_loio.csv", index=False)
bb = TC[TC.loss == "base"]
for col in ["len_bias", "len_bias_Tval", "len_bias_T10"]:
    res[f"base_{col}"] = dict(median=bb[col].median(), min=bb[col].min(), max=bb[col].max(), abs_median=bb[col].abs().median())
for l in ["cldice", "skelrecall"]:
    x = TC[TC.loss == l]; res[f"{l}_len_bias"] = dict(median=x.len_bias.median(), abs_median=x.len_bias.abs().median(), T10_median=x.len_bias_T10.median(), T10_abs_median=x.len_bias_T10.abs().median())
ts = pd.read_csv("analysis/loio/trait_summary.csv")
res["random_region_trait_bias"] = ts.groupby(["split", "condition"])[["len_bias_median", "area_bias_median"]].agg(["min", "max"]).round(2).to_string()

# ---- REV-21: H1 extras
h1 = {}
Ub = U.sort_values(["backbone", "institute"]).reset_index(drop=True)
for l in ["cldice", "skelrecall"]:
    I = Ub.groupby("institute")[["area_ratio_base", f"delta_{l}"]].mean()
    r10 = spearmanr(I.area_ratio_base, I[f"delta_{l}"]).statistic
    g = np.random.default_rng(1); perm = [spearmanr(I.area_ratio_base, g.permutation(I[f"delta_{l}"].values)).statistic for _ in range(10000)]
    # block permutation over the 20 units: permute institution labels jointly for both backbones
    x = Ub.area_ratio_base.values; y = Ub[f"delta_{l}"].values; inst = Ub.institute.values; bbv = Ub.backbone.values
    r20 = spearmanr(x, y).statistic; c = 0
    for _ in range(10000):
        p = dict(zip(INST, g.permutation(INST)))
        yp = np.array([Ub[(Ub.institute == p[inst[k]]) & (Ub.backbone == bbv[k])][f"delta_{l}"].values[0] for k in range(len(Ub))])
        c += spearmanr(x, yp).statistic <= r20
    # partial Spearman given baseline IoU (rank residuals)
    rk = lambda v: pd.Series(v).rank().values
    def resid(a, z): A_ = np.c_[np.ones(len(z)), z]; return a - A_ @ np.linalg.lstsq(A_, a, rcond=None)[0]
    part = np.corrcoef(resid(rk(x), rk(Ub.stemIoU_base)), resid(rk(y), rk(Ub.stemIoU_base)))[0, 1]
    nt = Ub.institute != "UTokyo"; rnt = spearmanr(x[nt], y[nt]).statistic
    h1[l] = dict(rho_inst10=r10, p_inst10=(np.sum(np.array(perm) <= r10) + 1) / 10001, p_block20=(c + 1) / 10001, partial_rho_given_baseIoU=part, rho_without_UTokyo=rnt)
res["h1_extras"] = h1

# ---- REV-13: H2 with a practical margin and H3
P = pd.read_csv("analysis/loio/paired_loio.csv"); P = P[P.metric == "stem_iou"]
und = U[U.area_ratio_base < 0.7]
res["h2_margin"] = {l: dict(delta_gt0=int((und[f"delta_{l}"] > 0).sum()), delta_ge1=int((und[f"delta_{l}"] >= 1).sum()),
                         meets_H3=int(P[(P.loss == l) & P.improves].merge(und[["institute", "backbone"]]).shape[0])) for l in ["cldice", "skelrecall"]}

# ---- REV-02: cluster bootstrap (institutions) for Table 12 contrasts, LOIO
E = pd.read_csv("analysis/loio/e3_thresholds.csv"); E = E[E.split.str.startswith("loio")]
cb = []
for k, t in [(0, "Tval"), (5, "T5"), (10, "T10")]:
    Q = E[E.k == k].pivot_table(index=["institute", "backbone"], columns="condition", values="stem_iou").reset_index()
    bt = f"base+{t}"
    for name, a, b_ in [("base+off vs base", bt, "base"), ("clDice vs base+off", "cldice", bt), ("SR vs base+off", "skelrecall", bt),
                        ("clDice+off vs base+off", f"cldice+{t}", bt), ("SR+off vs base+off", f"skelrecall+{t}", bt)]:
        diff = (Q[a] - Q[b_]).values; inst = Q.institute.values
        g = np.random.default_rng(0); boots = []
        for _ in range(5000):
            s = g.choice(INST, len(INST)); boots.append(np.mean(np.concatenate([diff[inst == j] for j in s])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cb.append(dict(offset=t, contrast=name, mean=diff.mean(), ci_low=lo, ci_high=hi, n_higher=int((diff > 0).sum()), within_1pt=bool(lo > -1 and hi < 1)))
CB = pd.DataFrame(cb); CB.to_csv(f"{OUT}/table12_cluster_bootstrap.csv", index=False)
json.dump(res, open(f"{OUT}/block1_results.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(json.dumps({k: v for k, v in res.items() if k != "random_region_trait_bias"}, indent=1, default=lambda v: round(float(v), 3)))
print(res["random_region_trait_bias"]); print(CB.round(2).to_string(index=False))
