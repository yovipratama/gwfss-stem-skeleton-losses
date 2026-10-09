"""E4 image-derived stem traits on LOIO (H5). Run from the project root."""
import importlib.util, numpy as np, pandas as pd
from scipy.stats import wilcoxon
spec = importlib.util.spec_from_file_location("A", "src/gwfss_stem/analysis.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
INST = ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UQ", "USASK", "UTokyo"]
U = pd.read_csv("analysis/loio/units.csv").set_index(["institute", "backbone"])
def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); return 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
def load(i, b, l):
    d = pd.read_csv(f"results/loio_{i}/{b}/{l}/scale1.0/seed0/metrics_ss.csv"); d = d[d.gt_stem_px >= 500].set_index("name")
    d["len_err"] = (d.skel_pred_px - d.skel_gt_px) / d.skel_gt_px; d["area_err"] = (d.pred_stem_px - d.gt_stem_px) / d.gt_stem_px
    return d
S, T = [], []
for b in ["segformer_b2", "upernet_convnext_t"]:
    for i in INST:
        D = {l: load(i, b, l) for l in ["base", "cldice", "skelrecall"]}; ar = U.loc[(i, b), "area_ratio_base"]
        for l, d in D.items():
            S.append(dict(institute=i, backbone=b, area_ratio_base=ar, condition=l, n=len(d), len_bias=d.len_err.median(), len_abs=d.len_err.abs().median(),
                          len_ccc=ccc(d.skel_pred_px, d.skel_gt_px), area_bias=d.area_err.median(), area_abs=d.area_err.abs().median()))
        for l in ["cldice", "skelrecall"]:
            for t in ["len", "area"]:
                pair = pd.concat([D["base"][f"{t}_err"].abs(), D[l][f"{t}_err"].abs()], axis=1, keys=["b", "a"]).dropna(); diff = (pair.a - pair.b).to_numpy()
                lo, hi = A.bootstrap_ci(diff)
                T.append(dict(institute=i, backbone=b, area_ratio_base=ar, loss=l, trait=t, n=len(diff), mean_diff=diff.mean(), ci_low=lo, ci_high=hi, p=wilcoxon(diff).pvalue if np.any(diff) else 1.0))
S, T = pd.DataFrame(S), pd.DataFrame(T)
T["p_holm"] = T.groupby("trait").p.transform(lambda s: A.holm(s.to_numpy()))
T["reduces"] = (T.ci_high < 0) & (T.p_holm < 0.05); T["increases"] = (T.ci_low > 0) & (T.p_holm < 0.05)
S.to_csv("analysis/loio/traits_loio_summary.csv", index=False); T.to_csv("analysis/loio/traits_loio_tests.csv", index=False)
und = S[(S.area_ratio_base < 0.7)]; nr = S[(S.area_ratio_base >= 0.7)]
print("under-segmenting units:", und[und.condition == "base"].shape[0])
print("baseline median length bias <0 in", int((und[und.condition == "base"].len_bias < 0).sum()), "; area bias <0 in", int((und[und.condition == "base"].area_bias < 0).sum()))
print(und.groupby("condition")[["len_bias", "len_abs", "len_ccc", "area_bias", "area_abs"]].median().round(2))
print(nr.groupby("condition")[["len_bias", "len_abs", "len_ccc", "area_bias", "area_abs"]].median().round(2))
T["group"] = np.where(T.area_ratio_base < 0.7, "under(<0.7)", "near(>=0.7)")
print(T.groupby(["group", "loss", "trait"])[["reduces", "increases"]].sum().join(T.groupby(["group", "loss", "trait"]).size().rename("units")).to_string())
