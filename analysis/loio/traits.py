"""Image-derived stem traits (spec: specifications/01_spec_loio_calibration_traits.md, E4). Run from the project root."""
import glob, importlib.util, numpy as np, pandas as pd
from scipy.stats import wilcoxon
spec = importlib.util.spec_from_file_location("A", "src/gwfss_stem/analysis.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
MIN_REF = 500
man = pd.read_csv("data_cache/gw/manifest.csv").set_index("name")
labelled = (man.height * man.width - man.unknown_px)

def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)

def load(split, bb, cond):
    loss, tag = (cond[:-4], "thr") if cond.endswith("+thr") else (cond, "ss")
    fr = []
    for f in sorted(glob.glob(f"results/{split}/{bb}/{loss}/scale1.0/seed*/metrics_{tag}.csv")):
        d = pd.read_csv(f); d = d[d.gt_stem_px >= MIN_REF].copy()
        d["len_err"] = (d.skel_pred_px - d.skel_gt_px) / d.skel_gt_px
        d["area_err"] = (d.pred_stem_px - d.gt_stem_px) / d.gt_stem_px
        d["frag_err"] = (d.stem_cc_pred - d.stem_cc_gt) / d.stem_cc_gt
        d["frac_pred"] = d.pred_stem_px / labelled.loc[d.name].values; d["frac_gt"] = d.gt_stem_px / labelled.loc[d.name].values
        fr.append(d)
    return pd.concat(fr).groupby("name").mean(numeric_only=True)  # average over seeds per image

CONDS = ["base", "base+thr", "cldice", "skelrecall"]
TRAITS = {"len": ("skel_pred_px", "skel_gt_px"), "area": ("frac_pred", "frac_gt"), "frag": ("stem_cc_pred", "stem_cc_gt")}
summ, tests = [], []
for split in ["random", "region"]:
    for bb in sorted({p.split("/")[2] for p in glob.glob(f"results/{split}/*/base")}):
        D = {c: load(split, bb, c) for c in CONDS}
        for c, d in D.items():
            r = dict(split=split, backbone=bb, condition=c, n=len(d))
            for t, (pc, gc) in TRAITS.items():
                r[f"{t}_bias_median"] = d[f"{t}_err"].median(); r[f"{t}_abs_median"] = d[f"{t}_err"].abs().median(); r[f"{t}_ccc"] = ccc(d[pc], d[gc])
            summ.append(r)
        for c in ["cldice", "skelrecall", "base+thr"]:
            for t in TRAITS:
                pair = pd.concat([D["base"][f"{t}_err"].abs(), D[c][f"{t}_err"].abs()], axis=1, keys=["b", "a"]).dropna()
                diff = (pair.a - pair.b).to_numpy()
                lo, hi = A.bootstrap_ci(diff)
                tests.append(dict(split=split, backbone=bb, comparison=f"{c} vs base", trait=t, n=len(diff), mean_diff_abs_err=diff.mean(),
                                  ci_low=lo, ci_high=hi, p=wilcoxon(diff).pvalue if np.any(diff) else 1.0))
S = pd.DataFrame(summ); T = pd.DataFrame(tests)
T["p_holm"] = T.groupby(["split", "trait"]).p.transform(lambda s: A.holm(s.to_numpy()))
T["reduces_error"] = (T.ci_high < 0) & (T.p_holm < 0.05); T["increases_error"] = (T.ci_low > 0) & (T.p_holm < 0.05)
S.to_csv("analysis/loio/trait_summary.csv", index=False); T.to_csv("analysis/loio/trait_tests.csv", index=False)
pd.set_option("display.width", 250)
print(S[["split", "backbone", "condition", "n", "len_bias_median", "len_abs_median", "len_ccc", "area_bias_median", "area_abs_median", "area_ccc", "frag_bias_median"]].round(2).to_string(index=False))
print(T.groupby(["split", "comparison", "trait"])[["reduces_error", "increases_error"]].sum().to_string())
