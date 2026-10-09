"""LOIO H1-H3 (spec: specifications/01_spec_loio_calibration_traits.md, E1-E2). Run from the project root."""
import glob, importlib.util, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
spec = importlib.util.spec_from_file_location("A", "src/gwfss_stem/analysis.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
INST = ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UQ", "USASK", "UTokyo"]
BB = ["segformer_b2", "upernet_convnext_t"]
M = A.PRIMARY + A.SECONDARY
R = "results/loio_{i}/{b}/{l}/scale1.0/seed0"
units, tests = [], []
for b in BB:
    for i in INST:
        raw = {l: pd.read_csv(R.format(i=i, b=b, l=l) + "/metrics_ss.csv") for l in A_LOSSES} if False else None
        base_raw = pd.read_csv(R.format(i=i, b=b, l="base") + "/metrics_ss.csv")
        P = {l: A.per_image(R.format(i=i, b=b, l=l)).set_index("name") for l in ["base", "cldice", "skelrecall"]}
        u = dict(institute=i, backbone=b, n_test=len(base_raw),
                 area_ratio_base=base_raw.pred_stem_px.sum() / base_raw.gt_stem_px.sum())
        for l in ["base", "cldice", "skelrecall"]:
            d = pd.read_csv(R.format(i=i, b=b, l=l) + "/metrics_ss.csv")
            u[f"stemIoU_{l}"] = 100 * d.inter_stem.sum() / d.union_stem.sum()
            u[f"area_ratio_{l}"] = d.pred_stem_px.sum() / d.gt_stem_px.sum()
        for l in ["cldice", "skelrecall"]:
            for m in M:
                pair = pd.concat([P["base"][m], P[l][m]], axis=1, keys=["b", "a"]).dropna()
                diff = (pair.a - pair.b).to_numpy()
                lo, hi = A.bootstrap_ci(diff)
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                tests.append(dict(institute=i, backbone=b, loss=l, metric=m, n=len(diff), mean_diff=diff.mean(), ci_low=lo, ci_high=hi, p=p))
                if m == "stem_iou": u[f"delta_{l}"] = 100 * diff.mean()
        units.append(u)
U = pd.DataFrame(units); T = pd.DataFrame(tests)
T["p_holm"] = T.groupby("metric").p.transform(lambda s: A.holm(s.to_numpy()))
better = np.where(T.metric == "stem_betti0_err", T.ci_high < 0, T.ci_low > 0); worse = np.where(T.metric == "stem_betti0_err", T.ci_low > 0, T.ci_high < 0)
T["improves"] = better & (T.p_holm < 0.05); T["degrades"] = worse & (T.p_holm < 0.05)
U.to_csv("analysis/loio/units.csv", index=False); T.to_csv("analysis/loio/paired_loio.csv", index=False)

# H1: Spearman rho(delta, area ratio) across 20 units, one-sided permutation test (10,000)
rng = np.random.default_rng(0); h1 = []
for l in ["cldice", "skelrecall"]:
    for scope, sub in [("all 20", U)] + [(b, U[U.backbone == b]) for b in BB]:
        x, y = sub.area_ratio_base.to_numpy(), sub[f"delta_{l}"].to_numpy()
        rho = spearmanr(x, y).statistic
        perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(10000)])
        h1.append(dict(loss=l, scope=scope, n=len(sub), rho=rho, p_one_sided=(np.sum(perm <= rho) + 1) / (len(perm) + 1)))
H1 = pd.DataFrame(h1); H1.to_csv("analysis/loio/h1_spearman.csv", index=False)
# H2 counts
h2 = []
for l in ["cldice", "skelrecall"]:
    lo, hi = U[U.area_ratio_base < 0.7], U[U.area_ratio_base >= 0.85]
    h2.append(dict(loss=l, n_under=len(lo), under_delta_pos=int((lo[f"delta_{l}"] > 0).sum()), n_near=len(hi), near_delta_le0=int((hi[f"delta_{l}"] <= 0).sum()),
                   n_mid=int(((U.area_ratio_base >= 0.7) & (U.area_ratio_base < 0.85)).sum())))
H2 = pd.DataFrame(h2); H2.to_csv("analysis/loio/h2_counts.csv", index=False)
pd.set_option("display.width", 250)
print(U[["institute", "backbone", "area_ratio_base", "stemIoU_base", "stemIoU_cldice", "stemIoU_skelrecall", "delta_cldice", "delta_skelrecall", "area_ratio_cldice", "area_ratio_skelrecall"]].round(2).to_string(index=False))
print(H1.round(4).to_string(index=False)); print(H2.to_string(index=False))
C = T.groupby(["loss", "metric"])[["improves", "degrades"]].sum(); print(C.to_string())
