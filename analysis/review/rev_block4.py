"""Revision runs (spec: specifications/02_spec_revision_runs.md). R-K40: clDice k=40 at x2; R-S1: second-seed LOIO baselines."""
import glob, json, importlib.util, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
spec = importlib.util.spec_from_file_location("A", "src/gwfss_stem/analysis.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
OUT = "analysis/review"; res = {}
M = A.PRIMARY + A.SECONDARY
def seedavg(pat):
    return pd.concat([A.per_image(d) for d in sorted(glob.glob(pat))]).groupby("name")[M].mean()
# ---- R-K40
rows = []
for b in ["segformer_b1", "deeplabv3plus_r101"]:
    base = seedavg(f"results/random/{b}/base/scale2.0/seed*"); k10 = seedavg(f"results/random/{b}/cldice/scale2.0/seed*")
    k40 = seedavg(f"results/random/{b}/cldice_k40/scale2.0/seed*")
    for cname, a, r in [("k40 vs base", k40, base), ("k40 vs k10", k40, k10)]:
        for m in M:
            pair = pd.concat([r[m], a[m]], axis=1, keys=["b", "a"]).dropna(); d = (pair.a - pair.b).values
            lo, hi = A.bootstrap_ci(d)
            rows.append(dict(backbone=b, comparison=cname, metric=m, n=len(d), mean_diff=d.mean(), ci_low=lo, ci_high=hi, p=wilcoxon(d).pvalue if np.any(d) else 1.0))
K = pd.DataFrame(rows); K["p_holm"] = K.groupby("metric").p.transform(lambda s: A.holm(s.to_numpy()))
better = np.where(K.metric == "stem_betti0_err", K.ci_high < 0, K.ci_low > 0); worse = np.where(K.metric == "stem_betti0_err", K.ci_low > 0, K.ci_high < 0)
K["improves"] = better & (K.p_holm < .05); K["degrades"] = worse & (K.p_holm < .05)
K.to_csv(f"{OUT}/rk40_paired.csv", index=False)
pooled = []
for b in ["segformer_b1", "deeplabv3plus_r101"]:
    for l in ["base", "cldice", "cldice_k40"]:
        js = [json.load(open(f)) for f in sorted(glob.glob(f"results/random/{b}/{l}/scale2.0/seed*/summary_ss.json"))]
        ms = [pd.read_csv(f) for f in sorted(glob.glob(f"results/random/{b}/{l}/scale2.0/seed*/metrics_ss.csv"))]
        pooled.append(dict(backbone=b, loss=l, stem_iou=np.mean([100 * j["iou_stem"] for j in js]), stem_iou_sd=np.std([100 * j["iou_stem"] for j in js], ddof=1),
                           miou=np.mean([100 * j["miou"] for j in js]), cldice=np.mean([100 * j["cldice_stem"] for j in js]),
                           area_ratio=np.mean([m.pred_stem_px.sum() / m.gt_stem_px.sum() for m in ms]),
                           precision=np.mean([100 * m.inter_stem.sum() / m.pred_stem_px.sum() for m in ms]), recall=np.mean([100 * m.inter_stem.sum() / m.gt_stem_px.sum() for m in ms]),
                           components=np.mean([m.stem_cc_pred.mean() for m in ms])))
P = pd.DataFrame(pooled); P.to_csv(f"{OUT}/rk40_pooled.csv", index=False)
# ---- R-S1
U = pd.read_csv("analysis/loio/units.csv")
a1, iou1 = [], []
for _, u in U.iterrows():
    d = pd.read_csv(f"results/loio_{u.institute}/{u.backbone}/base/scale1.0/seed1/metrics_ss.csv")
    a1.append(d.pred_stem_px.sum() / d.gt_stem_px.sum()); iou1.append(100 * d.inter_stem.sum() / d.union_stem.sum())
U["area_ratio_seed1"] = a1; U["stemIoU_base_seed1"] = iou1
for l in ["cldice", "skelrecall"]:  # delta' = seed-0 loss vs seed-1 baseline (per image)
    dl = []
    for _, u in U.iterrows():
        b1 = A.per_image(f"results/loio_{u.institute}/{u.backbone}/base/scale1.0/seed1").set_index("name").stem_iou
        lo = A.per_image(f"results/loio_{u.institute}/{u.backbone}/{l}/scale1.0/seed0").set_index("name").stem_iou
        dl.append(100 * (lo - b1).dropna().mean())
    U[f"delta1_{l}"] = dl
g = np.random.default_rng(0); h1 = []
for l in ["cldice", "skelrecall"]:
    for xa, ya, lab in [("area_ratio_seed1", f"delta_{l}", "A from seed 1, delta seed-0 vs seed-0"), ("area_ratio_seed1", f"delta1_{l}", "A from seed 1, delta seed-0 loss vs seed-1 base")]:
        x, y = U[xa].values, U[ya].values; r = spearmanr(x, y).statistic
        perm = [spearmanr(x, g.permutation(y)).statistic for _ in range(10000)]
        h1.append(dict(loss=l, variant=lab, rho=r, p_one_sided=(np.sum(np.array(perm) <= r) + 1) / 10001))
H = pd.DataFrame(h1); H.to_csv(f"{OUT}/rs1_h1.csv", index=False)
U["dA"] = U.area_ratio_seed1 - U.area_ratio_base; U["dIoU"] = U.stemIoU_base_seed1 - U.stemIoU_base
U.to_csv(f"{OUT}/rs1_units.csv", index=False)
res["seed_spread"] = dict(A_abs_diff_median=U.dA.abs().median(), A_abs_diff_max=U.dA.abs().max(), A_rank_rho=spearmanr(U.area_ratio_base, U.area_ratio_seed1).statistic,
                          iou_abs_diff_median=U.dIoU.abs().median(), iou_abs_diff_max=U.dIoU.abs().max(),
                          under07_seed0=int((U.area_ratio_base < .7).sum()), under07_seed1=int((U.area_ratio_seed1 < .7).sum()))
by = U.groupby("backbone")[["dA", "dIoU"]].agg(lambda s: s.abs().median()); res["seed_spread_by_backbone"] = by.round(3).to_dict()
json.dump(res, open(f"{OUT}/block4_results.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(K[K.metric.isin(["stem_iou", "stem_cldice", "stem_skel_recall", "stem_betti0_err", "leaf_iou"])].assign(mean_diff=lambda d: d.mean_diff * np.where(d.metric == "stem_betti0_err", 1, 100)).round(3).to_string(index=False))
print(P.round(2).to_string(index=False)); print(H.round(4).to_string(index=False)); print(json.dumps(res, indent=1, default=lambda v: round(float(v), 3)))
