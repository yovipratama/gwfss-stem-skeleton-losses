"""Review blocks A-C (spec: specifications/03_spec_recall_control_topology.md). Run from the repository root.
A (H6a/H6b): recall-weighted Tversky control; B (H7/H8): spatial topology at matched predicted area; C: loss weight (if runs exist)."""
import os, json, importlib.util, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
spec = importlib.util.spec_from_file_location("A", "src/gwfss_stem/analysis.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
INST = ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UQ", "USASK", "UTokyo"]
BB = ["segformer_b2", "upernet_convnext_t"]
LOSSES = ["base", "cldice", "skelrecall", "tversky"]
R = os.environ.get("MIRROR", "results") + "/loio_{i}/{b}/{l}/scale1.0/seed0"
OUT = os.environ.get("OUT", "analysis/review2"); os.makedirs(OUT, exist_ok=True)
res = {}
pd.set_option("display.width", 250)


def cluster_boot(diff, inst, n=10000, seed=0):
    g = np.random.default_rng(seed); boots = []
    for _ in range(n):
        s = g.choice(INST, len(INST)); boots.append(np.mean(np.concatenate([diff[inst == j] for j in s])))
    return np.percentile(boots, [2.5, 97.5])


def classify(lo, hi):
    if lo > -1 and hi < 1 and not (lo > 0 or hi < 0):
        return "equivalent (within ±1)"
    return "better" if lo > 0 else "worse" if hi < 0 else "not distinguishable"


def perm_rho(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed); rho = spearmanr(x, y).statistic
    perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
    return rho, (np.sum(perm <= rho) + 1) / (n + 1)


# ---------------- Block A: units, H6a, H6b, side effects ----------------
units, tests = [], []
for b in BB:
    for i in INST:
        raw = {l: pd.read_csv(R.format(i=i, b=b, l=l) + "/metrics_ss.csv") for l in LOSSES}
        P = {l: A.per_image(R.format(i=i, b=b, l=l)).set_index("name") for l in LOSSES}
        u = dict(institute=i, backbone=b, area_ratio_base=raw["base"].pred_stem_px.sum() / raw["base"].gt_stem_px.sum())
        for l, d in raw.items():
            u[f"stemIoU_{l}"] = 100 * d.inter_stem.sum() / d.union_stem.sum()
            u[f"area_ratio_{l}"] = d.pred_stem_px.sum() / d.gt_stem_px.sum()
            u[f"precision_{l}"] = 100 * d.inter_stem.sum() / d.pred_stem_px.sum()
            u[f"recall_{l}"] = 100 * d.inter_stem.sum() / d.gt_stem_px.sum()
        for l in ["cldice", "skelrecall", "tversky"]:
            for m in A.PRIMARY + A.SECONDARY:
                pair = pd.concat([P["base"][m], P[l][m]], axis=1, keys=["b", "a"]).dropna(); diff = (pair.a - pair.b).to_numpy()
                lo, hi = A.bootstrap_ci(diff)
                tests.append(dict(institute=i, backbone=b, loss=l, metric=m, n=len(diff), mean_diff=diff.mean(), ci_low=lo, ci_high=hi,
                                  p=wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0))
                if m == "stem_iou": u[f"delta_{l}"] = 100 * diff.mean()
        units.append(u)
U = pd.DataFrame(units); T = pd.DataFrame(tests)
T["p_holm"] = T.groupby(["loss", "metric"]).p.transform(lambda s: A.holm(s.to_numpy()))
better = np.where(T.metric == "stem_betti0_err", T.ci_high < 0, T.ci_low > 0); worse = np.where(T.metric == "stem_betti0_err", T.ci_low > 0, T.ci_high < 0)
T["improves"] = better & (T.p_holm < 0.05); T["degrades"] = worse & (T.p_holm < 0.05)
U.to_csv(f"{OUT}/units_with_tversky.csv", index=False); T.to_csv(f"{OUT}/paired_loio_with_tversky.csv", index=False)
# consistency: the baseline/clDice/SR deltas must reproduce the published units.csv
U0 = pd.read_csv("analysis/loio/units.csv").set_index(["institute", "backbone"])
chk = max(abs(U.set_index(["institute", "backbone"])[f"delta_{l}"] - U0[f"delta_{l}"]).max() for l in ["cldice", "skelrecall"])
res["check_delta_vs_published_max_abs"] = float(chk)

x, inst = U.area_ratio_base.to_numpy(), U.institute.to_numpy()
rho, p = perm_rho(x, U.delta_tversky.to_numpy())
M10 = U.groupby("institute")[["area_ratio_base", "delta_tversky"]].mean()
rho10, p10 = perm_rho(M10.area_ratio_base.to_numpy(), M10.delta_tversky.to_numpy())
res["H6a"] = dict(rho_20=rho, p_one_sided_20=p, rho_inst10=rho10, p_one_sided_inst10=p10,
                  by_backbone={b: perm_rho(U[U.backbone == b].area_ratio_base.to_numpy(), U[U.backbone == b].delta_tversky.to_numpy()) for b in BB})
h6b = {}
for l in ["cldice", "skelrecall"]:
    d = (U[f"delta_{l}"] - U.delta_tversky).to_numpy(); lo, hi = cluster_boot(d, inst)
    und = U.area_ratio_base < 0.7; lo_u, hi_u = cluster_boot(d[und], inst[und])
    h6b[l] = dict(mean_D=d.mean(), ci_low=lo, ci_high=hi, decision=classify(lo, hi), units_skel_higher=int((d > 0).sum()),
                  under_mean_D=d[und].mean(), under_ci=(lo_u, hi_u))
res["H6b"] = h6b
res["tversky_descriptive"] = dict(
    delta_mean=U.delta_tversky.mean(), delta_under_mean=U[U.area_ratio_base < 0.7].delta_tversky.mean(),
    area_ratio_median={l: U[f"area_ratio_{l}"].median() for l in LOSSES},
    precision_median={l: U[f"precision_{l}"].median() for l in LOSSES}, recall_median={l: U[f"recall_{l}"].median() for l in LOSSES},
    counts=T.groupby(["loss", "metric"])[["improves", "degrades"]].sum().loc["tversky"].to_dict())

# T10 calibration with the same 20 draws as thresholds_e3.py
def best_offset(sw, names):
    s = sw[sw.name.isin(names)].groupby("offset")[["inter_stem", "union_stem"]].sum()
    iou = (s.inter_stem / s.union_stem.replace(0, np.nan)).fillna(0).round(6); top = iou[iou == iou.max()].index
    return min(top, key=abs)
def score(sw, names, b):
    s = sw[(sw.offset == b) & sw.name.isin(names)][["inter_stem", "union_stem"]].sum(); return 100 * s.inter_stem / s.union_stem
cal = []
for b in BB:
    for i in INST:
        SW = {l: pd.read_csv(R.format(i=i, b=b, l=l) + "/sweep_test.csv") for l in ["base", "tversky", "cldice", "skelrecall"]}
        names = sorted(SW["base"].name.unique()); acc = {}
        for s in range(20):
            c = list(np.random.default_rng(s).choice(names, 10, replace=False)); rest = [n for n in names if n not in c]
            for l, sw in SW.items():
                acc.setdefault(l, []).append(score(sw, rest, 0.0)); acc.setdefault(l + "+T10", []).append(score(sw, rest, best_offset(sw, c)))
        cal.append(dict(institute=i, backbone=b, **{k: np.mean(v) for k, v in acc.items()}))
C = pd.DataFrame(cal); C.to_csv(f"{OUT}/t10_with_tversky.csv", index=False)
E = pd.read_csv("analysis/loio/e3_thresholds.csv"); E = E[(E.k == 10) & E.split.str.startswith("loio")]
chk2 = max(abs(C.set_index(["institute", "backbone"])[c] - E[E.condition == c].set_index(["institute", "backbone"]).stem_iou).max() for c in ["base", "base+T10", "cldice", "skelrecall"])
res["check_T10_vs_published_max_abs"] = float(chk2)
ci = C.institute.to_numpy(); t10 = {}
for name, a, b_ in [("Tversky vs base+T10", "tversky", "base+T10"), ("Tversky+T10 vs base+T10", "tversky+T10", "base+T10"),
                    ("clDice vs Tversky+T10", "cldice", "tversky+T10"), ("SR vs Tversky+T10", "skelrecall", "tversky+T10")]:
    d = (C[a] - C[b_]).to_numpy(); lo, hi = cluster_boot(d, ci); t10[name] = dict(mean=d.mean(), ci_low=lo, ci_high=hi, n_higher=int((d > 0).sum()))
res["T10"] = t10

# traits (Table 6 procedure)
def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); return 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
tr = []
for b in BB:
    for i in INST:
        for l in LOSSES:
            d = pd.read_csv(R.format(i=i, b=b, l=l) + "/metrics_ss.csv"); d = d[d.gt_stem_px >= 500]
            e = (d.skel_pred_px - d.skel_gt_px) / d.skel_gt_px
            tr.append(dict(institute=i, backbone=b, condition=l, len_bias=e.median(), len_abs=e.abs().median(), len_ccc=ccc(d.skel_pred_px, d.skel_gt_px),
                           skel_outside=1 - d.skel_pred_in_gt.sum() / d.skel_pred_px.sum()))
TR = pd.DataFrame(tr); TR.to_csv(f"{OUT}/traits_with_tversky.csv", index=False)
res["traits_median_over_units"] = TR.groupby("condition")[["len_bias", "len_abs", "len_ccc", "skel_outside"]].median().to_dict()

# ---------------- Block B: topology (H7, H8) ----------------
def unit_topo(sw):
    g = sw.groupby("offset").agg(pred=("pred_stem_px", "sum"), n_pred=("n_pred", "mean"), n_ref=("n_ref", "mean"),
                                 excess_merge=("excess_merge", "mean"), excess_split=("excess_split", "mean"),
                                 spurious=("spurious", "mean"), missed=("missed", "mean"))
    g["betti"] = sw.assign(e=(sw.n_pred - sw.n_ref).abs()).groupby("offset").e.mean()
    return g
TM = ["n_pred", "betti", "excess_merge", "excess_split", "spurious", "missed"]
rows, chk3 = [], []
for b in BB:
    for i in INST:
        G = {}
        for l in LOSSES:
            topo = pd.read_csv(R.format(i=i, b=b, l=l) + "/topo_test.csv")
            sw = pd.read_csv(R.format(i=i, b=b, l=l) + "/sweep_test.csv")
            z = topo[topo.offset == 0].set_index("name"); s0 = sw[sw.offset == 0].set_index("name")
            chk3.append(int((z.pred_stem_px - s0.pred_stem_px.reindex(z.index)).abs().max()))
            G[l] = unit_topo(topo)
        for treat, comp in [("cldice", "base"), ("skelrecall", "base"), ("tversky", "base"), ("cldice", "tversky"), ("skelrecall", "tversky")]:
            t0 = G[treat].loc[0.0]; off = (G[comp].pred - t0.pred).abs().idxmin(); c = G[comp].loc[off]
            r = dict(institute=i, backbone=b, treat=treat, comp=comp, comp_offset=off, area_match=c.pred / t0.pred)
            for m in TM:
                r[f"{m}_treat"], r[f"{m}_comp_matched"], r[f"{m}_comp_off0"] = t0[m], c[m], G[comp].loc[0.0][m]
            rows.append(r)
TP = pd.DataFrame(rows); TP.to_csv(f"{OUT}/topology_matched.csv", index=False)
res["check_topo_vs_sweep_pred_px_max_abs"] = max(chk3)
h7, h8 = {}, {}
for l in ["cldice", "skelrecall"]:
    x = TP[(TP.treat == l) & (TP.comp == "tversky")]; inst = x.institute.to_numpy(); h7[l] = {}
    for m in ["n_pred", "excess_split", "betti", "excess_merge"]:
        d = (x[f"{m}_treat"] - x[f"{m}_comp_matched"]).to_numpy(); lo, hi = cluster_boot(d, inst)
        h7[l][m] = dict(mean=d.mean(), ci_low=lo, ci_high=hi, units_lower=int((d < 0).sum()))
    h7[l]["supported"] = bool(h7[l]["n_pred"]["ci_high"] < 0 and h7[l]["excess_split"]["ci_high"] < 0)
    h7[l]["area_match_range"] = (x.area_match.min(), x.area_match.max())
for l in ["cldice", "skelrecall", "tversky"]:
    x = TP[(TP.treat == l) & (TP.comp == "base")]; inst = x.institute.to_numpy(); h8[l] = {}
    for m in TM:
        d = (x[f"{m}_treat"] - x[f"{m}_comp_matched"]).to_numpy(); lo, hi = cluster_boot(d, inst)
        h8[l][m] = dict(mean=d.mean(), ci_low=lo, ci_high=hi)
    inc_merge, dec_split = h8[l]["excess_merge"]["mean"], -h8[l]["excess_split"]["mean"]
    h8[l]["decision"] = "over-merging" if inc_merge >= dec_split else "reduced splitting of reference stems"
    h8[l]["area_match_range"] = (x.area_match.min(), x.area_match.max())
res["H7"], res["H8"] = h7, h8

# ---------------- Block C: loss weight (only if the runs exist) ----------------
lam_rows = []
for i in ["UTokyo", "USASK", "INRAE", "NJAU"]:
    base = A.per_image(R.format(i=i, b="segformer_b2", l="base")).set_index("name")
    rb = pd.read_csv(R.format(i=i, b="segformer_b2", l="base") + "/metrics_ss.csv")
    for l in ["cldice", "skelrecall"]:
        for lam, tag in [(0.25, f"{l}_lam0.25"), (0.5, f"{l}_lam0.5"), (1.0, l)]:
            d = R.format(i=i, b="segformer_b2", l=tag)
            if not os.path.exists(d + "/metrics_ss.csv"): continue
            P = A.per_image(d).set_index("name"); raw = pd.read_csv(d + "/metrics_ss.csv"); r = dict(institute=i, loss=l, lam=lam,
                area_ratio=raw.pred_stem_px.sum() / raw.gt_stem_px.sum(), area_ratio_base=rb.pred_stem_px.sum() / rb.gt_stem_px.sum(),
                precision=100 * raw.inter_stem.sum() / raw.pred_stem_px.sum(), recall=100 * raw.inter_stem.sum() / raw.gt_stem_px.sum(),
                precision_base=100 * rb.inter_stem.sum() / rb.pred_stem_px.sum(), recall_base=100 * rb.inter_stem.sum() / rb.gt_stem_px.sum())
            for m in ["stem_iou", "leaf_iou", "background_iou", "head_iou"]:
                pair = pd.concat([base[m], P[m]], axis=1, keys=["b", "a"]).dropna(); diff = (pair.a - pair.b).to_numpy(); lo, hi = A.bootstrap_ci(diff)
                r[f"d_{m}"], r[f"d_{m}_lo"], r[f"d_{m}_hi"] = 100 * diff.mean(), 100 * lo, 100 * hi
            lam_rows.append(r)
if lam_rows:
    L = pd.DataFrame(lam_rows); L.to_csv(f"{OUT}/lambda_sensitivity.csv", index=False); res["blockC_runs_found"] = int((L.lam < 1).sum())

json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=lambda o: float(o) if np.isscalar(o) else list(o))
print(json.dumps(res, indent=1, default=lambda o: round(float(o), 3) if np.isscalar(o) else [round(float(v), 3) for v in o]))
