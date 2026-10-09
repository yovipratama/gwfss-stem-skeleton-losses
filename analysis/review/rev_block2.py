"""Stage 4 revision analyses, block 2 (offline). REV-22 matched-area comparison; REV-17 seed-level spread + two-level bootstrap;
REV-19 2-px Boundary IoU paired test; REV-23 n per test + false-positive stems on stem-free images; REV-15 x2 vs x1 baseline;
REV-41 image-level moderator; REV-27 stem radius distribution; REV-34 per-institution stem width vs A."""
import glob, json, importlib.util, numpy as np, pandas as pd, cv2
from scipy.stats import wilcoxon, spearmanr
from scipy import ndimage
from skimage.morphology import skeletonize
spec = importlib.util.spec_from_file_location("A", "src/gwfss_stem/analysis.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
OUT = "analysis/review"; res = {}
INST = ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UQ", "USASK", "UTokyo"]
BB4 = ["deeplabv3plus_r101", "segformer_b1", "segformer_b2", "upernet_convnext_t"]

# ---- REV-22: compare each loss model with the baseline at matched predicted stem area (sweeps: LOIO seed0, region all seeds)
def at_area(sw, target):
    g = sw.groupby("offset").agg(pred=("pred_stem_px", "sum"), inter=("inter_stem", "sum"), union=("union_stem", "sum"),
                                 cc=("stem_cc_pred", "mean"), ccgt=("stem_cc_gt", "mean"))
    g["betti"] = sw.assign(e=(sw.stem_cc_pred - sw.stem_cc_gt).abs()).groupby("offset").e.mean()
    b = (g.pred - target).abs().idxmin(); r = g.loc[b]
    return dict(offset=b, area_match=r.pred / target, iou=100 * r.inter / r.union, cc=r.cc, betti=r.betti, ccgt=r.ccgt)
rows = []
units = [(f"results/loio_{i}/{b}/{{l}}/scale1.0/seed0", "loio", i, b) for b in ["segformer_b2", "upernet_convnext_t"] for i in INST]
units += [(f"results/region/{b}/{{l}}/scale1.0/seed{s}", "region", "UQ", b) for b in BB4 for s in range(3)]
for pat, split, i, b in units:
    base = pd.read_csv(pat.format(l="base") + "/sweep_test.csv")
    for l in ["cldice", "skelrecall"]:
        sw = pd.read_csv(pat.format(l=l) + "/sweep_test.csv"); z = sw[sw.offset == 0]
        tgt = z.pred_stem_px.sum(); m = at_area(base, tgt)
        rows.append(dict(split=split, institute=i, backbone=b, run=pat, loss=l, loss_iou=100 * z.inter_stem.sum() / z.union_stem.sum(),
                         loss_cc=z.stem_cc_pred.mean(), loss_betti=(z.stem_cc_pred - z.stem_cc_gt).abs().mean(), ref_cc=z.stem_cc_gt.mean(),
                         base_off=m["offset"], base_area_match=m["area_match"], base_iou=m["iou"], base_cc=m["cc"], base_betti=m["betti"]))
MA = pd.DataFrame(rows)
MA = MA.groupby(["split", "institute", "backbone", "loss"]).mean(numeric_only=True).reset_index()   # average region seeds
MA.to_csv(f"{OUT}/matched_area.csv", index=False)
s = {}
for l in ["cldice", "skelrecall"]:
    x = MA[MA.loss == l]
    s[l] = dict(units=len(x), iou_loss_minus_base_mean=(x.loss_iou - x.base_iou).mean(), iou_loss_higher=int((x.loss_iou > x.base_iou).sum()),
                cc_loss_minus_base_median=(x.loss_cc - x.base_cc).median(), cc_loss_lower=int((x.loss_cc < x.base_cc).sum()),
                betti_loss_minus_base_mean=(x.loss_betti - x.base_betti).mean(), betti_loss_lower=int((x.loss_betti < x.base_betti).sum()),
                area_match_range=(round(x.base_area_match.min(), 2), round(x.base_area_match.max(), 2)))
res["matched_area"] = s

# ---- REV-17: seed-level spread of per-image delta (random, region), two-level bootstrap (seeds x images)
def per_image(d): return A.per_image(d).set_index("name")
rows = []
for split in ["random", "region"]:
    for b in BB4:
        B = {s: per_image(f"results/{split}/{b}/base/scale1.0/seed{s}") for s in range(3)}
        for l in ["cldice", "skelrecall"]:
            Lr = {s: per_image(f"results/{split}/{b}/{l}/scale1.0/seed{s}") for s in range(3)}
            # matrix image x (loss seed, base seed): all 9 pairings (seeds are not paired across arms)
            names = sorted(set(B[0].index))
            M = np.stack([[Lr[sl].loc[names, "stem_iou"].values - B[sb].loc[names, "stem_iou"].values for sb in range(3)] for sl in range(3)])  # 3x3xN
            seed_delta = [np.nanmean(Lr[s].loc[names, "stem_iou"].values - B[s].loc[names, "stem_iou"].values) for s in range(3)]
            g = np.random.default_rng(0); boots = []
            valid = ~np.isnan(M).any(axis=(0, 1)); M = M[:, :, valid]; n = M.shape[2]
            for _ in range(2000):
                sl = g.integers(0, 3, 3); sb = g.integers(0, 3, 3); im = g.integers(0, n, n)
                boots.append(np.mean([M[a, c][im].mean() for a, c in zip(sl, sb)]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(split=split, backbone=b, loss=l, seed_delta_min=100 * min(seed_delta), seed_delta_max=100 * max(seed_delta),
                             mean=100 * M.mean(), two_level_ci_low=100 * lo, two_level_ci_high=100 * hi))
SD = pd.DataFrame(rows); SD.to_csv(f"{OUT}/seed_variance_two_level.csv", index=False)
old = pd.concat([pd.read_csv("analysis/main/paired_tests_sq1.csv").assign(split="random"),
                 pd.read_csv("analysis/main/paired_region_s1.0.csv").assign(split="region")])
old = old[old.metric == "stem_iou"][["split", "backbone", "loss", "improves", "degrades", "ci_low", "ci_high"]]
SD = SD.merge(old, on=["split", "backbone", "loss"])
SD["two_level_excludes0"] = (SD.two_level_ci_low > 0) | (SD.two_level_ci_high < 0)
SD["image_only_excludes0"] = (SD.ci_low > 0) | (SD.ci_high < 0)
res["seed_two_level_changes"] = int((SD.two_level_excludes0 != SD.image_only_excludes0).sum())
SD.to_csv(f"{OUT}/seed_variance_two_level.csv", index=False)

# ---- REV-19 + REV-23: boundary 2px paired test; n per test; FP stems on stem-free images
rows, fp = [], []
for split in ["random", "region"]:
    for b in BB4:
        def bimg(l):
            fr = []
            for d in sorted(glob.glob(f"results/{split}/{b}/{l}/scale1.0/seed*")):
                x = pd.read_csv(f"{d}/metrics_ss.csv"); x["biou"] = x.inter_stem_boundary2px / x.union_stem_boundary2px.replace(0, np.nan)
                x["fp_free"] = (x.gt_stem_px == 0) & (x.pred_stem_px > 0); x["fp_px"] = np.where(x.gt_stem_px == 0, x.pred_stem_px, np.nan)
                fr.append(x)
            return pd.concat(fr).groupby("name")[["biou", "fp_free", "fp_px", "gt_stem_px"]].mean()
        Bm = bimg("base")
        for l in ["base", "cldice", "skelrecall"]:
            Lm = bimg(l); free = Lm[Lm.gt_stem_px == 0]
            fp.append(dict(split=split, backbone=b, loss=l, n_stem_free=len(free), frac_with_fp=free.fp_free.mean(), mean_fp_px=free.fp_px.mean()))
            if l == "base": continue
            pair = pd.concat([Bm.biou, Lm.biou], axis=1, keys=["b", "a"]).dropna(); diff = (pair.a - pair.b).values
            lo, hi = A.bootstrap_ci(diff)
            rows.append(dict(split=split, backbone=b, loss=l, n=len(diff), mean_diff=100 * diff.mean(), ci_low=100 * lo, ci_high=100 * hi, p=wilcoxon(diff).pvalue))
BI = pd.DataFrame(rows); BI["p_holm"] = BI.groupby("split").p.transform(lambda s: A.holm(s.to_numpy()))
BI["improves"] = (BI.ci_low > 0) & (BI.p_holm < .05); BI["degrades"] = (BI.ci_high < 0) & (BI.p_holm < .05)
BI.to_csv(f"{OUT}/boundary2px_tests.csv", index=False); FP = pd.DataFrame(fp); FP.to_csv(f"{OUT}/fp_stem_free.csv", index=False)
res["boundary2px"] = {f"{k[0]}/{k[1]}": v for k, v in BI.groupby(["split", "loss"])[["improves", "degrades"]].sum().to_dict("index").items()}
res["n_per_test_stem_iou"] = {f"{r.split}/{r.backbone}/{r.loss}": int(r.n) for r in BI.itertuples()}

# ---- REV-15: x2 vs x1 baseline, per-image paired, native-resolution evaluation
rows = []
for b in ["segformer_b1", "deeplabv3plus_r101"]:
    def pim(sc): return pd.concat([A.per_image(d).assign(seed=d[-1]) for d in sorted(glob.glob(f"results/random/{b}/base/scale{sc}/seed*"))]).groupby("name").stem_iou.mean()
    pair = pd.concat([pim("1.0"), pim("2.0")], axis=1, keys=["x1", "x2"]).dropna(); diff = (pair.x2 - pair.x1).values; lo, hi = A.bootstrap_ci(diff)
    rows.append(dict(backbone=b, n=len(diff), mean_diff=100 * diff.mean(), ci_low=100 * lo, ci_high=100 * hi, p=wilcoxon(diff).pvalue))
res["x2_vs_x1_baseline"] = pd.DataFrame(rows).round(4).to_dict("records")

# ---- REV-41: image-level moderator for clDice in-distribution (per-image delta vs reference stem pixels)
rows = []
for b in BB4:
    def pim(l): return pd.concat([A.per_image(d) for d in sorted(glob.glob(f"results/random/{b}/{l}/scale1.0/seed*"))]).groupby("name").stem_iou.mean()
    m = pd.read_csv("data_cache/gw/manifest.csv").set_index("name")
    dlt = (pim("cldice") - pim("base")).dropna()
    rows.append(dict(backbone=b, rho_delta_vs_stem_px=spearmanr(dlt.values, m.loc[dlt.index, "stem_px"].values).statistic))
res["cldice_moderator_random"] = rows

# ---- REV-27 + REV-34: stem radius distribution (reference masks), per institution
m = pd.read_csv("data_cache/gw/manifest.csv")
rad_all, per_inst = [], []
for inst, g in m.groupby("institute"):
    r_inst, maxr = [], []
    for n in g.name:
        mk = cv2.imread(f"data_cache/gw/masks/{n}", cv2.IMREAD_UNCHANGED); st = (mk == 2)
        if st.sum() < 500: continue
        dt = ndimage.distance_transform_edt(st); sk = skeletonize(st); r = dt[sk]
        r_inst.append(r); maxr.append(r.max())
    r_inst = np.concatenate(r_inst); rad_all.append(r_inst)
    per_inst.append(dict(institute=inst, img_px=int(g.height.median()), median_radius=np.median(r_inst), p90_radius=np.percentile(r_inst, 90),
                         frac_gt10=(r_inst > 10).mean(), frac_gt5_native_equiv_x2=(r_inst > 5).mean(), median_max_radius=np.median(maxr)))
PI = pd.DataFrame(per_inst); ra = np.concatenate(rad_all)
res["radius_all"] = dict(median=np.median(ra), p90=np.percentile(ra, 90), frac_gt10_native=(ra > 10).mean(), frac_gt10_at_x2=(ra > 5).mean(), frac_gt40_at_x2=(ra > 20).mean())
U = pd.read_csv("analysis/loio/units.csv"); Ai = U.groupby("institute").area_ratio_base.mean()
PI["A_mean"] = PI.institute.map(Ai); PI.to_csv(f"{OUT}/stem_radius_per_institute.csv", index=False)
res["rho_A_vs_median_radius"] = spearmanr(PI.A_mean, PI.median_radius).statistic
json.dump(res, open(f"{OUT}/block2_results.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(json.dumps(res, indent=1, default=lambda v: round(float(v), 3)))
print(SD.round(2).to_string(index=False)); print(FP.round(3).to_string(index=False)); print(PI.round(3).to_string(index=False))
