"""Aggregate results and run the pre-declared statistical comparisons."""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRIMARY = ["stem_iou", "stem_cldice", "stem_skel_recall", "stem_betti0_err"]
# SQ4 side effects: per-image IoU of the other classes (tested with the same paired procedure)
SECONDARY = ["head_iou", "leaf_iou", "background_iou"]
METRICS = PRIMARY + SECONDARY


def collect(results_root, tag="ss"):
    """One row per finished+evaluated run with dataset-level metrics."""
    rows = []
    for summ in glob.glob(os.path.join(results_root, "*", "*", "*", "*", "*", f"summary_{tag}.json")):
        run_dir = os.path.dirname(summ)
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)["config"]
        with open(summ) as f:
            s = json.load(f)
        rows.append({**{k: cfg[k] for k in ("split", "backbone", "loss", "scale", "seed")}, **s, "run_dir": run_dir})
    return pd.DataFrame(rows)


def summary_table(runs, metrics=("iou_stem", "cldice_stem", "skel_recall_stem", "betti0_err_stem_mean",
                                 "boundary2px_iou_stem", "boundary_iou_stem", "miou", "iou_head", "iou_leaf", "iou_background")):
    metrics = [m for m in metrics if m in runs]
    g = runs.groupby(["split", "scale", "backbone", "loss"])
    mean, sd = g[list(metrics)].mean(), g[list(metrics)].std()
    out = mean.copy()
    for m in metrics:
        out[m] = [f"{a:.4f} ± {b:.4f}" if not np.isnan(b) else f"{a:.4f}" for a, b in zip(mean[m], sd[m])]
    out["n_seeds"] = g.size()
    return out.reset_index()


def per_image(run_dir, tag="ss"):
    df = pd.read_csv(os.path.join(run_dir, f"metrics_{tag}.csv"))
    for c in ("stem", "head", "leaf", "background"):
        df[f"{c}_iou"] = df[f"inter_{c}"] / df[f"union_{c}"].replace(0, np.nan)
    return df[["name"] + METRICS]


def seed_averaged(runs, tag="ss"):
    frames = []
    for _, r in runs.iterrows():
        d = per_image(r["run_dir"], tag)
        for k in ("split", "scale", "backbone", "loss", "seed"):
            d[k] = r[k]
        frames.append(d)
    allimg = pd.concat(frames)
    return allimg.groupby(["split", "scale", "backbone", "loss", "name"])[METRICS].mean().reset_index()


def bootstrap_ci(diff, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired_tests(runs, baseline="base", tag="ss"):
    """Per (split, scale, backbone, loss != baseline, metric): paired per-image comparison vs baseline.

    Betti-0 error is 'lower is better'; for it a negative difference is an improvement.
    Secondary metrics (other-class IoU) answer SQ4: 'improves' is True/False, and 'degrades'
    flags a significant decrease (CI entirely below 0, Holm p < 0.05).
    Holm correction is applied within each metric across all comparisons.
    """
    img = seed_averaged(runs, tag)
    rows = []
    for (split, scale, bb), grp in img.groupby(["split", "scale", "backbone"]):
        base = grp[grp["loss"] == baseline].set_index("name")
        for loss in sorted(set(grp["loss"]) - {baseline}):
            other = grp[grp["loss"] == loss].set_index("name")
            for m in METRICS:
                pair = pd.concat([base[m], other[m]], axis=1, keys=["a", "b"]).dropna()
                if len(pair) < 5:
                    continue
                diff = (pair["b"] - pair["a"]).to_numpy()
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                lo, hi = bootstrap_ci(diff)
                rows.append(dict(split=split, scale=scale, backbone=bb, loss=loss, metric=m, n_images=len(diff),
                                 mean_diff=diff.mean(), ci_low=lo, ci_high=hi, p_wilcoxon=p))
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = res.groupby("metric")["p_wilcoxon"].transform(lambda s: holm(s.to_numpy()))
        better = np.where(res["metric"] == "stem_betti0_err", res["ci_high"] < 0, res["ci_low"] > 0)
        worse = np.where(res["metric"] == "stem_betti0_err", res["ci_low"] > 0, res["ci_high"] < 0)
        res["improves"] = better & (res["p_holm"] < 0.05)
        res["degrades"] = worse & (res["p_holm"] < 0.05)
        res["role"] = np.where(res["metric"].isin(PRIMARY), "primary", "secondary")
    return res


def consistency(tests):
    """Pre-declared consistency criterion: improvement (CI excludes 0, Holm p<0.05) in every backbone."""
    if tests.empty:
        return tests
    return (tests.groupby(["split", "scale", "loss", "metric"])
            .agg(n_backbones=("backbone", "nunique"), n_improves=("improves", "sum"), n_degrades=("degrades", "sum"))
            .assign(consistent=lambda d: d["n_backbones"] == d["n_improves"]).reset_index())
