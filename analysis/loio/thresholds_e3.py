"""E3 threshold controls from offset sweeps (spec: specifications/01_spec_loio_calibration_traits.md). Run from the project root."""
import glob, numpy as np, pandas as pd
CL = ["background", "head", "stem", "leaf"]
INST = ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UQ", "USASK", "UTokyo"]
LOSSES = ["base", "cldice", "skelrecall"]

def best_offset(sw, names):
    s = sw[sw.name.isin(names)].groupby("offset")[["inter_stem", "union_stem"]].sum()
    iou = (s.inter_stem / s.union_stem.replace(0, np.nan)).fillna(0).round(6)
    top = iou[iou == iou.max()].index
    return min(top, key=abs)

def score(sw, names, b):
    s = sw[(sw.offset == b) & sw.name.isin(names)][[f"{k}_{c}" for c in CL for k in ("inter", "union")]].sum()
    ious = [s[f"inter_{c}"] / s[f"union_{c}"] if s[f"union_{c}"] else np.nan for c in CL]
    return 100 * ious[2], 100 * np.nanmean(ious)

def run_units(units):  # units: list of (split, institute/None, backbone, {loss: [run dirs]})
    rows, chk = [], []
    for split, inst, bb, dirs in units:
        SW = {l: [pd.read_csv(f"{d}/sweep_test.csv") for d in dirs[l]] for l in LOSSES}
        VAL = {l: [pd.read_csv(f"{d}/sweep_val.csv") if split.startswith("loio") else None for d in dirs[l]] for l in LOSSES}
        names = sorted(SW["base"][0].name.unique())
        # consistency: sweep offset 0 vs main evaluation
        for l in LOSSES:
            for d, sw in zip(dirs[l], SW[l]):
                m = pd.read_csv(f"{d}/metrics_ss.csv")
                chk.append(100 * (sw[sw.offset == 0].inter_stem.sum() / sw[sw.offset == 0].union_stem.sum() - m.inter_stem.sum() / m.union_stem.sum()))
        base = dict(split=split, institute=inst, backbone=bb)
        # no offset and T-val (LOIO only; region T-val is Table 11)
        for l in LOSSES:
            r0 = [score(sw, names, 0.0) for sw in SW[l]]
            rows.append({**base, "condition": l, "k": 0, "stem_iou": np.mean([r[0] for r in r0]), "miou": np.mean([r[1] for r in r0])})
            if split.startswith("loio"):
                rv = [score(sw, names, best_offset(v, v.name.unique())) for sw, v in zip(SW[l], VAL[l])]
                bs = [best_offset(v, v.name.unique()) for v in VAL[l]]
                rows.append({**base, "condition": l + "+Tval", "k": 0, "offset": np.mean(bs), "stem_iou": np.mean([r[0] for r in rv]), "miou": np.mean([r[1] for r in rv])})
        # T-k: draws shared across conditions (and backbones) of the same test institution
        for k in (5, 10):
            acc = {}
            for s in range(20):
                cal = list(np.random.default_rng(s).choice(names, k, replace=False)); rest = [n for n in names if n not in cal]
                for l in LOSSES:
                    for sw in SW[l]:
                        b = best_offset(sw, cal)
                        for cond, off in [(l, 0.0), (l + f"+T{k}", b)]:
                            st, mi = score(sw, rest, off)
                            a = acc.setdefault(cond, [[], [], []]); a[0].append(st); a[1].append(mi); a[2].append(off)
            for cond, (st, mi, off) in acc.items():
                rows.append({**base, "condition": cond, "k": k, "offset": np.mean(off), "stem_iou": np.mean(st), "miou": np.mean(mi)})
    return pd.DataFrame(rows), np.array(chk)

units = []
for bb in ["segformer_b2", "upernet_convnext_t"]:
    for i in INST:
        units.append((f"loio_{i}", i, bb, {l: [f"results/loio_{i}/{bb}/{l}/scale1.0/seed0"] for l in LOSSES}))
for bb in ["deeplabv3plus_r101", "segformer_b1", "segformer_b2", "upernet_convnext_t"]:
    units.append(("region", "UQ", bb, {l: sorted(glob.glob(f"results/region/{bb}/{l}/scale1.0/seed*")) for l in LOSSES}))
D, chk = run_units(units)
D.to_csv("analysis/loio/e3_thresholds.csv", index=False)
print("sweep offset-0 vs main evaluation, stem IoU difference (points): max |d| = %.3f, mean = %.3f" % (np.abs(chk).max(), chk.mean()))
