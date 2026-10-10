"""Tables 7 and 8 and Supplementary Tables S21 and S22 (review blocks A-C) from the outputs of rev_block6.py.
Run from the project root after rev_block6.py: python3 analysis/review2/rev_block6_tables.py"""
import json
import os

import numpy as np
import pandas as pd

OUT = os.environ.get("OUT", "analysis/review2")
INST = ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UQ", "USASK", "UTokyo"]


def cluster_boot(diff, inst, n=10000, seed=0):  # as in rev_block6.py
    g = np.random.default_rng(seed); boots = []
    for _ in range(n):
        s = g.choice(INST, len(INST)); boots.append(np.mean(np.concatenate([diff[inst == j] for j in s])))
    return np.percentile(boots, [2.5, 97.5])


def f(x, d=1):
    return f"{x:+.{d}f}".replace("-", "−")


U = pd.read_csv(f"{OUT}/units_with_tversky.csv"); res = json.load(open(f"{OUT}/results.json"))
T10 = pd.read_csv(f"{OUT}/t10_with_tversky.csv")
lo = U.area_ratio_base < 0.7
print("Table 7")
for l in ["cldice", "skelrecall", "tversky"]:
    d = U[f"delta_{l}"].to_numpy(); ci = cluster_boot(d, U.institute.to_numpy())
    t10 = T10[l] - T10["base+T10"]; ci10 = cluster_boot(t10.to_numpy(), T10.institute.to_numpy())
    print(l, f"Δ {f(d.mean())} [{f(ci[0])}, {f(ci[1])}] ({(d > 0).sum()}/20)", f"A<0.7: {(d[lo] > 0).sum()}/14",
          f"median A {U[f'area_ratio_{l}'].median():.2f}", f"prec {U[f'precision_{l}'].median():.1f} rec {U[f'recall_{l}'].median():.1f}",
          f"vs base+T10 {f(t10.mean())} [{f(ci10[0])}, {f(ci10[1])}] ({(t10 > 0).sum()}/20)")
print("base", f"median A {U.area_ratio_base.median():.2f}", f"prec {U.precision_base.median():.1f} rec {U.recall_base.median():.1f}")

TP = pd.read_csv(f"{OUT}/topology_matched.csv")
print("Table 8")
for treat, comp in [("cldice", "base"), ("skelrecall", "base"), ("tversky", "base"), ("cldice", "tversky"), ("skelrecall", "tversky")]:
    x = TP[(TP.treat == treat) & (TP.comp == comp)]; inst = x.institute.to_numpy(); cells = []
    for m in ["n_pred", "excess_merge", "excess_split", "spurious", "missed"]:
        d = (x[f"{m}_treat"] - x[f"{m}_comp_matched"]).to_numpy(); a, b = cluster_boot(d, inst)
        cells.append(f"{f(d.mean(), 2)} [{f(a, 2)}, {f(b, 2)}]")
    print(treat, comp, f"{x.area_match.min():.2f}–{x.area_match.max():.2f}", " | ".join(cells))
print("offset-0 means per model", {l: round(TP[(TP.treat == l) & (TP.comp == 'base')].n_pred_treat.mean(), 1) for l in ["cldice", "skelrecall", "tversky"]},
      "baseline", round(TP[(TP.treat == "cldice") & (TP.comp == "base")].n_pred_comp_off0.mean(), 1))

# Supplementary Table S21: Block B at offset 0 (both models without an offset)
print("\nTable S21")
for treat, comp in [("cldice", "base"), ("skelrecall", "base"), ("tversky", "base"), ("cldice", "tversky"), ("skelrecall", "tversky")]:
    x = TP[(TP.treat == treat) & (TP.comp == comp)]; inst = x.institute.to_numpy(); cells = []
    for m in ["n_pred", "excess_merge", "excess_split", "spurious", "missed"]:
        d = (x[f"{m}_treat"] - x[f"{m}_comp_off0"]).to_numpy(); a, b = cluster_boot(d, inst)
        cells.append(f"{f(d.mean(), 2)} [{f(a, 2)}, {f(b, 2)}]")
    print(f"| {treat} vs {comp} | " + " | ".join(cells) + " |")

# Supplementary Table S22: Block C (loss weight)
print("\nTable S22")
L = pd.read_csv(f"{OUT}/lambda_sensitivity.csv")
NAME = {"cldice": "clDice", "skelrecall": "Skeleton Recall"}
for i in ["UTokyo", "USASK", "INRAE", "NJAU"]:
    for l in ["cldice", "skelrecall"]:
        for _, r in L[(L.institute == i) & (L.loss == l)].sort_values("lam").iterrows():
            side = []
            for c in ["leaf", "background", "head"]:
                s = f(r[f"d_{c}_iou"]) + ("*" if r[f"d_{c}_iou_hi"] < 0 or r[f"d_{c}_iou_lo"] > 0 else "")
                side.append(s)
            print(f"| {i} ({r.area_ratio_base:.2f}) | {NAME[l]} | {r.lam:g} | {r.area_ratio:.2f} | {r.precision:.1f} | {r.recall:.1f} | "
                  f"{f(r.d_stem_iou)} [{f(r.d_stem_iou_lo)}, {f(r.d_stem_iou_hi)}] | " + " | ".join(side) + " |")
    b = L[L.institute == i].iloc[0]
    print(f"baseline {i}: precision {b.precision_base:.1f} recall {b.recall_base:.1f}")
