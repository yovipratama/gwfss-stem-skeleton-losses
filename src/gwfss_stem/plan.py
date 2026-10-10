"""Experiment plan derived from the Stage-1 Methodology Blueprint."""
from .models import BACKBONES

LOSSES = ["base", "cldice", "skelrecall"]
SEEDS = [0, 1, 2]


def full_plan():
    runs = []
    # SQ1: consistency across backbones (random split stratified by institute)
    for b in BACKBONES:
        for l in LOSSES:
            for s in SEEDS:
                runs.append(dict(sq="SQ1", split="random", backbone=b, loss=l, seed=s, scale=1.0))
    # SQ2: cross-institute robustness (GWFSS region split)
    for b in BACKBONES:
        for l in LOSSES:
            for s in SEEDS:
                runs.append(dict(sq="SQ2", split="region", backbone=b, loss=l, seed=s, scale=1.0))
    # SQ3: resolution confound (2x input scale); multi-scale inference is evaluated on SQ1 models
    for b in ["segformer_b1", "deeplabv3plus_r101"]:
        for l in LOSSES:
            for s in SEEDS:
                runs.append(dict(sq="SQ3", split="random", backbone=b, loss=l, seed=s, scale=2.0))
    return runs


def select(runs, **filters):
    """Filter runs, e.g. select(full_plan(), sq='SQ1', backbone='segformer_b1')."""
    def ok(r):
        return all((r[k] in v) if isinstance(v, (list, tuple, set)) else r[k] == v for k, v in filters.items())
    return [r for r in runs if ok(r)]


# LOIO (specification: specifications/01_spec_loio_calibration_traits.md)
LOIO_INSTITUTES = ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege", "UQ", "USASK", "UTokyo"]
LOIO_BACKBONES = ["segformer_b2", "upernet_convnext_t"]


def loio_plan():
    return [dict(sq="LOIO", split=f"loio_{i}", backbone=b, loss=l, seed=0, scale=1.0)
            for b in LOIO_BACKBONES for i in LOIO_INSTITUTES for l in LOSSES]


# Revision runs (added in revision; specification specifications/02_spec_revision_runs.md)
def revision_plan():
    runs = [dict(sq="R-K40", split="random", backbone=b, loss="cldice", seed=s, scale=2.0, skel_iters=40)
            for b in ["segformer_b1", "deeplabv3plus_r101"] for s in SEEDS]
    runs += [dict(sq="R-S1", split=f"loio_{i}", backbone=b, loss="base", seed=1, scale=1.0)
             for b in LOIO_BACKBONES for i in LOIO_INSTITUTES]
    return runs


# Review controls (added in review; specification specifications/03_spec_recall_control_topology.md)
LAMBDA_FOLDS = ["UTokyo", "USASK", "INRAE", "NJAU"]   # two lowest and two highest seed-0 SegFormer-B2 area ratios


def review_plan():
    runs = [dict(sq="R-TV", split=f"loio_{i}", backbone=b, loss="tversky", seed=0, scale=1.0)
            for b in LOIO_BACKBONES for i in LOIO_INSTITUTES]
    runs += [dict(sq="R-LAM", split=f"loio_{i}", backbone="segformer_b2", loss=l, seed=0, scale=1.0, lam=lam)
             for i in LAMBDA_FOLDS for l in ["cldice", "skelrecall"] for lam in [0.25, 0.5]]
    return runs
