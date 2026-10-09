"""Per-image metrics: per-class intersection/union, and stem connectivity metrics."""
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .data import CLASSES, IGNORE, STEM

EIGHT = np.ones((3, 3), dtype=bool)


# Version of the per-image metric table; evaluate_run re-evaluates runs written with an older version.
METRICS_VERSION = 2
BOUNDARY_PX = 2  # fixed band for thin structures (stem half-width median ~5.5 px on GWFSS)


def boundary(mask, dilation_ratio=0.02, width=None):
    """Inner boundary band of a binary mask (Boundary IoU, Cheng et al. 2021).

    width (pixels) overrides dilation_ratio. With the default 2 % of the diagonal (~14 px at 512 px)
    the band covers the whole of a thin stem, so Boundary IoU collapses to IoU; width=BOUNDARY_PX
    keeps the band narrower than the structure.
    """
    h, w = mask.shape
    d = width if width is not None else max(1, int(round(dilation_ratio * np.sqrt(h * h + w * w))))
    m = mask.astype(np.uint8)
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=d)[1:-1, 1:-1]
    return (m - eroded).astype(bool)


def image_metrics(pred, gt):
    """pred, gt: (H,W) uint8 class maps. Pixels with gt==IGNORE are excluded."""
    valid = gt != IGNORE
    out = {}
    for c, name in enumerate(CLASSES):
        p, g = (pred == c) & valid, (gt == c) & valid
        out[f"inter_{name}"] = int((p & g).sum())
        out[f"union_{name}"] = int((p | g).sum())
    p, g = (pred == STEM) & valid, (gt == STEM) & valid
    out["gt_stem_px"], out["pred_stem_px"] = int(g.sum()), int(p.sum())
    sp, sg = skeletonize(p), skeletonize(g)
    out["skel_pred_px"], out["skel_gt_px"] = int(sp.sum()), int(sg.sum())
    out["skel_pred_in_gt"] = int((sp & g).sum())
    out["skel_gt_in_pred"] = int((sg & p).sum())
    tprec = out["skel_pred_in_gt"] / out["skel_pred_px"] if out["skel_pred_px"] else np.nan
    tsens = out["skel_gt_in_pred"] / out["skel_gt_px"] if out["skel_gt_px"] else np.nan
    out["stem_skel_recall"] = tsens
    out["stem_cldice"] = (2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) > 0 else (0.0 if g.any() or p.any() else np.nan)
    out["stem_cc_pred"] = int(ndimage.label(p, structure=EIGHT)[1])
    out["stem_cc_gt"] = int(ndimage.label(g, structure=EIGHT)[1])
    out["stem_betti0_err"] = abs(out["stem_cc_pred"] - out["stem_cc_gt"])
    bp, bg = boundary(p) & valid, boundary(g) & valid
    out["inter_stem_boundary"], out["union_stem_boundary"] = int((bp & bg).sum()), int((bp | bg).sum())
    bp, bg = boundary(p, width=BOUNDARY_PX) & valid, boundary(g, width=BOUNDARY_PX) & valid
    out["inter_stem_boundary2px"], out["union_stem_boundary2px"] = int((bp & bg).sum()), int((bp | bg).sum())
    out["metrics_version"] = METRICS_VERSION
    return out


def dataset_iou(df):
    """Dataset-level IoU per class from summed intersections and unions."""
    res = {}
    for name in CLASSES:
        u = df[f"union_{name}"].sum()
        res[f"iou_{name}"] = df[f"inter_{name}"].sum() / u if u else np.nan
    res["miou"] = float(np.nanmean([res[f"iou_{n}"] for n in CLASSES]))
    u = df["union_stem_boundary"].sum()
    res["boundary_iou_stem"] = df["inter_stem_boundary"].sum() / u if u else np.nan
    if "union_stem_boundary2px" in df:
        u = df["union_stem_boundary2px"].sum()
        res["boundary2px_iou_stem"] = df["inter_stem_boundary2px"].sum() / u if u else np.nan
    sp, sg = df["skel_pred_px"].sum(), df["skel_gt_px"].sum()
    tprec = df["skel_pred_in_gt"].sum() / sp if sp else np.nan
    tsens = df["skel_gt_in_pred"].sum() / sg if sg else np.nan
    res["cldice_stem"] = 2 * tprec * tsens / (tprec + tsens) if (tprec + tsens) > 0 else np.nan
    res["skel_recall_stem"] = tsens
    res["betti0_err_stem_mean"] = df["stem_betti0_err"].mean()
    return res
