"""Loss functions: CE + Dice baseline, clDice, Skeleton Recall and a recall-weighted Tversky control on the stem channel."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss

from .data import IGNORE, STEM


def soft_erode(x):
    p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(x):
    return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))


def soft_skel(x, iters):
    """Soft skeleton (Shit et al., CVPR 2021). x: (B,1,H,W) in [0,1]."""
    skel = F.relu(x - soft_dilate(soft_erode(x)))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_dilate(soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SegLoss(nn.Module):
    """total = CE + Dice + lam * connectivity term (stem channel only).

    kind: 'base' | 'cldice' | 'skelrecall' | 'tversky'
    'tversky' (review control, specifications/03): recall-weighted overlap without a skeleton term.
    """

    def __init__(self, kind="base", lam=1.0, skel_iters=10, smooth=1.0, alpha=0.3, beta=0.7):
        super().__init__()
        assert kind in ("base", "cldice", "skelrecall", "tversky")
        self.kind, self.lam, self.skel_iters, self.smooth = kind, lam, skel_iters, smooth
        self.alpha, self.beta = alpha, beta
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
        self.dice = DiceLoss(mode="multiclass", ignore_index=IGNORE)

    def forward(self, logits, target, skeleton=None):
        ce, dice = self.ce(logits, target), self.dice(logits, target)
        parts = {"ce": ce.item(), "dice": dice.item()}
        total = ce + dice
        if self.kind == "base":
            return total, parts
        valid = (target != IGNORE).float().unsqueeze(1)
        p = torch.softmax(logits.float(), 1)[:, STEM:STEM + 1] * valid
        y = (target == STEM).float().unsqueeze(1)
        s = self.smooth
        if self.kind == "cldice":
            sp, sy = soft_skel(p, self.skel_iters), soft_skel(y, self.skel_iters)
            tprec = ((sp * y).sum() + s) / (sp.sum() + s)
            tsens = ((sy * p).sum() + s) / (sy.sum() + s)
            conn = 1.0 - 2.0 * tprec * tsens / (tprec + tsens)
        elif self.kind == "tversky":
            yv = y * valid
            tp, fp, fn = (p * yv).sum(), (p * (1 - yv)).sum(), ((1 - p) * yv).sum()
            conn = 1.0 - (tp + s) / (tp + self.alpha * fp + self.beta * fn + s)
        else:
            sk = skeleton.unsqueeze(1).to(p.dtype)
            if sk.sum() == 0:
                conn = p.sum() * 0.0
            else:
                conn = 1.0 - ((p * sk).sum() + s) / (sk.sum() + s)
        parts["conn"] = conn.item()
        return total + self.lam * conn, parts
