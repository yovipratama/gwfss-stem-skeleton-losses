"""Resumable training and evaluation. Designed for Colab sessions that may disconnect."""
import csv
import json
import os
import platform
import random
import socket
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import MEAN, STD, SegDataset, load_split
from .losses import SegLoss
from .metrics import METRICS_VERSION, dataset_iou, image_metrics
from .models import build_model

DEFAULTS = dict(
    split="random", backbone="segformer_b1", loss="base", lam=1.0, seed=0, scale=1.0,
    iters=8000, batch_size=8, lr=1e-4, weight_decay=1e-2, warmup=500, crop=512,
    skel_iters=10, amp=True, num_workers=2, val_every=1000, ckpt_every=500,
)
STALE_LOCK_SEC = 30 * 60


def run_name(cfg):
    loss = cfg["loss"]
    if cfg.get("skel_iters", DEFAULTS["skel_iters"]) != DEFAULTS["skel_iters"]:
        loss += f"_k{cfg['skel_iters']}"  # non-default soft-skeleton depth (revision run R-K40)
    return f"{cfg['split']}/{cfg['backbone']}/{loss}/scale{cfg['scale']}/seed{cfg['seed']}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_info():
    import segmentation_models_pytorch as smp

    return dict(
        host=socket.gethostname(), python=platform.python_version(), torch=torch.__version__,
        smp=smp.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def acquire_lock(out_dir):
    """A lock is honoured only if it is fresh AND held by another machine.

    A lock left by this same Colab VM (e.g. after a crashed cell) is taken over immediately.
    """
    path = os.path.join(out_dir, "RUNNING")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < STALE_LOCK_SEC:
        with open(path) as f:
            if f.read().strip() != socket.gethostname():
                return False
    with open(path, "w") as f:
        f.write(socket.gethostname())
    return True


def touch_lock(out_dir):
    os.utime(os.path.join(out_dir, "RUNNING"))


def release_lock(out_dir):
    p = os.path.join(out_dir, "RUNNING")
    if os.path.exists(p):
        os.remove(p)


def poly_lr(it, cfg):
    if it < cfg["warmup"]:
        return cfg["lr"] * (it + 1) / cfg["warmup"]
    return cfg["lr"] * (1 - (it - cfg["warmup"]) / (cfg["iters"] - cfg["warmup"])) ** 0.9


@torch.no_grad()
def _forward_padded(model, x):
    h, w = x.shape[-2:]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    x = F.pad(x, (0, pw, 0, ph), mode="reflect" if (ph < h and pw < w) else "constant")
    return model(x)[..., :h, :w]


@torch.no_grad()
def _forward_sliding(model, x, window=1024, stride=768, num_classes=4):
    h, w = x.shape[-2:]
    if max(h, w) <= window:
        return _forward_padded(model, x)
    out = torch.zeros((x.shape[0], num_classes, h, w), device=x.device)
    cnt = torch.zeros((1, 1, h, w), device=x.device)
    ys = list(range(0, max(h - window, 0) + 1, stride))
    xs = list(range(0, max(w - window, 0) + 1, stride))
    if ys[-1] + window < h:
        ys.append(h - window)
    if xs[-1] + window < w:
        xs.append(w - window)
    for y in ys:
        for x0 in xs:
            tile = x[..., y:y + window, x0:x0 + window]
            out[..., y:y + tile.shape[-2], x0:x0 + tile.shape[-1]] += _forward_padded(model, tile)
            cnt[..., y:y + tile.shape[-2], x0:x0 + tile.shape[-1]] += 1
    return out / cnt


@torch.no_grad()
def predict(model, image, scales=(1.0,), flip=False, amp=True):
    """image: (1,3,H,W) normalised tensor on device. Returns (H,W) uint8 class map."""
    h, w = image.shape[-2:]
    prob = 0
    with torch.autocast(device_type=image.device.type, enabled=amp and image.device.type == "cuda"):
        for s in scales:
            x = image if s == 1.0 else F.interpolate(image, scale_factor=s, mode="bilinear", align_corners=False)
            views = [x, x.flip(-1)] if flip else [x]
            for k, v in enumerate(views):
                logit = _forward_sliding(model, v).float()
                if k == 1:
                    logit = logit.flip(-1)
                logit = F.interpolate(logit, size=(h, w), mode="bilinear", align_corners=False)
                prob = prob + logit.softmax(1)
    return prob.argmax(1)[0].byte().cpu().numpy()


def evaluate_names(model, data_root, names, scale=1.0, scales=(1.0,), flip=False, device="cuda", save_dir=None):
    ds = SegDataset(data_root, names, train=False, scale=scale)
    rows = []
    model.eval()
    for i in range(len(ds)):
        item = ds[i]
        pred = predict(model, item["image"][None].to(device), scales=scales, flip=flip)
        gt = item["mask"].numpy().astype(np.uint8)
        if scale != 1.0:
            # Score at native resolution so that all runs share the same ground truth.
            native = cv2.imread(os.path.join(data_root, "masks", item["name"]), cv2.IMREAD_UNCHANGED)
            pred = cv2.resize(pred, native.shape[::-1], interpolation=cv2.INTER_NEAREST)
            gt = native
        rows.append(dict(name=item["name"], **image_metrics(pred, gt)))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, item["name"]), pred)
    return pd.DataFrame(rows)


def train_run(cfg, data_root, split_file, results_root, device="cuda", log=print):
    cfg = {**DEFAULTS, **cfg}
    out_dir = os.path.join(results_root, run_name(cfg))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "final.pt")):
        log(f"[skip] already finished: {run_name(cfg)}")
        return out_dir
    if not acquire_lock(out_dir):
        log(f"[skip] locked by another session: {run_name(cfg)}")
        return None
    try:
        split = load_split(split_file, cfg["split"])
        set_seed(cfg["seed"])
        need_skel = cfg["loss"] == "skelrecall"
        train_ds = SegDataset(data_root, split["train"], train=True, crop=cfg["crop"], scale=cfg["scale"], with_skeleton=need_skel)
        g = torch.Generator()
        g.manual_seed(cfg["seed"])
        loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["num_workers"],
            pin_memory=True, generator=g, persistent_workers=cfg["num_workers"] > 0,
            worker_init_fn=lambda w: (random.seed(cfg["seed"] * 1000 + w), np.random.seed(cfg["seed"] * 1000 + w)),
        )
        model = build_model(cfg["backbone"]).to(device)
        crit = SegLoss(cfg["loss"], lam=cfg["lam"], skel_iters=cfg["skel_iters"])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        use_amp = cfg["amp"] and device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        start = 0
        ckpt = os.path.join(out_dir, "last.pt")
        if os.path.exists(ckpt):
            st = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            scaler.load_state_dict(st["scaler"])
            start = st["iter"]
            random.setstate(st["py_rng"])
            np.random.set_state(st["np_rng"])
            # map_location moves every tensor to the GPU; RNG states must stay on the CPU.
            torch.set_rng_state(st["torch_rng"].cpu())
            if "gen_rng" in st:
                g.set_state(st["gen_rng"].cpu())
            with open(os.path.join(out_dir, "resume_log.txt"), "a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} resumed at iter {start} on {env_info()['gpu']}\n")
            log(f"[resume] {run_name(cfg)} from iter {start}")
        else:
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump({"config": cfg, "env": env_info()}, f, indent=1)
        log_path = os.path.join(out_dir, "train_log.csv")
        new_log = not os.path.exists(log_path)
        logf = open(log_path, "a", newline="")
        writer = csv.writer(logf)
        if new_log:
            writer.writerow(["iter", "lr", "loss", "ce", "dice", "conn", "sec_per_iter", "val_stem_iou", "val_miou"])
        model.train()
        it, t0, data_iter = start, time.time(), iter(loader)
        while it < cfg["iters"]:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            lr = poly_lr(it, cfg)
            for pg in opt.param_groups:
                pg["lr"] = lr
            x, y = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            sk = batch["skeleton"].to(device) if need_skel else None
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
            loss, parts = crit(logits.float(), y, sk)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            it += 1
            val = ["", ""]
            if it % cfg["val_every"] == 0 or it == cfg["iters"]:
                df = evaluate_names(model, data_root, split["val"], scale=cfg["scale"], device=device)
                r = dataset_iou(df)
                val = [round(r["iou_stem"], 4), round(r["miou"], 4)]
                model.train()
                log(f"[val] {run_name(cfg)} iter {it}: stem IoU {val[0]} mIoU {val[1]}")
            if it % 50 == 0 or val[0] != "":
                spi = (time.time() - t0) / max(1, it - start)
                writer.writerow([it, f"{lr:.2e}", f"{loss.item():.4f}", f"{parts['ce']:.4f}", f"{parts['dice']:.4f}",
                                 f"{parts.get('conn', 0):.4f}", f"{spi:.3f}", *val])
                logf.flush()
            if it % 200 == 0:
                spi = (time.time() - t0) / max(1, it - start)
                log(f"{run_name(cfg)} iter {it}/{cfg['iters']} loss {loss.item():.3f} "
                    f"({spi:.2f}s/it, ~{spi * (cfg['iters'] - it) / 60:.0f} min left)")
            if it % cfg["ckpt_every"] == 0 and it < cfg["iters"]:
                _atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), iter=it,
                                  py_rng=random.getstate(), np_rng=np.random.get_state(),
                                  torch_rng=torch.get_rng_state(), gen_rng=g.get_state()), ckpt)
                touch_lock(out_dir)
        logf.close()
        # The final iterate is the reported model (no checkpoint selection on validation).
        _atomic_save(model.state_dict(), os.path.join(out_dir, "final.pt"))
        if os.path.exists(ckpt):
            os.remove(ckpt)
        log(f"[done] {run_name(cfg)}")
        return out_dir
    finally:
        release_lock(out_dir)


def evaluate_run(run_dir, data_root, split_file, subset="test", tag="ss", scales=(1.0,), flip=False,
                 device="cuda", save_preds=False, log=print):
    """Writes metrics_{tag}.csv (per image) and summary_{tag}.json (dataset level) in run_dir."""
    out_csv = os.path.join(run_dir, f"metrics_{tag}.csv")
    pred_dir = os.path.join(run_dir, f"pred_{tag}")
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        current = "metrics_version" in old and (old["metrics_version"] >= METRICS_VERSION).all()
        preds_ok = (not save_preds) or (os.path.isdir(pred_dir) and len(os.listdir(pred_dir)) >= len(old))
        if current and preds_ok:
            log(f"[skip] {run_dir} {tag}")
            return old
        reason = f"metric table older than version {METRICS_VERSION}" if not current else "prediction PNGs missing"
        log(f"[re-eval] {run_dir} {tag}: {reason}")
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)["config"]
    model = build_model(cfg["backbone"], pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "final.pt"), map_location=device))
    names = load_split(split_file, cfg["split"])[subset]
    df = evaluate_names(model, data_root, names, scale=cfg["scale"], scales=scales, flip=flip, device=device,
                        save_dir=pred_dir if save_preds else None)
    df.to_csv(out_csv, index=False)
    summary = dict(dataset_iou(df), subset=subset, tag=tag, scales=list(scales), flip=flip, n_images=len(df))
    with open(os.path.join(run_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    log(f"[eval] {run_dir} {tag}: stem IoU {summary['iou_stem']:.4f} clDice {summary['cldice_stem']:.4f} mIoU {summary['miou']:.4f}")
    return df


# ---------------------------------------------------------------------------
# Decision-threshold control (specified before running; see specifications/ and the manuscript, Section 3.8):
# A scalar offset b is added to the stem logit before argmax.
OFFSETS = [round(-2.0 + 0.5 * i, 1) for i in range(21)]  # -2.0 ... +8.0 (extended from +4.0 before any test result was computed)


@torch.no_grad()
def _native_logits(model, image, amp=True):
    with torch.autocast(device_type=image.device.type, enabled=amp and image.device.type == "cuda"):
        return _forward_sliding(model, image).float()


def evaluate_offset(run_dir, data_root, split_file, device="cuda", offsets=OFFSETS, log=print):
    """Choose the stem-logit offset on the validation set (max dataset-level stem IoU, ties -> smallest |b|),
    then evaluate the test set once at that offset. Writes offset_val.csv, metrics_thr.csv, summary_thr.json."""
    from .data import STEM, IGNORE
    out_csv = os.path.join(run_dir, "metrics_thr.csv")
    if os.path.exists(out_csv):
        log(f"[skip] {run_dir} thr")
        return pd.read_csv(out_csv)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)["config"]
    model = build_model(cfg["backbone"], pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "final.pt"), map_location=device))
    model.eval()
    split = load_split(split_file, cfg["split"])
    # 1) validation: stem IoU for every offset (cheap confusion counts)
    inter = np.zeros(len(offsets)); union = np.zeros(len(offsets))
    ds = SegDataset(data_root, split["val"], train=False, scale=cfg["scale"])
    for i in range(len(ds)):
        it = ds[i]
        lg = _native_logits(model, it["image"][None].to(device))
        gt = it["mask"].to(device); valid = gt != IGNORE; g = (gt == STEM) & valid
        for k, b in enumerate(offsets):
            l2 = lg.clone(); l2[:, STEM] += b
            p = (l2.argmax(1)[0] == STEM) & valid
            inter[k] += (p & g).sum().item(); union[k] += (p | g).sum().item()
    iou = inter / np.maximum(union, 1)
    best = max(range(len(offsets)), key=lambda k: (round(iou[k], 6), -abs(offsets[k])))
    b = offsets[best]
    pd.DataFrame({"offset": offsets, "val_stem_iou": iou}).to_csv(os.path.join(run_dir, "offset_val.csv"), index=False)
    # 2) test once at the chosen offset
    ds = SegDataset(data_root, split["test"], train=False, scale=cfg["scale"])
    rows = []
    for i in range(len(ds)):
        it = ds[i]
        lg = _native_logits(model, it["image"][None].to(device)); lg[:, STEM] += b
        pred = lg.argmax(1)[0].byte().cpu().numpy()
        gt = it["mask"].numpy().astype(np.uint8)
        if cfg["scale"] != 1.0:
            native = cv2.imread(os.path.join(data_root, "masks", it["name"]), cv2.IMREAD_UNCHANGED)
            pred = cv2.resize(pred, native.shape[::-1], interpolation=cv2.INTER_NEAREST); gt = native
        rows.append(dict(name=it["name"], **image_metrics(pred, gt)))
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    summary = dict(dataset_iou(df), subset="test", tag="thr", offset=b, val_stem_iou_at_offset=float(iou[best]),
                   val_stem_iou_at_zero=float(iou[offsets.index(0.0)]), n_images=len(df))
    with open(os.path.join(run_dir, "summary_thr.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    log(f"[thr] {run_dir}: offset {b:+.1f}  val stem IoU {iou[offsets.index(0.0)]:.4f}->{iou[best]:.4f}  test stem IoU {summary['iou_stem']:.4f}")
    return df


@torch.no_grad()
def offset_sweep(run_dir, data_root, split_file, subset="test", device="cuda", offsets=OFFSETS, log=print):
    """Per image and per stem-logit offset: class confusion counts and stem trait values.
    Writes sweep_{subset}.csv (long format) for offline threshold selection (validation or few target labels)."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from .data import STEM, IGNORE
    from .metrics import CLASSES, EIGHT
    out_csv = os.path.join(run_dir, f"sweep_{subset}.csv")
    if os.path.exists(out_csv):
        log(f"[skip] {run_dir} sweep_{subset}")
        return pd.read_csv(out_csv)
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)["config"]
    assert cfg["scale"] == 1.0, "offset sweep is defined for native-resolution runs"
    model = build_model(cfg["backbone"], pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "final.pt"), map_location=device))
    model.eval()
    split = load_split(split_file, cfg["split"])
    ds = SegDataset(data_root, split[subset], train=False, scale=1.0)
    rows = []
    for i in range(len(ds)):
        it = ds[i]
        lg = _native_logits(model, it["image"][None].to(device))
        gt = it["mask"].numpy().astype(np.uint8)
        valid = gt != IGNORE
        g_stem = (gt == STEM) & valid
        ref = dict(gt_stem_px=int(g_stem.sum()), skel_gt_px=int(skeletonize(g_stem).sum()),
                   stem_cc_gt=int(ndimage.label(g_stem, structure=EIGHT)[1]), valid_px=int(valid.sum()))
        for b in offsets:
            l2 = lg.clone(); l2[:, STEM] += b
            pred = l2.argmax(1)[0].byte().cpu().numpy()
            r = dict(name=it["name"], offset=b, **ref)
            for c, name in enumerate(CLASSES):
                p, g = (pred == c) & valid, (gt == c) & valid
                r[f"inter_{name}"] = int((p & g).sum()); r[f"union_{name}"] = int((p | g).sum())
            p = (pred == STEM) & valid
            r["pred_stem_px"] = int(p.sum())
            r["skel_pred_px"] = int(skeletonize(p).sum()) if r["pred_stem_px"] else 0
            r["stem_cc_pred"] = int(ndimage.label(p, structure=EIGHT)[1])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log(f"[sweep] {run_dir} {subset}: {df.name.nunique()} images x {len(offsets)} offsets")
    return df
