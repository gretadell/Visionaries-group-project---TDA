"""
features.py - compute CLIP image features ONCE, then run TDA on them in seconds.

Why: CLIP is frozen in TDA, so the image feature of a test image never changes. The slow part
(the GPU reading images) is done once by `extract`; every later experiment (stream orders,
50 repetitions, method variants) only reorders rows of a saved table and runs the cache logic.

Three commands (run from the TDA_replication folder; paths come from the same environment
variables as experiments.py: REPO_DIR, DATA_DIR, RESULTS_DIR):

  python features.py extract --dataset dtd    --backbone RN50
  python features.py extract --dataset core50 --backbone RN50 --root /path/to/core50_350x350
      -> RESULTS_DIR/features/<dataset>__<backbone>.pt

  python features.py check   --dataset dtd    --backbone RN50
      -> week-1 gate: TDA on images (tda_replica.py) vs TDA on saved features, same stream.
         PASS = 0 different predictions. (tda_replica.py already equals the official code.)

  python features.py run     --dataset core50 --backbone RN50 --order random --seed 1
      -> one TDA run on saved features; one JSON line appended to RESULTS_DIR/feature_runs.jsonl

What is saved in the .pt file (a dict):
  feats         [N, C] RAW output of CLIP's image encoder (model dtype, not normalised), so that
                tda_replica.clip_forward() computes exactly what it computes from the images
  labels        [N] class index
  meta          per-image columns: path, and for CORe50 also session, object, frame
  classnames, templates, text_weights [C, num_classes]  (the zero-shot classifier)
  rows are in a FIXED order (dataset index order; for CORe50 sorted by session, object, frame)

The method itself is NOT copied here: `run` and `check` call tda_replica.run_tda() with a stand-in
"model" whose image encoder returns the saved feature. Any change to the method in
tda_replica.py (e.g. M2) is therefore used automatically on features too.
Only single-view datasets are supported (cross-domain datasets and CORe50), not ImageNet-A/V/R/S,
whose 64 AugMix views are random at every run.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import zlib

TDA_HOME = os.environ.get("TDA_HOME", os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.environ.get("REPO_DIR", os.path.join(TDA_HOME, "TDA"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/tda_data"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(TDA_HOME, "results"))
FEATURE_DIR = os.path.join(RESULTS_DIR, "features")
sys.path[:0] = [REPO_DIR, TDA_HOME]       # official repo (clip/, utils.py) + tda_replica.py

import torch
from PIL import Image
from tqdm import tqdm

import clip
import tda_replica as tda
from utils import build_test_data_loader, get_config_file

# CORe50: 50 objects, 5 per category (o1-o5 plug adapters, o6-o10 mobile phones, ...).
# Classification is done on the 10 categories: CLIP cannot tell object instances apart by name.
CORE50_CLASSES = ["plug adapter", "mobile phone", "scissors", "light bulb", "can",
                  "glasses", "ball", "marker", "cup", "remote control"]
CORE50_TEMPLATES = ["a photo of a {}."]
CORE50_TEST_SESSIONS = [3, 7, 10]         # official test sessions of the CORe50 benchmark
ORDERS = ["stored", "random", "sorted"]   # S2/S3 orders for CORe50 are added in the next step


def feature_path(dataset, backbone):
    return os.path.join(FEATURE_DIR, f"{dataset}__{backbone.replace('/', '-')}.pt")


# --------------------------------------------------------------------------------------
# CORe50 images
# --------------------------------------------------------------------------------------
def find_core50_frames(root, sessions, every):
    """All frames under root/.../s<k>/o<j>/*.png, sorted by (session, object, frame)."""
    rows = []
    for dirpath, _, files in os.walk(root):
        obj_dir, ses_dir = os.path.basename(dirpath), os.path.basename(os.path.dirname(dirpath))
        mo, ms = re.fullmatch(r"o(\d+)", obj_dir), re.fullmatch(r"s(\d+)", ses_dir)
        if not (mo and ms):
            continue
        session, obj = int(ms.group(1)), int(mo.group(1))
        if sessions and session not in sessions:
            continue
        for name in files:
            if not name.lower().endswith((".png", ".jpg")):
                continue
            nums = re.findall(r"\d+", os.path.splitext(name)[0])
            if not nums:
                continue
            rows.append({"path": os.path.join(dirpath, name), "session": session, "object": obj,
                         "frame": int(nums[-1])})       # C_01_01_XXX.png -> XXX
    if not rows:
        raise SystemExit(f"No CORe50 frames found under {root} (expected .../s1/o1/*.png).")
    rows.sort(key=lambda r: (r["session"], r["object"], r["frame"]))
    if every > 1:                                       # keep 1 frame out of `every` in each video
        rows = [r for r in rows if r["frame"] % every == 0]
    return rows


class Core50(torch.utils.data.Dataset):
    def __init__(self, rows, preprocess):
        self.rows, self.preprocess = rows, preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = self.preprocess(Image.open(r["path"]).convert("RGB"))
        return img, (r["object"] - 1) // 5             # category index 0..9


# --------------------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------------------
@torch.no_grad()
def extract(args):
    model, preprocess = clip.load(args.backbone)
    model.eval()
    if args.dataset == "core50":
        sessions = [int(s) for s in args.sessions.split(",")] if args.sessions else None
        rows = find_core50_frames(args.root or os.path.join(DATA_DIR, "core50"), sessions, args.every)
        dataset, classnames, templates = Core50(rows, preprocess), CORE50_CLASSES, CORE50_TEMPLATES
        meta = {k: [r[k] for r in rows] for k in ("path", "session", "object", "frame")}
    else:
        loader, classnames, templates = build_test_data_loader(args.dataset, DATA_DIR + "/", preprocess)
        dataset = loader.dataset
        meta = {"path": [getattr(d, "impath", "") for d in getattr(dataset, "data_source", [])]}

    # shuffle=False: rows are saved in dataset index order (the index space used by make_stream)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
    feats, labels, start = [], [], time.time()
    for images, target in tqdm(loader, desc=f"features {args.dataset}", mininterval=30):
        if isinstance(images, list):
            if len(images) != 1:
                raise SystemExit("Multi-view dataset (AugMix): features are random at every run, not supported.")
            images = images[0]
        feats.append(model.encode_image(images.cuda()).cpu())
        labels.append(target)
    text_weights = tda.build_text_classifier(classnames, templates, model, "cuda").cpu()

    out = {"dataset": args.dataset, "backbone": args.backbone,
           "feats": torch.cat(feats), "labels": torch.cat(labels).long(), "meta": meta,
           "classnames": classnames, "templates": templates, "text_weights": text_weights,
           "batch_size": args.batch_size, "time": time.strftime("%Y-%m-%d %H:%M")}
    path = feature_path(args.dataset, args.backbone)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    torch.save(out, path)
    print(f"saved {path}: {tuple(out['feats'].shape)} features, {len(classnames)} classes, "
          f"{(time.time() - start) / 60:.1f} min")


# --------------------------------------------------------------------------------------
# TDA on features
# --------------------------------------------------------------------------------------
class SavedFeatureEncoder:
    """Stand-in for CLIP: the "image" handed to it already is the saved encoder output."""

    def encode_image(self, x):
        return x


class FeatureStream:
    """Yields (feature [1, C], label [1]) in the given order, like a batch-size-1 DataLoader."""

    def __init__(self, feats, labels, order_idx):
        self.feats, self.labels, self.idx = feats, labels, order_idx

    def __len__(self):
        return len(self.idx)

    def __iter__(self):
        for i in self.idx:
            yield self.feats[i:i + 1], self.labels[i:i + 1]


def stream_order(labels, order, seed, dataset):
    """Same formula as tda_replica.make_stream, so 'random'/'sorted' give the same stream as on images."""
    n = len(labels)
    if order == "stored":
        return list(range(n))
    g = torch.Generator()
    g.manual_seed(seed * 1000003 + zlib.crc32(dataset.encode()))
    idx = torch.randperm(n, generator=g).tolist()
    if order == "sorted":
        lab = labels.tolist()
        idx.sort(key=lambda i: lab[i])
    elif order != "random":
        raise SystemExit(f"unknown order {order!r}; choose from {ORDERS}")
    return idx


def load_features(dataset, backbone):
    path = feature_path(dataset, backbone)
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found: run `python features.py extract --dataset {dataset} ...` first.")
    return torch.load(path, map_location="cpu")


def make_cfg(args):
    cfg = get_config_file(os.path.join(REPO_DIR, "configs"), args.cfg_like or args.dataset)
    tda.apply_overrides(cfg, tda.parse_overrides(args.overrides))
    cfg["positive"]["enabled"] &= args.mode in ("tda", "pos")
    cfg["negative"]["enabled"] &= args.mode in ("tda", "neg")
    return cfg


def run_on_features(cfg, data, idx, device, limit=None, trace=None):
    feats = data["feats"].to(device)
    text_weights = data["text_weights"].to(device)
    if device == "cpu":                     # fp16 maths is slow/partial on CPU; results differ in the last digits
        feats, text_weights = feats.float(), text_weights.float()
    stream = FeatureStream(feats, data["labels"], idx)
    return tda.run_tda(cfg, stream, SavedFeatureEncoder(), text_weights, device, limit=limit, trace=trace)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_features(args.dataset, args.backbone)
    cfg = make_cfg(args)
    idx = stream_order(data["labels"], args.order, args.seed, args.dataset)
    res = run_on_features(cfg, data, idx, device, limit=args.limit)
    res.update({"impl": "features", "dataset": args.dataset, "backbone": args.backbone, "mode": args.mode,
                "order": args.order, "seed": args.seed, "overrides": tda.parse_overrides(args.overrides),
                "cfg_like": args.cfg_like or args.dataset, "device": device, "time": time.strftime("%Y-%m-%d %H:%M")})
    print(json.dumps(res))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "feature_runs.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


# --------------------------------------------------------------------------------------
# check (week-1 gate)
# --------------------------------------------------------------------------------------
def check(args):
    if args.dataset == "core50":
        raise SystemExit("Run the check on a dataset that tda_replica.py can read from images, e.g. dtd or eurosat.")
    data = load_features(args.dataset, args.backbone)
    cfg = make_cfg(args)

    # 1) TDA on images, with the same random stream (order 'random', tda_replica.make_stream)
    model, preprocess = clip.load(args.backbone)
    model.eval()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    loader, classnames, templates = build_test_data_loader(args.dataset, DATA_DIR + "/", preprocess)
    text_weights = tda.build_text_classifier(classnames, templates, model, "cuda")
    same_text = torch.equal(text_weights.cpu(), data["text_weights"])
    stream = tda.make_stream(loader, "random", args.seed, args.dataset)
    img_trace = []
    img_res = tda.run_tda(cfg, stream, model, text_weights, "cuda", limit=args.limit, trace=img_trace)

    # 2) TDA on saved features, same order
    idx = stream_order(data["labels"], "random", args.seed, args.dataset)
    feat_trace = []
    feat_res = run_on_features(cfg, data, idx, "cuda", limit=args.limit, trace=feat_trace)

    diff_pred = sum(a["pred"] != b["pred"] for a, b in zip(img_trace, feat_trace))
    diff_label = sum(a["label"] != b["label"] for a, b in zip(img_trace, feat_trace))
    ok = same_text and diff_pred == 0 and diff_label == 0 and len(img_trace) == len(feat_trace)
    res = {"dataset": args.dataset, "backbone": args.backbone, "images": len(img_trace),
           "acc_images": img_res["acc"], "acc_features": feat_res["acc"],
           "seconds_images": img_res["seconds"], "seconds_features": feat_res["seconds"],
           "same_text_weights": same_text, "different_labels": diff_label,
           "different_predictions": diff_pred, "result": "PASS" if ok else "FAIL",
           "time": time.strftime("%Y-%m-%d %H:%M")}
    print(json.dumps(res, indent=1))
    print("RESULT:", res["result"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "feature_check.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="CLIP features once, TDA on features")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "check", "run"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", required=True, help="core50, or a cross-domain name (dtd, eurosat, ...)")
        s.add_argument("--backbone", required=True, choices=["RN50", "ViT-B/16"])
        if name == "extract":
            s.add_argument("--root", default=None, help="CORe50 folder containing s1..s11 (default DATA_DIR/core50)")
            s.add_argument("--sessions", default=None, help="CORe50 sessions to keep, e.g. 3,7,10 (default all)")
            s.add_argument("--every", type=int, default=1, help="CORe50: keep 1 frame out of N (default all)")
            s.add_argument("--batch-size", type=int, default=1,
                           help="1 (default) gives features bit-identical to the batch-size-1 TDA stream")
            s.add_argument("--workers", type=int, default=4)
        else:
            s.add_argument("--mode", default="tda", choices=tda.MODES)
            s.add_argument("--seed", type=int, default=1)
            s.add_argument("--cfg-like", default=None,
                           help="use the hyper-parameters of this dataset's official config "
                                "(needed for core50, which has none; e.g. caltech101)")
            s.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
            s.add_argument("--limit", type=int, default=None)
        if name == "run":
            s.add_argument("--order", default="random", choices=ORDERS)
    args = p.parse_args()

    if args.cmd != "run" and not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for extract and check.")
    if args.dataset == "core50" and args.cmd != "extract" and not args.cfg_like:
        raise SystemExit("core50 has no official config: pass e.g. --cfg-like caltech101")
    if getattr(args, "root", None):
        args.root = os.path.abspath(args.root)
    os.chdir(REPO_DIR)                      # same working directory as tda_replica.py runs
    {"extract": extract, "check": check, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
