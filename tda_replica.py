"""
tda_replica.py - Clean re-implementation of TDA
("Efficient Test-Time Adaptation of Vision-Language Models", Karmanov et al., CVPR 2024).

This is THE file to edit for method extensions. You normally do not call it by hand:
`experiments.py` runs it for every experiment declared in `experiments.yaml`.

It is run from inside the official repository (https://github.com/kdiAAA/TDA, cloned into
TDA/ by the setup) with that folder on PYTHONPATH: it reuses the repo's `clip/` package and
the official data loaders, so the stream of test images (preprocessing, AugMix views, and with
--order official also the order) is exactly the one the authors used. Everything that is the
*method* (CLIP logits, view selection, entropies, positive and negative caches, adapted logits)
is re-implemented here from scratch.

Fidelity notes (behaviour of the official code, reproduced on purpose):
  * Test-time batch size is 1. With --order official the stream is the official shuffled
    loader, seeded once (random.seed / torch.manual_seed) AFTER loading CLIP. With several
    datasets in one run, the order of dataset k then depends on datasets 1..k-1, so the
    benchmark sequence of the official scripts must be run to get the authors' streams.
  * ImageNet-A/V/R/S use 1 original + 63 AugMix views; the 10% lowest-entropy views
    (int(64 * 0.1) = 6) are averaged. The averaged feature is NOT re-normalised.
    ImageNet ("I") and the 10 cross-domain datasets use a single view.
  * The entropy stored in the caches is in nats, but the gate of the negative cache
    divides it by log2(N) (not ln N). Thresholds in configs/*.yaml assume this.
  * Negative cache: values are the binary mask p_l < p < 1.0 (p_l = 0.03) of the
    stored probability map, and the cache logits are SUBTRACTED. (Eq. 4 + Eq. 6 in
    the paper contain a double minus sign; the code is the reference.)
  * Everything runs in the CLIP model's dtype (fp16 on GPU), like the original.

Where to plug an extension:
  * new stream order          -> make_stream()
  * new cache / read-out rule -> DynamicCache, or a new class used in run_tda()
  * new variant of the method -> add a name to MODES and handle it in run_tda()
  * any hyper-parameter       -> no code: `--set positive.alpha=3` (or a grid in experiments.yaml)

Output: one JSON line per dataset appended to --out, and (with --trace-dir) a per-image trace
<trace-dir>/<run name>.csv.gz used by notebooks/analysis.ipynb. Trace columns:
  step, label, clip_pred, pred, clip_correct, correct, entropy (normalised by log2 N),
  pos_update / neg_update ("add", "replace" or ""), pos_size, neg_size, pos_classes
"""

import argparse
import csv
import gzip
import json
import math
import os
import random
import re
import subprocess
import time
import zlib

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import clip  # the repo's own copy of OpenAI CLIP (identical API)
from utils import build_test_data_loader, get_config_file  # official data pipeline

MODES = ["tda", "pos", "neg", "clip"]          # full TDA / positive only / negative only / zero-shot CLIP
ORDERS = ["official", "random", "sorted"]


# --------------------------------------------------------------------------------------
# CLIP side: zero-shot classifier and per-sample logits
# --------------------------------------------------------------------------------------
@torch.no_grad()
def build_text_classifier(classnames, templates, model, device):
    """W_c: one L2-normalised text embedding per class (prompt ensemble), shape [C, N]."""
    weights = []
    for name in classnames:
        name = name.replace("_", " ")
        tokens = clip.tokenize([t.format(name) for t in templates]).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0)
        weights.append(emb / emb.norm())
    return torch.stack(weights, dim=1)


def top_class(logits):
    """Index of the highest logit. topk (not argmax) so that ties break as in the original."""
    return logits.topk(1, 1, True, True)[1].t()


def entropy_of_logits(logits):
    """Shannon entropy (nats) of softmax(logits), one value per row."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def entropy_of_mean_prob(logits):
    """Entropy (nats) of the averaged probability over views (TPT marginal entropy)."""
    logp = logits - logits.logsumexp(dim=-1, keepdim=True)
    avg_logp = logp.logsumexp(dim=0) - math.log(logp.shape[0])
    avg_logp = torch.clamp(avg_logp, min=torch.finfo(avg_logp.dtype).min)
    return -(avg_logp * torch.exp(avg_logp)).sum(dim=-1)


@torch.no_grad()
def clip_forward(images, model, text_weights, device, select_ratio=0.1):
    """
    Returns (feature f [1, C], CLIP logits [1, N], entropy H (nats), prob map [1, N], pseudo label).
    With multiple views, keeps the `select_ratio` lowest-entropy views (confidence selection).
    """
    images = torch.cat(images, dim=0) if isinstance(images, list) else images
    images = images.to(device)
    feats = model.encode_image(images)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    logits = 100.0 * feats @ text_weights

    if feats.size(0) > 1:  # augmented views (ImageNet-A/V/R/S)
        ent = entropy_of_logits(logits)
        keep = torch.argsort(ent, descending=False)[: int(ent.size(0) * select_ratio)]
        sel_logits = logits[keep]
        feat = feats[keep].mean(0).unsqueeze(0)          # not re-normalised (as in the original)
        mean_logits = sel_logits.mean(0).unsqueeze(0)
        ent = entropy_of_mean_prob(sel_logits)
        prob = sel_logits.softmax(1).mean(0).unsqueeze(0)
        pred = int(top_class(mean_logits))
        return feat, mean_logits, ent, prob, pred

    ent = entropy_of_logits(logits)
    prob = logits.softmax(1)
    pred = int(top_class(logits)[0])
    return feats, logits, ent, prob, pred


# --------------------------------------------------------------------------------------
# TDA: dynamic key-value cache
# --------------------------------------------------------------------------------------
class DynamicCache:
    """
    One priority queue per pseudo-class, ordered by entropy (lowest first), with a
    maximum of `capacity` entries per class ("shot capacity", k in the paper).
    Entry = (key feature [1, C], entropy, value), where value is the class index for
    the positive cache or the probability map [1, N] for the negative cache.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.queues = {}

    def __bool__(self):
        return bool(self.queues)

    def __len__(self):
        return sum(len(q) for q in self.queues.values())

    def update(self, pred, feat, entropy, value):
        """Insert a sample. Returns "add", "replace" or "" (sample discarded), for the trace."""
        entry = (feat, entropy, value)
        queue = self.queues.get(pred)
        if queue is None:                       # first sample predicted as this class
            self.queues[pred] = [entry]
            return "add"
        action = ""
        if len(queue) < self.capacity:          # condition 1: still room -> add
            queue.append(entry)
            action = "add"
        elif entropy < queue[-1][1]:            # condition 2: replace the most uncertain
            queue[-1] = entry
            action = "replace"
        queue.sort(key=lambda e: e[1])          # stable sort, lowest entropy first
        return action

    def _keys_values(self):
        keys, values = [], []
        for cls in sorted(self.queues):         # same key order as the official code
            for feat, _, value in self.queues[cls]:
                keys.append(feat)
                values.append(value)
        return torch.cat(keys, dim=0).t(), values

    @staticmethod
    def _adapt(feat, keys, values, alpha, beta):
        """Tip-Adapter style read-out: alpha * exp(-beta (1 - f K^T)) V."""
        affinity = feat @ keys
        return alpha * (((-1) * (beta - beta * affinity)).exp() @ values)

    def positive_logits(self, feat, num_classes, alpha, beta):
        keys, labels = self._keys_values()
        one_hot = F.one_hot(torch.tensor(labels, dtype=torch.int64), num_classes=num_classes)
        one_hot = one_hot.to(device=feat.device, dtype=feat.dtype)
        return self._adapt(feat, keys, one_hot, alpha, beta)

    def negative_logits(self, feat, alpha, beta, p_low, p_high):
        keys, probs = self._keys_values()
        probs = torch.cat(probs, dim=0)
        mask = ((probs > p_low) & (probs < p_high)).to(device=feat.device, dtype=feat.dtype)
        return self._adapt(feat, keys, mask, alpha, beta)


# --------------------------------------------------------------------------------------
# Test stream
# --------------------------------------------------------------------------------------
def make_stream(loader, order, seed, dataset_name):
    """
    The order in which test images arrive.
      official : the official shuffled loader (global RNG, seeded once at start-up) -> paper numbers
      random   : a fresh shuffle that depends only on (seed, dataset), so a dataset can be run
                 alone or in any sequence and still get the same stream
      sorted   : images grouped by class (class 0 first, random order inside a class):
                 a non-i.i.d. stream, the hardest case for an online cache
    Add new orders here (e.g. class blocks, increasing difficulty) and list them in ORDERS.
    """
    if order == "official":
        return loader
    ds = loader.dataset
    labels = list(ds.targets) if hasattr(ds, "targets") else [d.label for d in ds.data_source]
    g = torch.Generator()
    g.manual_seed(seed * 1000003 + zlib.crc32(dataset_name.encode()))
    idx = torch.randperm(len(labels), generator=g).tolist()
    if order == "sorted":
        idx.sort(key=lambda i: labels[i])        # stable: keeps the random order inside a class
    elif order != "random":
        raise ValueError(f"unknown order {order!r}; choose from {ORDERS}")
    return torch.utils.data.DataLoader(ds, batch_size=1, sampler=idx, num_workers=loader.num_workers,
                                       pin_memory=loader.pin_memory)


# --------------------------------------------------------------------------------------
# Test-time adaptation loop
# --------------------------------------------------------------------------------------
def top1(logits, target):
    pred = top_class(logits)
    return 100.0 * float(pred.eq(target.view(1, -1).expand_as(pred)).float().sum()) / target.shape[0]


@torch.no_grad()
def run_tda(cfg, loader, model, text_weights, device, verify=False, limit=None, trace=None):
    """Runs the stream once. `trace`, if a list, receives one dict per test image."""
    pos_cfg, neg_cfg = cfg["positive"], cfg["negative"]
    num_classes = text_weights.size(1)
    pos_cache = DynamicCache(pos_cfg["shot_capacity"])
    neg_cache = DynamicCache(neg_cfg["shot_capacity"])
    accs, clip_accs = [], []

    if verify:  # official implementation running side by side on the same samples
        import tda_runner as official
        from utils import get_clip_logits as official_clip_logits, get_entropy as official_entropy
        off_pos, off_neg = {}, {}
        mismatches, max_diff = 0, 0.0

    start = time.time()
    for i, (images, target) in enumerate(tqdm(loader, desc="TDA stream", mininterval=30)):
        if limit is not None and i >= limit:
            break
        target = target.to(device)
        feat, clip_logits, ent, prob, pred = clip_forward(images, model, text_weights, device)
        norm_ent = float(ent / math.log2(num_classes))

        pos_action = neg_action = ""
        if pos_cfg["enabled"]:
            pos_action = pos_cache.update(pred, feat, ent, pred)
        if neg_cfg["enabled"] and neg_cfg["entropy_threshold"]["lower"] < norm_ent < neg_cfg["entropy_threshold"]["upper"]:
            neg_action = neg_cache.update(pred, feat, ent, prob)

        logits = clip_logits.clone()
        if pos_cfg["enabled"] and pos_cache:
            logits += pos_cache.positive_logits(feat, num_classes, pos_cfg["alpha"], pos_cfg["beta"])
        if neg_cfg["enabled"] and neg_cache:
            logits -= neg_cache.negative_logits(
                feat, neg_cfg["alpha"], neg_cfg["beta"],
                neg_cfg["mask_threshold"]["lower"], neg_cfg["mask_threshold"]["upper"])

        acc, clip_acc = top1(logits, target), top1(clip_logits, target)
        accs.append(acc)
        clip_accs.append(clip_acc)
        if trace is not None:
            trace.append({"step": i, "label": int(target), "clip_pred": pred, "pred": int(top_class(logits)),
                          "clip_correct": int(clip_acc > 0), "correct": int(acc > 0),
                          "entropy": round(norm_ent, 4), "pos_update": pos_action, "neg_update": neg_action,
                          "pos_size": len(pos_cache), "neg_size": len(neg_cache),
                          "pos_classes": len(pos_cache.queues)})

        if verify:
            o_feat, o_clip, o_loss, o_prob, o_pred = official_clip_logits(images, model, text_weights)
            o_ent = official_entropy(o_loss, text_weights)
            if pos_cfg["enabled"]:
                official.update_cache(off_pos, o_pred, [o_feat, o_loss], pos_cfg["shot_capacity"])
            if neg_cfg["enabled"] and neg_cfg["entropy_threshold"]["lower"] < o_ent < neg_cfg["entropy_threshold"]["upper"]:
                official.update_cache(off_neg, o_pred, [o_feat, o_loss, o_prob], neg_cfg["shot_capacity"], True)
            o_logits = o_clip.clone()
            if pos_cfg["enabled"] and off_pos:
                o_logits += official.compute_cache_logits(o_feat, off_pos, pos_cfg["alpha"], pos_cfg["beta"], text_weights)
            if neg_cfg["enabled"] and off_neg:
                o_logits -= official.compute_cache_logits(
                    o_feat, off_neg, neg_cfg["alpha"], neg_cfg["beta"], text_weights,
                    (neg_cfg["mask_threshold"]["lower"], neg_cfg["mask_threshold"]["upper"]))
            max_diff = max(max_diff, float((o_logits.float() - logits.float()).abs().max()))
            mismatches += int(int(top_class(o_logits)) != int(top_class(logits)))

    out = {
        "acc": sum(accs) / len(accs),
        "clip_acc": sum(clip_accs) / len(clip_accs),
        "n": len(accs),
        "seconds": round(time.time() - start, 1),
        "pos_cache_size": len(pos_cache),
        "neg_cache_size": len(neg_cache),
    }
    if verify:
        out["verify_pred_mismatches"] = mismatches
        out["verify_max_abs_logit_diff"] = max_diff
    return out


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def parse_overrides(items):
    """['positive.alpha=3', 'negative.shot_capacity=1'] -> {'positive.alpha': 3, ...} (values typed by YAML)."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def apply_overrides(cfg, overrides):
    for key, value in overrides.items():
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            if p not in node:
                raise SystemExit(f"--set {key}: '{p}' is not in the config ({list(node)})")
            node = node[p]
        if parts[-1] not in node:
            raise SystemExit(f"--set {key}: '{parts[-1]}' is not in the config ({list(node)})")
        node[parts[-1]] = value


def run_name(experiment, dataset, backbone, mode, seed, order, overrides):
    """Readable, file-system-safe identifier of one run (used for the trace file)."""
    parts = [experiment, dataset, backbone, mode, f"seed{seed}", order]
    parts += [f"{k}={v}" for k, v in sorted(overrides.items())]
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", "__".join(parts))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_trace(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Re-implementation of TDA (CVPR 2024)")
    p.add_argument("--config", default="configs", help="folder with the official yaml configs")
    p.add_argument("--datasets", required=True, help="slash-separated, e.g. I/A/V/R/S or dtd/eurosat")
    p.add_argument("--data-root", default="./dataset/")
    p.add_argument("--backbone", required=True, choices=["RN50", "ViT-B/16"])
    p.add_argument("--mode", default="tda", choices=MODES,
                   help="full TDA, positive cache only, negative cache only, CLIP only")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--order", default="official", choices=ORDERS, help="order of the test stream")
    p.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                   help="override a config value, e.g. --set positive.shot_capacity=1 (repeatable)")
    p.add_argument("--experiment", default="manual", help="experiment name stored in the results")
    p.add_argument("--verify", action="store_true", help="also run the official functions and compare")
    p.add_argument("--limit", type=int, default=None, help="only the first N stream samples (smoke test)")
    p.add_argument("--out", default="results.jsonl", help="append one JSON line per dataset")
    p.add_argument("--trace-dir", default=None, help="save a per-image trace (csv.gz) in this folder")
    args = p.parse_args()
    overrides = parse_overrides(args.overrides)

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required (the official data/CLIP helpers call .cuda()).")
    device = "cuda"

    model, preprocess = clip.load(args.backbone)
    model.eval()
    random.seed(args.seed)          # same seeding point as tda_runner.py
    torch.manual_seed(args.seed)
    commit = git_commit()

    for name in args.datasets.split("/"):
        cfg = get_config_file(args.config, name)
        apply_overrides(cfg, overrides)
        cfg["positive"]["enabled"] &= args.mode in ("tda", "pos")
        cfg["negative"]["enabled"] &= args.mode in ("tda", "neg")
        print(f"\n=== {name} | {args.backbone} | mode={args.mode} seed={args.seed} order={args.order} "
              f"overrides={overrides} | cfg={cfg}", flush=True)

        loader, classnames, templates = build_test_data_loader(name, args.data_root, preprocess)
        loader = make_stream(loader, args.order, args.seed, name)
        text_weights = build_text_classifier(classnames, templates, model, device)
        trace = [] if args.trace_dir else None
        res = run_tda(cfg, loader, model, text_weights, device, verify=args.verify, limit=args.limit, trace=trace)

        rname = run_name(args.experiment, name, args.backbone, args.mode, args.seed, args.order, overrides)
        if trace:
            write_trace(os.path.join(args.trace_dir, rname + ".csv.gz"), trace)
        res.update({
            "experiment": args.experiment, "impl": "replica", "dataset": name, "backbone": args.backbone,
            "mode": args.mode, "seed": args.seed, "order": args.order, "overrides": overrides,
            "sequence": args.datasets, "limit": args.limit, "trace": (rname + ".csv.gz") if trace else None,
            "cfg": cfg, "repo_commit": commit, "time": time.strftime("%Y-%m-%d %H:%M"),
        })
        print(json.dumps({k: v for k, v in res.items() if k != "cfg"}), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
