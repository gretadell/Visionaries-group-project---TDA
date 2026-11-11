"""
context_cache.py - TDA with CONTEXT SUBTRACTION in the cache (runs on saved features, like streams.py).

Why: our results suggest that TDA's cache finds neighbours by background/context rather than by
object (S2 helps, the gain grows with visible background, and on CORe50 350 + RN50 TDA falls below
CLIP on shuffled streams). Here we estimate the current context from the recent stream and subtract
it before the cache is written and read:

    f        = normalised CLIP image feature (as in TDA)
    c_t      = context estimate from the frames seen BEFORE t (no labels, no future frames)
    f_cache  = normalise(f - strength * c_t)      <- used as cache key AND as cache query
    CLIP's own zero-shot logits are unchanged (they still use f); only the cache read-out changes.

Two ways of estimating the context (both online):
  ctx-w<W>   mean of the last W frames (W = "all": running mean of the whole stream so far).
  ctxx-w<W>  the same, but EXCLUDING frames whose CLIP pseudo-label equals the current frame's
             pseudo-label. Reason: in a temporally ordered stream (S3) the last frames are mostly
             the SAME object, so a plain recent mean estimates the object rather than the background
             and subtracting it would remove the object; excluding the current pseudo-class keeps
             the estimate on the other objects of the same context.
Expected (hypotheses to test):
  * S2 (one session at a time): the recent mean is the session context -> should help most.
  * S3: plain ctx with short W may hurt (object leaks into the context); ctxx should not.
  * S0 (shuffled): the recent mean is close to the global mean -> acts as global centring of
    the features, not a session-specific correction; small or no effect.

Results go to the same RESULTS_DIR/feature_runs.jsonl as streams.py, so `python streams.py summary`
and `python analyze_traces.py` show them next to tda / m2 / oracle. Nothing in tda_replica.py or
streams.py is changed: this file only reuses them.

  python context_cache.py --dataset core50 --backbone RN50 --cfg-like caltech101 --sessions 3,7,10 \
                          --order s0,s2,s3 --mode ctx,ctxx --window 50,300,all --repeats 3
  python streams.py summary --dataset core50
Options: --strength 1.0 (how much of the context to subtract), --trace (per-frame trace, as streams.py).
Choose W (and strength) on the CORe50 NON-test sessions (--sessions 1,2,4,5,6,8,9,11 --every 5), then freeze.
"""

import argparse
import json
import math
import os
import time

import streams                                  # sets sys.path for the official repo and tda_replica
import tda_replica as tda
import torch
from tqdm import tqdm

RESULTS_DIR = streams.RESULTS_DIR


class Context:
    """Online context estimate from past frames; optionally excluding one pseudo-class."""

    def __init__(self, window, num_classes, dim, device):
        self.window = window                     # int, or None for the whole stream so far
        self.buf = []                            # (feat [dim] fp32, pseudo-label), only if windowed
        self.total = torch.zeros(dim, device=device)
        self.n = 0
        self.per_class = torch.zeros(num_classes, dim, device=device)
        self.per_class_n = torch.zeros(num_classes, device=device)

    def estimate(self, exclude=None):
        n, s = self.n, self.total
        if exclude is not None:
            n, s = n - float(self.per_class_n[exclude]), s - self.per_class[exclude]
        if n <= 0:
            return None
        return s / n

    def add(self, feat, label):
        f = feat.float().view(-1)
        self.total += f
        self.n += 1
        self.per_class[label] += f
        self.per_class_n[label] += 1
        if self.window is not None:
            self.buf.append((f, label))
            if len(self.buf) > self.window:      # drop the oldest frame from the window
                old, old_label = self.buf.pop(0)
                self.total -= old
                self.n -= 1
                self.per_class[old_label] -= old
                self.per_class_n[old_label] -= 1


@torch.no_grad()
def run_context_tda(cfg, data, idx, device, window, exclude_class, strength, limit=None, trace=None):
    """Same loop as tda_replica.run_tda, but the cache uses context-subtracted features."""
    feats = data["feats"].to(device)
    text_weights = data["text_weights"].to(device)
    if device == "cpu":
        feats, text_weights = feats.float(), text_weights.float()
    pos_cfg, neg_cfg = cfg["positive"], cfg["negative"]
    num_classes = text_weights.size(1)
    pos_cache = tda.DynamicCache(pos_cfg["shot_capacity"])
    neg_cache = tda.DynamicCache(neg_cfg["shot_capacity"])
    ctx = Context(window, num_classes, feats.size(1), device)
    hook = streams.cache_stats_hook(data, idx) if trace is not None else None
    labels = data["labels"]
    accs, clip_accs = [], []
    start = time.time()
    order = idx[:limit] if limit else idx
    for i, r in enumerate(tqdm(order, desc="context TDA", mininterval=30)):
        target = labels[r:r + 1].to(device)
        feat, clip_logits, ent, prob, pred = tda.clip_forward(feats[r:r + 1], streams.SavedFeatureEncoder(),
                                                              text_weights, device)
        norm_ent = float(ent / math.log2(num_classes))

        c = ctx.estimate(exclude=pred if exclude_class else None)
        if c is None:
            key = feat
        else:
            key = feat.float() - strength * c.view(1, -1)
            key = (key / key.norm(dim=-1, keepdim=True).clamp(min=1e-6)).to(feat.dtype)
        ctx.add(feat, pred)                      # the current frame joins the context AFTER it is used

        pos_action = neg_action = ""
        if pos_cfg["enabled"]:
            pos_action = pos_cache.update(pred, key, ent, pred, tag=i)
        if neg_cfg["enabled"] and neg_cfg["entropy_threshold"]["lower"] < norm_ent < neg_cfg["entropy_threshold"]["upper"]:
            neg_action = neg_cache.update(pred, key, ent, prob, tag=i)

        logits = clip_logits.clone()
        if pos_cfg["enabled"] and pos_cache:
            logits += pos_cache.positive_logits(key, num_classes, pos_cfg["alpha"], pos_cfg["beta"])
        if neg_cfg["enabled"] and neg_cache:
            logits -= neg_cache.negative_logits(key, neg_cfg["alpha"], neg_cfg["beta"],
                                                neg_cfg["mask_threshold"]["lower"], neg_cfg["mask_threshold"]["upper"])

        acc, clip_acc = tda.top1(logits, target), tda.top1(clip_logits, target)
        accs.append(acc)
        clip_accs.append(clip_acc)
        if trace is not None:
            trace.append({"step": i, "label": int(target), "clip_pred": pred, "pred": int(tda.top_class(logits)),
                          "clip_correct": int(clip_acc > 0), "correct": int(acc > 0),
                          "entropy": round(norm_ent, 4), "pos_update": pos_action, "neg_update": neg_action,
                          "pos_size": len(pos_cache), "neg_size": len(neg_cache),
                          "pos_classes": len(pos_cache.queues)})
            hook(i, trace[-1], pos_cache, neg_cache)
    return {"acc": sum(accs) / len(accs), "clip_acc": sum(clip_accs) / len(clip_accs), "n": len(accs),
            "seconds": round(time.time() - start, 1), "pos_cache_size": len(pos_cache),
            "neg_cache_size": len(neg_cache)}


def main():
    p = argparse.ArgumentParser(description="TDA with context subtraction in the cache (saved features)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--backbone", required=True, choices=["RN50", "ViT-B/16"])
    p.add_argument("--mode", default="ctx,ctxx", help="ctx (plain recent mean) and/or ctxx (excluding own pseudo-class)")
    p.add_argument("--window", default="50,300,all", help="frames in the context window; 'all' = whole stream so far")
    p.add_argument("--strength", type=float, default=1.0, help="fraction of the context subtracted (default 1)")
    p.add_argument("--order", default="s0,s2,s3")
    p.add_argument("--rho", default="0.9", help="s1 only")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cfg-like", default=None)
    p.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    p.add_argument("--sessions", default=None)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = streams.split_list(args.sessions, int) if args.sessions else None
    data = streams.select_rows(streams.load_features(args.dataset, args.backbone), sessions, args.every)
    base = data.get("dataset", args.dataset)
    if base in ("core50", "icwt", "stream51") and not args.cfg_like:
        raise SystemExit(f"{base} has no official config: pass e.g. --cfg-like caltech101")
    if base.startswith("ucf101") and not args.cfg_like:
        args.cfg_like = "ucf101"
    os.chdir(streams.REPO_DIR)
    cfg = streams.make_cfg(args, "tda")
    overrides = tda.parse_overrides(args.overrides)
    if args.strength != 1.0:
        overrides = dict(overrides, **{"context.strength": args.strength})   # keeps runs apart in the summary
    out_file = os.path.join(RESULTS_DIR, "feature_runs.jsonl")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = {r.get("key") for r in streams.read_jsonl(out_file) if r.get("trace") or not args.trace}
    print(f"{args.dataset} {args.backbone}: {len(data['labels'])} frames, device {device}", flush=True)

    for m in streams.split_list(args.mode):
        if m not in ("ctx", "ctxx"):
            raise SystemExit(f"unknown mode {m!r}: use ctx and/or ctxx")
        for w in streams.split_list(args.window):
            window = None if w == "all" else int(w)
            mode = f"{m}-w{w}"
            for order in streams.split_list(args.order):
                for rho in (streams.split_list(args.rho, float) if order == "s1" else [None]):
                    for seed in range(args.seed, args.seed + args.repeats):
                        tag = order + (f"-rho{rho}" if rho is not None else "")
                        key = json.dumps([args.dataset, args.backbone, mode, tag, seed, overrides,
                                          args.cfg_like or args.dataset, args.sessions, args.every, args.limit])
                        if key in done:
                            continue
                        idx = streams.stream_order(data, order, seed, base, rho=rho)
                        trace = [] if args.trace else None
                        res = run_context_tda(cfg, data, idx, device, window, m == "ctxx", args.strength,
                                              limit=args.limit, trace=trace)
                        name = tda.run_name("features", args.dataset, args.backbone, mode, seed, tag, overrides)
                        if trace:
                            meta = data["meta"]
                            for row, i in zip(trace, idx):
                                row["row"] = i
                                for k in ("session", "object", "frame"):
                                    src = k if k in meta else ("video" if k == "object" and "video" in meta else None)
                                    row.setdefault(k, meta[src][i] if src else -1)
                            tda.write_trace(os.path.join(RESULTS_DIR, "feature_traces", name + ".csv.gz"), trace)
                        res.update({"key": key, "impl": "context", "dataset": args.dataset,
                                    "backbone": args.backbone, "mode": mode, "order": order, "rho": rho,
                                    "seed": seed, "overrides": overrides, "cfg_like": args.cfg_like or args.dataset,
                                    "sessions": args.sessions, "every": args.every, "window": w,
                                    "strength": args.strength, "trace": (name + ".csv.gz") if trace else None,
                                    "device": device, "time": time.strftime("%Y-%m-%d %H:%M")})
                        print(f"  {mode:10s} {tag:10s} seed {seed:3d}: acc {res['acc']:.2f}  "
                              f"(CLIP {res['clip_acc']:.2f}, {res['seconds']}s)", flush=True)
                        with open(out_file, "a") as f:
                            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
