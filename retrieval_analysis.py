"""
retrieval_analysis.py - WHICH cache entries drive TDA's predictions? (same session vs same class)

Our report argues that on full-frame CORe50 TDA's positive cache retrieves neighbours by shared
background (same session) rather than by object (same class), and that context subtraction (CTXX)
fixes this. So far that is inferred. This script measures it directly, on the saved CLIP features.

At every step it runs TDA (or CTXX, exactly as in context_cache.py) and, for the POSITIVE cache,
looks at the weight with which each cached sample votes for the current frame:

    w_j = exp(-beta * (1 - q . k_j))          (TDA's own affinity; alpha is a common factor)

q is the query and k_j the cached key: raw features for TDA, context-subtracted features for CTXX.
The current frame's own entry is left out (TDA writes it before reading it). With the true label y
and session s of the current frame, and the true label / session / object of each cached sample:

    session_mass   = sum_j w_j [s_j = s]               / sum_j w_j
    class_mass     = sum_j w_j [y_j = y]               / sum_j w_j   (true class of the cached sample)
    trap_mass      = sum_j w_j [s_j = s and y_j != y]  / sum_j w_j   (same scene, wrong object)
    object_mass    = sum_j w_j [same object instance]  / sum_j w_j
    top1_session / top1_class : is the single strongest neighbour from the same session / class?

The same shares WITHOUT weights (plain composition of the cache) are reported next to them:
    session_bias = session_mass - session_share,   class_bias = class_mass - class_share
A positive session_bias means the read-out favours same-session samples beyond what the cache
contains, i.e. retrieval by context. Labels are used ONLY for this measurement, never by the method.
"broken" columns repeat the masses on frames that CLIP got right and the method got wrong.

Usage (from the TDA_replication folder, same environment as streams.py):
  python retrieval_analysis.py run --dataset core50_350 --backbone RN50 --cfg-like caltech101 \
         --sessions 3,7,10 --order s0,s2,s3 --mode tda,ctxx --repeats 3
  python retrieval_analysis.py run --dataset core50 ...        (128 crops, same command)
  python retrieval_analysis.py summary
      -> RESULTS_DIR/retrieval/retrieval_summary.md and .csv (mean over seeds)
Runs already in RESULTS_DIR/retrieval/retrieval_runs.jsonl are skipped, so a command can be relaunched.
Accuracies are printed too: TDA / CTXX must match feature_runs.jsonl for the same seed (sanity check).
CTXX defaults are the frozen ones (--window 300 --strength 0.5).
"""

import argparse
import json
import math
import os
import time

import streams                                  # sets sys.path (official repo, tda_replica)
import tda_replica as tda
import torch
from context_cache import Context
from tqdm import tqdm

OUT_DIR = os.path.join(streams.RESULTS_DIR, "retrieval")
RUNS = os.path.join(OUT_DIR, "retrieval_runs.jsonl")
METRICS = ["session_mass", "session_share", "session_bias", "class_mass", "class_share", "class_bias",
           "trap_mass", "object_mass", "top1_session", "top1_class"]


@torch.no_grad()
def run_with_retrieval(cfg, data, idx, device, mode, window=300, strength=0.5):
    """TDA (mode 'tda') or context subtraction ('ctx'/'ctxx'), plus retrieval statistics per step."""
    feats = data["feats"].to(device)
    text_weights = data["text_weights"].to(device)
    if device == "cpu":
        feats, text_weights = feats.float(), text_weights.float()
    pos_cfg, neg_cfg = cfg["positive"], cfg["negative"]
    num_classes = text_weights.size(1)
    pos_cache = tda.DynamicCache(pos_cfg["shot_capacity"])
    neg_cache = tda.DynamicCache(neg_cfg["shot_capacity"])
    ctx = Context(window, num_classes, feats.size(1), device) if mode != "tda" else None
    labels = data["labels"].tolist()
    meta = data["meta"]
    sess = meta.get("session")
    obj = meta.get("object") or meta.get("video")
    if sess is None:
        raise SystemExit("retrieval analysis needs sessions in the feature metadata (CORe50, iCWT, UCF101)")
    beta = pos_cfg["beta"]

    sums = {k: 0.0 for k in METRICS}
    broken = {k: 0.0 for k in ("session_mass", "class_mass", "trap_mass")}
    n_meas = n_broken = 0
    accs, clip_accs = [], []
    start = time.time()
    for i, r in enumerate(tqdm(idx, desc=f"{mode} retrieval", mininterval=30)):
        target = data["labels"][r:r + 1].to(device)
        feat, clip_logits, ent, prob, pred = tda.clip_forward(feats[r:r + 1], streams.SavedFeatureEncoder(),
                                                              text_weights, device)
        norm_ent = float(ent / math.log2(num_classes))
        key = feat
        if ctx is not None:                      # identical to context_cache.run_context_tda
            c = ctx.estimate(exclude=pred if mode == "ctxx" else None)
            if c is not None:
                key = feat.float() - strength * c.view(1, -1)
                key = (key / key.norm(dim=-1, keepdim=True).clamp(min=1e-6)).to(feat.dtype)
            ctx.add(feat, pred)

        if pos_cfg["enabled"]:
            pos_cache.update(pred, key, ent, pred, tag=i)
        if neg_cfg["enabled"] and neg_cfg["entropy_threshold"]["lower"] < norm_ent < neg_cfg["entropy_threshold"]["upper"]:
            neg_cache.update(pred, key, ent, prob, tag=i)

        logits = clip_logits.clone()
        if pos_cfg["enabled"] and pos_cache:
            logits += pos_cache.positive_logits(key, num_classes, pos_cfg["alpha"], pos_cfg["beta"])
        if neg_cfg["enabled"] and neg_cache:
            logits -= neg_cache.negative_logits(key, neg_cfg["alpha"], neg_cfg["beta"],
                                                neg_cfg["mask_threshold"]["lower"], neg_cfg["mask_threshold"]["upper"])
        acc, clip_acc = tda.top1(logits, target), tda.top1(clip_logits, target)
        accs.append(acc)
        clip_accs.append(clip_acc)

        # ---- retrieval statistics on the positive cache (current frame's own entry excluded)
        entries = [(e[0], idx[e[3]]) for q in pos_cache.queues.values() for e in q if e[3] != i]
        if not entries:
            continue
        keys = torch.cat([e[0] for e in entries], 0).float()
        w = torch.exp(-beta * (1 - (key.float() @ keys.t()).view(-1)))
        w = (w / w.sum()).tolist()
        rows = [j for _, j in entries]
        y, s, o = labels[r], sess[r], (obj[r] if obj else None)
        same_s = [sess[j] == s for j in rows]
        same_y = [labels[j] == y for j in rows]
        same_o = [obj[j] == o for j in rows] if obj else [False] * len(rows)
        n = len(rows)
        m = {
            "session_mass": sum(wj for wj, a in zip(w, same_s) if a),
            "session_share": sum(same_s) / n,
            "class_mass": sum(wj for wj, a in zip(w, same_y) if a),
            "class_share": sum(same_y) / n,
            "trap_mass": sum(wj for wj, a, b in zip(w, same_s, same_y) if a and not b),
            "object_mass": sum(wj for wj, a in zip(w, same_o) if a),
        }
        top = max(range(n), key=lambda k: w[k])
        m["top1_session"], m["top1_class"] = float(same_s[top]), float(same_y[top])
        m["session_bias"] = m["session_mass"] - m["session_share"]
        m["class_bias"] = m["class_mass"] - m["class_share"]
        for k in METRICS:
            sums[k] += m[k]
        n_meas += 1
        if clip_acc > 0 and acc == 0:            # broken by the method: CLIP right, adapted prediction wrong
            for k in broken:
                broken[k] += m[k]
            n_broken += 1

    out = {k: round(100 * sums[k] / max(n_meas, 1), 3) for k in METRICS}
    out.update({f"broken_{k}": round(100 * v / max(n_broken, 1), 3) for k, v in broken.items()})
    out.update({"acc": sum(accs) / len(accs), "clip_acc": sum(clip_accs) / len(clip_accs), "n": len(accs),
                "n_broken": n_broken, "seconds": round(time.time() - start, 1)})
    return out


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = streams.split_list(args.sessions, int) if args.sessions else None
    data = streams.select_rows(streams.load_features(args.dataset, args.backbone), sessions, args.every)
    base = data.get("dataset", args.dataset)     # core50_350 -> core50: same streams as the final runs
    if base in ("core50", "icwt") and not args.cfg_like:
        raise SystemExit(f"{base} has no official config: pass --cfg-like caltech101")
    os.chdir(streams.REPO_DIR)
    cfg = streams.make_cfg(args, "tda")
    os.makedirs(OUT_DIR, exist_ok=True)
    done = {r.get("key") for r in streams.read_jsonl(RUNS)}
    print(f"{args.dataset} {args.backbone}: {len(data['labels'])} frames, device {device}", flush=True)
    for mode in streams.split_list(args.mode):
        if mode not in ("tda", "ctx", "ctxx"):
            raise SystemExit(f"unknown mode {mode!r}: use tda, ctx and/or ctxx")
        for order in streams.split_list(args.order):
            for seed in range(args.seed, args.seed + args.repeats):
                key = json.dumps([args.dataset, args.backbone, mode, order, seed, args.window, args.strength,
                                  args.cfg_like, args.sessions, args.every])
                if key in done:
                    continue
                idx = streams.stream_order(data, order, seed, base)
                res = run_with_retrieval(cfg, data, idx, device, mode, args.window, args.strength)
                res.update({"key": key, "dataset": args.dataset, "backbone": args.backbone, "mode": mode,
                            "order": order, "seed": seed, "window": args.window,
                            "strength": args.strength if mode != "tda" else None, "cfg_like": args.cfg_like,
                            "sessions": args.sessions, "every": args.every, "device": device,
                            "time": time.strftime("%Y-%m-%d %H:%M")})
                print(f"  {mode:5s} {order:4s} seed {seed}: acc {res['acc']:.2f} (CLIP {res['clip_acc']:.2f}) | "
                      f"session mass {res['session_mass']:.1f}% (share {res['session_share']:.1f}%) | "
                      f"class mass {res['class_mass']:.1f}% (share {res['class_share']:.1f}%) | "
                      f"trap {res['trap_mass']:.1f}% | {res['seconds']}s", flush=True)
                with open(RUNS, "a") as f:
                    f.write(json.dumps(res) + "\n")


def summary(args):
    rows = streams.latest_runs(streams.read_jsonl(RUNS))
    if not rows:
        raise SystemExit(f"no runs in {RUNS} yet")
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["backbone"], r["order"], r["mode"]), []).append(r)
    cols = ["acc", "session_share", "session_mass", "session_bias", "class_share", "class_mass", "class_bias",
            "trap_mass", "object_mass", "top1_session", "top1_class", "broken_session_mass", "broken_class_mass",
            "broken_trap_mass"]
    head = "| Dataset | Backbone | Order | Mode | Seeds | " + " | ".join(cols) + " |"
    lines = [head, "|" + "---|" * 5 + "---:|" * len(cols)]
    csv_rows = ["dataset,backbone,order,mode,seeds," + ",".join(cols)]
    for k in sorted(groups):
        g = groups[k]
        vals = [sum(r[c] for r in g) / len(g) for c in cols]
        lines.append(f"| {' | '.join(k)} | {len(g)} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
        csv_rows.append(",".join(list(k) + [str(len(g))] + [f"{v:.3f}" for v in vals]))
    text = "\n".join(lines)
    notes = """
All values in %, mean over seeds. *_share: plain composition of the positive cache; *_mass: the same share
weighted by TDA's affinity with the current frame (what actually votes); *_bias = mass - share.
trap_mass: weight on cached samples from the same session but a different true class.
top1_*: the strongest cached neighbour is from the same session / class. broken_*: frames CLIP got right and
the method got wrong. Chance for session_share in a shuffled stream: 1 / number of sessions (CORe50 test: 33%).
"""
    print(text + "\n" + notes)
    with open(os.path.join(OUT_DIR, "retrieval_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Retrieval analysis: which cache entries drive the prediction\n\n" + text + "\n" + notes)
    with open(os.path.join(OUT_DIR, "retrieval_summary.csv"), "w") as f:
        f.write("\n".join(csv_rows) + "\n")
    print("saved", os.path.join(OUT_DIR, "retrieval_summary.md"))


def main():
    p = argparse.ArgumentParser(description="Which cache entries drive TDA's predictions (session vs class)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--dataset", required=True, help="saved features, e.g. core50_350 or core50 (128)")
    s.add_argument("--backbone", required=True, choices=["RN50", "ViT-B/16"])
    s.add_argument("--mode", default="tda,ctxx", help="tda, ctx and/or ctxx")
    s.add_argument("--order", default="s0,s2,s3")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--window", type=int, default=300, help="CTX/CTXX window (frozen: 300)")
    s.add_argument("--strength", type=float, default=0.5, help="CTX/CTXX strength (frozen: 0.5)")
    s.add_argument("--cfg-like", default=None, help="caltech101 for CORe50 / iCWT")
    s.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    s.add_argument("--sessions", default=None, help="CORe50 test sessions: 3,7,10")
    s.add_argument("--every", type=int, default=1)
    sub.add_parser("summary")
    args = p.parse_args()
    if args.cmd == "summary":
        return summary(args)
    run(args)


if __name__ == "__main__":
    main()
