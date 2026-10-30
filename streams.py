"""
streams.py - TDA on SAVED CLIP features (made by extract_features.py), with different stream orders.

The features are computed once by extract_features.py; here every run only reorders rows of the
saved table and runs the cache logic, so a run takes seconds instead of minutes. This is the file
for the stream experiments (S0-S3, many seeds, cold start) and for method variants such as M2.

Commands (run from the TDA_replication folder; paths come from the same environment variables
as experiments.py: REPO_DIR, RESULTS_DIR):

  python streams.py run --dataset core50 --backbone RN50 --cfg-like caltech101 \
                        --order s0,s1,s2,s3 --rho 0.5,0.9 --mode tda,clip --repeats 50
      -> one run per mode x order x rho x seed; one JSON line each in RESULTS_DIR/feature_runs.jsonl
         (runs already there are skipped, so the same command can be relaunched)
         --sessions 3,7,10 / --every 5 use a subset of the CORe50 frames (the same for every order)
         --trace saves the per-image trace (cache size, correct, ...) in RESULTS_DIR/feature_traces/

  python streams.py summary [--dataset core50]
      -> table with mean accuracy over seeds and 95% CI; also RESULTS_DIR/feature_summary.csv

Orders (stream_order): s0 shuffled | s1 persistent labels (--rho) | s2 one session at a time |
s3 one session at a time, videos in temporal order. CLIP alone (--mode clip) has no memory, so its
accuracy must be the same in every order: a quick sanity check.

The method itself is NOT copied here: run_on_features() calls tda_replica.run_tda() with a stand-in
"model" whose image encoder returns the saved feature. Any change to the method in
tda_replica.py (e.g. M2) is therefore used automatically here too.
"""

import argparse
import json
import os
import random
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

import tda_replica as tda
from utils import get_config_file

ORDERS = ["stored", "random", "sorted", "s0", "s1", "s2", "s3"]


def feature_path(dataset, backbone):
    return os.path.join(FEATURE_DIR, f"{dataset}__{backbone.replace('/', '-')}.pt")


def load_features(dataset, backbone):
    path = feature_path(dataset, backbone)
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found: run `python extract_features.py extract --dataset {dataset} ...` first.")
    return torch.load(path, map_location="cpu")


# --------------------------------------------------------------------------------------
# Stream orders
# --------------------------------------------------------------------------------------
def stream_order(data, order, seed, dataset, rho=0.9):
    """
    Row indices in the order the test stream sees them.
      stored         : saved order (dataset index; CORe50 by session, object, frame)
      random / s0    : everything shuffled (same formula as tda_replica.make_stream -> same stream as on images)
      sorted         : grouped by class, random inside a class
      s1             : shuffled context, PERSISTENT LABELS: at each step, with probability rho the next
                       frame has the same class as the previous one, otherwise its class is drawn in
                       proportion to the frames left; the frame itself is drawn at random inside the
                       class (so any session). rho=0 is an ordinary shuffle. Every frame is used once.
      s2  (CORe50)   : one session at a time (sessions in random order), frames shuffled inside a session
      s3  (CORe50)   : one session at a time (random order); inside a session, the videos of the objects
                       in random order, each video with its frames in temporal order.
                       S3 differs from S2 only by the temporal order inside the session.
    The randomness depends only on (seed, dataset, order), so 50 seeds = 50 different streams.
    """
    labels = data["labels"]
    n = len(labels)
    if order == "stored":
        return list(range(n))
    if order in ("random", "s0", "sorted"):
        g = torch.Generator()
        g.manual_seed(seed * 1000003 + zlib.crc32(dataset.encode()))
        idx = torch.randperm(n, generator=g).tolist()
        if order == "sorted":
            lab = labels.tolist()
            idx.sort(key=lambda i: lab[i])
        return idx

    rng = random.Random(f"{seed}/{dataset}/{order}")
    if order == "s1":
        pools = {}
        for i, c in enumerate(labels.tolist()):
            pools.setdefault(c, []).append(i)
        for p in pools.values():
            rng.shuffle(p)
        idx, cls = [], None
        while len(idx) < n:
            if cls is None or not pools[cls] or rng.random() >= rho:
                left = [c for c in sorted(pools) if pools[c]]
                cls = rng.choices(left, weights=[len(pools[c]) for c in left])[0]
            idx.append(pools[cls].pop())
        return idx

    meta = data["meta"]
    if "session" not in meta:
        raise SystemExit(f"order {order} needs sessions/videos: only available for core50")
    sessions = sorted(set(meta["session"]))
    rng.shuffle(sessions)
    idx = []
    for s in sessions:
        rows = [i for i in range(n) if meta["session"][i] == s]     # stored order: by object, then frame
        if order == "s2":
            rng.shuffle(rows)
            idx += rows
        elif order == "s3":
            videos = {}
            for i in rows:
                videos.setdefault(meta["object"][i], []).append(i)
            objs = sorted(videos)
            rng.shuffle(objs)
            for o in objs:
                idx += videos[o]
        else:
            raise SystemExit(f"unknown order {order!r}; choose from {ORDERS}")
    return idx


def select_rows(data, sessions=None, every=1):
    """Keep only some CORe50 sessions and/or 1 frame out of `every` (same frames for every order)."""
    meta = data["meta"]
    keep = list(range(len(data["labels"])))
    if sessions:
        if "session" not in meta:
            raise SystemExit("--sessions is only available for core50")
        keep = [i for i in keep if meta["session"][i] in sessions]
    if every > 1:
        if "frame" not in meta:
            raise SystemExit("--every is only available for core50")
        keep = [i for i in keep if meta["frame"][i] % every == 0]
    if len(keep) == len(data["labels"]):
        return data
    out = dict(data, feats=data["feats"][keep], labels=data["labels"][keep])
    out["meta"] = {k: [v[i] for i in keep] for k, v in meta.items()}
    return out


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


def make_cfg(args, mode=None):
    mode = mode or args.mode
    cfg = get_config_file(os.path.join(REPO_DIR, "configs"), args.cfg_like or args.dataset)
    tda.apply_overrides(cfg, tda.parse_overrides(args.overrides))
    cfg["positive"]["enabled"] &= mode in ("tda", "pos")
    cfg["negative"]["enabled"] &= mode in ("tda", "neg")
    return cfg


def run_on_features(cfg, data, idx, device, limit=None, trace=None):
    feats = data["feats"].to(device)
    text_weights = data["text_weights"].to(device)
    if device == "cpu":                     # fp16 maths is slow/partial on CPU; results differ in the last digits
        feats, text_weights = feats.float(), text_weights.float()
    stream = FeatureStream(feats, data["labels"], idx)
    return tda.run_tda(cfg, stream, SavedFeatureEncoder(), text_weights, device, limit=limit, trace=trace)


def split_list(text, cast=str):
    return [cast(x) for x in str(text).split(",") if x.strip()]


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run(args):
    """
    One TDA run for every combination of --mode x --order x --rho (s1 only) x seed;
    one JSON line each in feature_runs.jsonl. Runs already in the file are skipped,
    so the same command can be relaunched after an interruption.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = split_list(args.sessions, int) if args.sessions else None
    data = select_rows(load_features(args.dataset, args.backbone), sessions, args.every)
    overrides = tda.parse_overrides(args.overrides)
    out_file = os.path.join(RESULTS_DIR, "feature_runs.jsonl")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = {r.get("key") for r in read_jsonl(out_file)}
    print(f"{args.dataset} {args.backbone}: {len(data['labels'])} frames, device {device}", flush=True)

    for mode in split_list(args.mode):
        cfg = make_cfg(args, mode)
        for order in split_list(args.order):
            for rho in (split_list(args.rho, float) if order == "s1" else [None]):
                for seed in range(args.seed, args.seed + args.repeats):
                    tag = order + (f"-rho{rho}" if rho is not None else "")
                    key = json.dumps([args.dataset, args.backbone, mode, tag, seed, overrides,
                                      args.cfg_like or args.dataset, args.sessions, args.every, args.limit])
                    if key in done:
                        continue
                    idx = stream_order(data, order, seed, args.dataset, rho=rho)
                    trace = [] if args.trace else None
                    res = run_on_features(cfg, data, idx, device, limit=args.limit, trace=trace)
                    name = tda.run_name("features", args.dataset, args.backbone, mode, seed, tag, overrides)
                    if trace:
                        for row, i in zip(trace, idx):   # saved row of each step (session, object, frame)
                            row["row"] = i
                        tda.write_trace(os.path.join(RESULTS_DIR, "feature_traces", name + ".csv.gz"), trace)
                    res.update({"key": key, "impl": "features", "dataset": args.dataset,
                                "backbone": args.backbone, "mode": mode, "order": order, "rho": rho,
                                "seed": seed, "overrides": overrides, "cfg_like": args.cfg_like or args.dataset,
                                "sessions": args.sessions, "every": args.every,
                                "trace": (name + ".csv.gz") if trace else None, "device": device,
                                "time": time.strftime("%Y-%m-%d %H:%M")})
                    print(f"  {mode:4s} {tag:10s} seed {seed:3d}: acc {res['acc']:.2f}  "
                          f"(CLIP {res['clip_acc']:.2f}, {res['seconds']}s)", flush=True)
                    with open(out_file, "a") as f:
                        f.write(json.dumps(res) + "\n")


def summary(args):
    """Table: mean accuracy over seeds with a 95% confidence interval, per dataset/backbone/mode/order."""
    rows = read_jsonl(os.path.join(RESULTS_DIR, "feature_runs.jsonl"))
    groups = {}
    for r in rows:
        if args.dataset and r["dataset"] != args.dataset:
            continue
        tag = r["order"] + (f" rho={r['rho']}" if r.get("rho") is not None else "")
        k = (r["dataset"], r["backbone"], r["mode"], tag, json.dumps(r.get("overrides") or {}))
        groups.setdefault(k, []).append(r)
    if not groups:
        raise SystemExit("No runs in feature_runs.jsonl yet: launch `python streams.py run ...` first.")
    lines = ["| Dataset | Backbone | Mode | Order | Overrides | Seeds | Acc (mean) | 95% CI ± | CLIP | TDA − CLIP |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    csv_rows = ["dataset,backbone,mode,order,overrides,seeds,acc_mean,ci95,clip_mean,gain"]
    for k in sorted(groups):
        accs = [r["acc"] for r in groups[k]]
        clips = [r["clip_acc"] for r in groups[k]]
        n, m, c = len(accs), sum(accs) / len(accs), sum(clips) / len(clips)
        sd = (sum((a - m) ** 2 for a in accs) / (n - 1)) ** 0.5 if n > 1 else 0.0
        ci = 1.96 * sd / n ** 0.5                     # normal approximation over seeds
        ov = "" if k[4] == "{}" else k[4]
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {ov} | {n} | {m:.2f} | {ci:.2f} | {c:.2f} | {m - c:+.2f} |")
        csv_rows.append(",".join([k[0], k[1], k[2], k[3], '"' + ov.replace('"', "'") + '"',
                                  str(n), f"{m:.4f}", f"{ci:.4f}", f"{c:.4f}", f"{m - c:.4f}"]))
    print("\n".join(lines))
    with open(os.path.join(RESULTS_DIR, "feature_summary.csv"), "w") as f:
        f.write("\n".join(csv_rows) + "\n")
    print("\nsaved", os.path.join(RESULTS_DIR, "feature_summary.csv"))


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="TDA on saved CLIP features, different stream orders")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--dataset", required=True, help="core50, or a dataset with saved features (dtd, eurosat, ...)")
    s.add_argument("--backbone", required=True, choices=["RN50", "ViT-B/16"])
    s.add_argument("--mode", default="tda", help=f"one or more, comma-separated, from {tda.MODES}, e.g. tda,clip")
    s.add_argument("--order", default="s0", help=f"one or more, comma-separated, from {ORDERS}, e.g. s0,s1,s2,s3")
    s.add_argument("--rho", default="0.9",
                   help="s1 only: probability that the label persists; several allowed, e.g. 0.5,0.9,0.95")
    s.add_argument("--seed", type=int, default=1, help="first seed")
    s.add_argument("--repeats", type=int, default=1, help="number of seeds (e.g. 50 different streams)")
    s.add_argument("--cfg-like", default=None,
                   help="use the hyper-parameters of this dataset's official config "
                        "(needed for core50, which has none; e.g. caltech101)")
    s.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                   help="override a config value, e.g. --set positive.alpha=3 (repeatable)")
    s.add_argument("--sessions", default=None, help="core50: only these sessions, e.g. 3,7,10")
    s.add_argument("--every", type=int, default=1, help="core50: only 1 frame out of N")
    s.add_argument("--limit", type=int, default=None, help="only the first N images of each stream")
    s.add_argument("--trace", action="store_true",
                   help="save a per-image trace (cache size, correct, ...) for cold-start analysis")
    s = sub.add_parser("summary", help="table of feature_runs.jsonl: mean and 95%% CI over seeds")
    s.add_argument("--dataset", default=None)
    args = p.parse_args()

    if args.cmd == "summary":
        return summary(args)
    bad = [x for x in split_list(args.mode) if x not in tda.MODES] + \
          [x for x in split_list(args.order) if x not in ORDERS]
    if bad:
        raise SystemExit(f"unknown mode/order {bad}; modes {tda.MODES}, orders {ORDERS}")
    if args.dataset == "core50" and not args.cfg_like:
        raise SystemExit("core50 has no official config: pass e.g. --cfg-like caltech101")
    os.chdir(REPO_DIR)                      # same working directory as tda_replica.py runs
    run(args)


if __name__ == "__main__":
    main()
