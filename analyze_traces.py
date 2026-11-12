"""
analyze_traces.py - what happens INSIDE the stream: plots and numbers from the per-image traces
saved by `streams.py run ... --trace`.

Question it helps answer: why does the stream order change TDA's accuracy (e.g. why S2 > S0 on
CORe50, and why S3 < S2)? It does not run any model: it only reads the trace files, so it works on
Colab or a laptop (needs pandas + matplotlib).

Usage (from the TDA_replication folder, same RESULTS_DIR as streams.py):

  python analyze_traces.py --dataset core50 --backbone RN50
  python analyze_traces.py --dataset core50 --backbone RN50 --modes tda,m2,oracle --orders s0,s2,s3
      --config caltech101   only runs with this config (as shown in the summary table)
      --frames "sessions 3,7,10"   only runs on these frames (as shown in the summary table)

Output: RESULTS_DIR/analysis/<dataset>__<backbone>/
  summary.md            one row per mode x order (mean over seeds), see the column notes in the file
  1_cold_start.png      accuracy along the stream (TDA vs CLIP): is the start (empty cache) worse?
  2_within_session.png  S2/S3: TDA gain over CLIP from the start to the end of each session block
  3_error_runs.png      how long are the runs of consecutive errors (errors that repeat = propagation)
  4_cache_session.png   TDA gain over CLIP vs share of the cache that comes from the current session
  5_cache_purity.png    share of cached samples with a correct pseudo-label, along the stream

Trace columns used (written by streams.py): step, label, pred, clip_pred, correct, clip_correct,
session, object, frame, pos_true_frac, pos_same_session_frac, pos_same_object_frac, pos_top_class_frac.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TDA_HOME = os.environ.get("TDA_HOME", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(TDA_HOME, "results"))
TRACE_DIR = os.path.join(RESULTS_DIR, "feature_traces")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def describe(r):
    """Same 'Config' and 'Frames' labels as `streams.py summary`."""
    ov = r.get("overrides") or {}
    config = (r.get("cfg_like") or r["dataset"]) + "".join(f" {k}={v}" for k, v in sorted(ov.items()))
    frames = "all" if not r.get("sessions") else f"sessions {r['sessions']}"
    if (r.get("every") or 1) > 1:
        frames += f", 1/{r['every']}"
    if r.get("limit"):
        frames += f", first {r['limit']}"
    return config, frames


def load_traces(args):
    """One DataFrame per run with a trace, plus the run's mode/order/seed."""
    path = os.path.join(RESULTS_DIR, "feature_runs.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found: run `python streams.py run ... --trace` first.")
    with open(path) as f:
        runs = [json.loads(line) for line in f if line.strip()]
    last = {}                                   # one line per run: the latest (e.g. a rerun with --trace)
    for r in runs:
        last[r.get("key") or id(r)] = r
    runs = list(last.values())
    modes = set(args.modes.split(",")) if args.modes else None
    orders = set(args.orders.split(",")) if args.orders else None
    out, configs = [], set()
    for r in runs:
        if not r.get("trace") or r["dataset"] != args.dataset or r["backbone"] != args.backbone:
            continue
        tag = r["order"] + (f" rho={r['rho']}" if r.get("rho") is not None else "")
        config, frames = describe(r)
        if (modes and r["mode"] not in modes) or (orders and r["order"] not in orders):
            continue
        if (args.config and config != args.config) or (args.frames and frames != args.frames):
            continue
        file = os.path.join(TRACE_DIR, r["trace"])
        if not os.path.isfile(file):
            print("missing trace:", file)
            continue
        t = pd.read_csv(file)
        t["pos"] = t["step"] / max(len(t) - 1, 1)            # position in the stream, 0 = start, 1 = end
        out.append({"mode": r["mode"], "order": tag, "seed": r["seed"], "trace": t})
        configs.add((config, frames))
    if not out:
        raise SystemExit("No traces match: check --dataset/--backbone/--modes/--orders, and that the runs used --trace.")
    if len(configs) > 1:
        print("WARNING: the traces mix several configs/frame sets; use --config and --frames to pick one:")
        for c in sorted(configs):
            print("   ", c)
    return out


# --------------------------------------------------------------------------------------
# Per-run quantities
# --------------------------------------------------------------------------------------
def error_runs(correct):
    """Lengths of the runs of consecutive errors (correct == 0)."""
    runs, n = [], 0
    for c in correct:
        if c == 0:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def session_blocks(t):
    """Position (0..1) of every frame inside its contiguous block of the same session (S2/S3)."""
    block = (t["session"] != t["session"].shift()).cumsum()
    rank = t.groupby(block).cumcount()
    size = t.groupby(block)["step"].transform("size")
    return rank / (size - 1).clip(lower=1)


def run_numbers(t, start_share):
    n_start = max(1, int(len(t) * start_share))
    has_cache = "pos_true_frac" in t
    num = lambda c: pd.to_numeric(t[c], errors="coerce").mean() if c in t else np.nan
    runs = error_runs(t["correct"].tolist())
    return {
        "acc": 100 * t["correct"].mean(),
        "clip": 100 * t["clip_correct"].mean(),
        "fixed": 100 * ((t["clip_correct"] == 0) & (t["correct"] == 1)).mean(),
        "broken": 100 * ((t["clip_correct"] == 1) & (t["correct"] == 0)).mean(),
        "start_gain": 100 * (t["correct"][:n_start].mean() - t["clip_correct"][:n_start].mean()),
        "rest_gain": 100 * (t["correct"][n_start:].mean() - t["clip_correct"][n_start:].mean()),
        "err_run": float(np.mean(runs)) if runs else 0.0,
        "err_in_long": 100 * sum(r for r in runs if r >= 10) / max(1, sum(runs)),
        "purity": 100 * num("pos_true_frac") if has_cache else np.nan,
        "same_session": 100 * num("pos_same_session_frac") if has_cache else np.nan,
        "same_object": 100 * num("pos_same_object_frac") if has_cache else np.nan,
        "top_class": 100 * num("pos_top_class_frac") if has_cache else np.nan,
    }


# --------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------
def binned(runs, col, by="pos", bins=40, fn="mean"):
    """Mean of `col` in `bins` equal slices of `by`, averaged over the runs (seeds)."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for r in runs:
        t = r["trace"]
        b = np.clip(np.digitize(t[by], edges) - 1, 0, bins - 1)
        rows.append(pd.Series(pd.to_numeric(t[col], errors="coerce")).groupby(b).agg(fn).reindex(range(bins)))
    return (edges[:-1] + edges[1:]) / 2, pd.concat(rows, axis=1).mean(axis=1)


def groups(traces):
    g = {}
    for r in traces:
        g.setdefault((r["mode"], r["order"]), []).append(r)
    return dict(sorted(g.items()))


def write_summary(traces, out_dir, start_share):
    cols = ["acc", "clip", "fixed", "broken", "start_gain", "rest_gain", "err_run", "err_in_long",
            "purity", "same_session", "same_object", "top_class"]
    lines = ["| Mode | Order | Seeds | " + " | ".join(cols) + " |", "|---|---|---:|" + "---:|" * len(cols)]
    for (mode, order), runs in groups(traces).items():
        nums = pd.DataFrame([run_numbers(r["trace"], start_share) for r in runs]).mean()
        cells = ["" if pd.isna(nums[c]) else f"{nums[c]:.2f}" for c in cols]
        lines.append(f"| {mode} | {order} | {len(runs)} | " + " | ".join(cells) + " |")
    notes = f"""
Column notes (all in %, mean over seeds, except err_run):
- acc / clip: accuracy of the mode and of CLIP alone on the same stream.
- fixed: frames CLIP gets wrong and the mode gets right; broken: CLIP right, mode wrong. acc - clip = fixed - broken.
- start_gain / rest_gain: (mode - CLIP) accuracy in the first {start_share:.0%} of the stream / in the rest (cold start).
- err_run: mean length of a run of consecutive errors; err_in_long: share of errors inside runs of 10 or more
  (long runs = an error that keeps repeating, e.g. a wrong pseudo-label propagating along a video).
- purity: share of samples in the positive cache whose pseudo-label is the true label.
- same_session / same_object: share of the positive cache coming from the session / the object (video) of the current frame.
- top_class: share of the positive cache taken by its largest class.
"""
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" + notes)
    print("\n".join(lines))


def plot_lines(ax, traces, col, by="pos", only_orders=None, minus_clip=False):
    for (mode, order), runs in groups(traces).items():
        if only_orders and order.split()[0] not in only_orders:
            continue
        if col not in runs[0]["trace"]:
            continue
        x, y = binned(runs, col, by)
        if minus_clip:
            y = y - binned(runs, "clip_correct", by)[1]
        ax.plot(x, 100 * y, label=f"{mode} {order}")


def make_plots(traces, out_dir):
    has_session = (traces[0]["trace"]["session"] >= 0).any()

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_lines(ax, traces, "correct")
    for order, runs in {o: [r for r in traces if r["order"] == o] for o in sorted({r["order"] for r in traces})}.items():
        x, y = binned(runs, "clip_correct")
        ax.plot(x, 100 * y, "k:", alpha=0.4)
    ax.set(xlabel="position in the stream (0 = start)", ylabel="accuracy (%)",
           title="Accuracy along the stream (dotted: CLIP alone)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "1_cold_start.png"), dpi=150); plt.close(fig)

    if has_session:
        for r in traces:
            r["trace"]["in_block"] = session_blocks(r["trace"])
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_lines(ax, traces, "correct", by="in_block", only_orders={"s2", "s3"}, minus_clip=True)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlabel="position inside the current session block (0 = new session starts)",
               ylabel="mode - CLIP accuracy (points)", title="Gain over CLIP inside each session (S2, S3)")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "2_within_session.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for (mode, order), runs in groups(traces).items():
        lengths = [n for r in runs for n in error_runs(r["trace"]["correct"].tolist())]
        if lengths:
            vals, counts = np.unique(np.clip(lengths, 1, 50), return_counts=True)
            ax.plot(vals, counts / counts.sum(), marker=".", label=f"{mode} {order}")
    ax.set(xscale="log", yscale="log", xlabel="length of a run of consecutive errors (50 = 50 or more)",
           ylabel="share of runs", title="Do errors repeat? (longer tail = errors propagate)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "3_error_runs.png"), dpi=150); plt.close(fig)

    with_cache = any("pos_true_frac" in r["trace"] for r in traces)    # clip/smooth runs have no cache
    if has_session and with_cache:
        fig, ax = plt.subplots(figsize=(7, 4))
        for (mode, order), runs in groups(traces).items():
            if "pos_same_session_frac" not in runs[0]["trace"]:
                continue
            t = pd.concat([r["trace"] for r in runs])
            share = pd.to_numeric(t["pos_same_session_frac"], errors="coerce")
            gain = t["correct"] - t["clip_correct"]
            b = pd.cut(share, np.linspace(0, 1, 11), include_lowest=True)
            g = gain.groupby(b, observed=True).mean()
            ax.plot([iv.mid for iv in g.index], 100 * g.values, marker="o", label=f"{mode} {order}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlabel="share of the positive cache from the current session", ylabel="mode - CLIP accuracy (points)",
               title="Does a cache from the same session help?")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "4_cache_session.png"), dpi=150); plt.close(fig)

    if with_cache:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_lines(ax, traces, "pos_true_frac")
        ax.set(xlabel="position in the stream (0 = start)", ylabel="cached samples with correct pseudo-label (%)",
               title="Cache purity along the stream")
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "5_cache_purity.png"), dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plots and numbers from streams.py traces")
    p.add_argument("--dataset", required=True, help="feature name, e.g. core50 or core50_350")
    p.add_argument("--backbone", required=True, choices=["RN50", "ViT-B/16"])
    p.add_argument("--modes", default=None, help="e.g. tda,m2,oracle (default: all with a trace)")
    p.add_argument("--orders", default=None, help="e.g. s0,s2,s3 (default: all)")
    p.add_argument("--config", default=None, help="only this config, as in the summary table (e.g. caltech101)")
    p.add_argument("--frames", default=None, help='only these frames, as in the summary table (e.g. "sessions 3,7,10")')
    p.add_argument("--start-share", type=float, default=0.05, help="share of the stream counted as 'start' (cold start)")
    args = p.parse_args()

    traces = load_traces(args)
    out_dir = os.path.join(RESULTS_DIR, "analysis", f"{args.dataset}__{args.backbone.replace('/', '-')}")
    os.makedirs(out_dir, exist_ok=True)
    write_summary(traces, out_dir, args.start_share)
    make_plots(traces, out_dir)
    print("\nsaved plots and summary.md in", out_dir)


if __name__ == "__main__":
    main()
