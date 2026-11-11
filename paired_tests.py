"""
paired_tests.py - per-seed comparisons of the runs saved by streams.py (feature_runs.jsonl).

`streams.py summary` gives mean and 95% CI of each mode/order separately. To say whether e.g. M2 beats
TDA, the right test compares the two modes seed by seed: for a given order and seed every mode sees
EXACTLY the same stream (stream_order() depends only on seed, dataset and order), so most of the
seed-to-seed noise cancels in the difference.

Two kinds of comparison:
  paired    same order, same seed, two modes (or a mode vs CLIP on the same run):
            M2 - TDA, Oracle - TDA, TDA - CLIP, Smooth - CLIP, Smooth - TDA (S3)
  unpaired  two ORDERS of the same mode (S2 - S0, S3 - S2, S1 - S0): with the same seed the two orders
            are independent streams (the generator is seeded with the order name too), so pairing by
            seed would be wrong; they are compared as two independent samples (Welch).

Usage (needs numpy; uses scipy for the t-test p-values if available):
  python paired_tests.py results/feature_runs.jsonl --dataset core50_350
  python paired_tests.py FILE.jsonl --dataset icwt --config caltech101 --out icwt_tests.md
Options:
  --config   only runs with this config, as in the summary table (e.g. caltech101)
  --frames   only runs on these frames, as in the summary table (e.g. "sessions 3,7,10")
  --modes    modes compared across orders in the unpaired tests (default tda)
  --boot     bootstrap resamples for the 95% CI (default 10000)
  --out      also write the tables to this Markdown file

Columns: diff = mean of A - B in accuracy points; CI = 95% bootstrap interval of that mean;
A>B / A<B / = = number of seeds (paired) where A is better / worse / equal; p sign = exact two-sided
sign test (paired only); p t = paired t-test (paired) or Welch t-test (unpaired).
Many tests are run at once: read a single p around 0.05 with care.
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
except ImportError:          # p-values of the t-tests fall back to a normal approximation
    stats = None


# --------------------------------------------------------------------------------------
# Loading (same run selection as streams.py summary and analyze_traces.py)
# --------------------------------------------------------------------------------------
def describe(r):
    """Same 'Config' and 'Frames' labels as `streams.py summary`."""
    ov = r.get("overrides") or {}
    config = (r.get("cfg_like") or r["dataset"]) + "".join(" {}={}".format(k, v) for k, v in sorted(ov.items()))
    frames = "all" if not r.get("sessions") else "sessions {}".format(r["sessions"])
    if (r.get("every") or 1) > 1:
        frames += ", 1/{}".format(r["every"])
    if r.get("limit"):
        frames += ", first {}".format(r["limit"])
    return config, frames


def order_tag(r):
    return r["order"] + (" rho={}".format(r["rho"]) if r.get("rho") is not None else "")


def load(path, dataset, config=None, frames=None):
    """{(backbone, config, frames): {(mode, order tag): {seed: run}}}, keeping the latest line of each run."""
    last = {}
    with open(path) as f:
        for n, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                last[r.get("key") or ("line", n)] = r
    out = defaultdict(lambda: defaultdict(dict))
    for r in last.values():
        if r["dataset"] != dataset:
            continue
        c, fr = describe(r)
        if (config and c != config) or (frames and fr != frames):
            continue
        out[(r["backbone"], c, fr)][(r["mode"], order_tag(r))][r["seed"]] = r
    return out


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def sign_test(pos, neg):
    """Exact two-sided sign test (ties already removed)."""
    n = pos + neg
    if n == 0:
        return float("nan")
    k = min(pos, neg)
    tail = sum(math.factorial(n) // (math.factorial(i) * math.factorial(n - i)) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2 * tail)


def t_pvalue(t, df):
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    if stats is not None:
        return float(2 * stats.t.sf(abs(t), df))
    return math.erfc(abs(t) / math.sqrt(2))          # normal approximation


def paired(a, b, boot, rs):
    """a, b: {seed: accuracy}. Differences on the seeds present in both."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    n = len(d)
    if n == 0:
        return None
    mean = float(d.mean())
    means = d[rs.randint(0, n, size=(boot, n))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    pos, neg = int((d > 1e-9).sum()), int((d < -1e-9).sum())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd > 0:
        t = mean / (sd / math.sqrt(n))
    else:
        t = float("nan") if abs(mean) < 1e-12 else math.copysign(float("inf"), mean)
    return {"n": str(n), "diff": mean, "lo": float(lo), "hi": float(hi),
            "counts": "{} / {} / {}".format(pos, neg, n - pos - neg),
            "p_sign": sign_test(pos, neg), "p_t": t_pvalue(t, n - 1)}


def unpaired(a, b, boot, rs):
    """a, b: {seed: accuracy} of two independent sets of streams (Welch)."""
    x = np.array(list(a.values()), dtype=float)
    y = np.array(list(b.values()), dtype=float)
    if len(x) == 0 or len(y) == 0:
        return None
    mean = float(x.mean() - y.mean())
    means = x[rs.randint(0, len(x), size=(boot, len(x)))].mean(1) - y[rs.randint(0, len(y), size=(boot, len(y)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    vx = x.var(ddof=1) / len(x) if len(x) > 1 else 0.0
    vy = y.var(ddof=1) / len(y) if len(y) > 1 else 0.0
    se = math.sqrt(vx + vy)
    if se > 0:
        t = mean / se
        df = (vx + vy) ** 2 / ((vx ** 2 / (len(x) - 1) if len(x) > 1 else 0) + (vy ** 2 / (len(y) - 1) if len(y) > 1 else 0))
    else:
        t, df = (float("nan") if abs(mean) < 1e-12 else math.copysign(float("inf"), mean)), 1
    return {"n": "{} vs {}".format(len(x), len(y)), "diff": mean, "lo": float(lo), "hi": float(hi),
            "counts": "", "p_sign": float("nan"), "p_t": t_pvalue(t, df)}


# --------------------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------------------
NAMES = {"tda": "TDA", "m2": "M2", "oracle": "Oracle", "clip": "CLIP", "pos": "Pos", "neg": "Neg"}


def name(mode):
    if mode.startswith("smooth-w"):
        return "Smooth w" + mode[len("smooth-w"):]
    return NAMES.get(mode, mode)


def label(tag):
    """'s1 rho=0.5' -> 'S1 rho=0.5'."""
    head, _, rest = tag.partition(" ")
    return head.upper() + (" " + rest if rest else "")


def acc(runs):
    return {s: r["acc"] for s, r in runs.items()}


def clip_acc(runs):
    return {s: r["clip_acc"] for s, r in runs.items()}


def comparisons(groups, modes_across, boot, rs):
    rows = []
    rank = {"s0": 0, "s1": 1, "s2": 2, "s3": 3, "instance": 4}
    tags = sorted({t for _, t in groups}, key=lambda t: (rank.get(t.split()[0], 9), t))
    # paired: same order, same seed
    for tag in tags:
        g = {m: runs for (m, t), runs in groups.items() if t == tag}
        for m in ("m2", "oracle"):
            if m in g and "tda" in g:
                rows.append(("{} − TDA".format(name(m)), tag, "paired", paired(acc(g[m]), acc(g["tda"]), boot, rs)))
        for m in sorted(g):
            if m == "clip":
                continue
            if not m.startswith("smooth") and m != "tda":
                continue
            rows.append(("{} − CLIP".format(name(m)), tag, "paired", paired(acc(g[m]), clip_acc(g[m]), boot, rs)))
            if m.startswith("smooth") and "tda" in g:
                rows.append(("{} − TDA".format(name(m)), tag, "paired", paired(acc(g[m]), acc(g["tda"]), boot, rs)))
    # unpaired: two orders of the same mode
    pairs = [("s2", "s0"), ("s3", "s2"), ("s3", "s0")] + [(t, "s0") for t in tags if t.startswith("s1 ")]
    for m in modes_across:
        for a, b in pairs:
            if (m, a) in groups and (m, b) in groups:
                rows.append(("{}: {} − {}".format(name(m), label(a), label(b)), "", "unpaired",
                             unpaired(acc(groups[(m, a)]), acc(groups[(m, b)]), boot, rs)))
    return [r for r in rows if r[3] is not None]


def fmt_p(p):
    if p != p:          # nan
        return ""
    return "<0.001" if p < 0.001 else "{:.3f}".format(p)


def table(rows):
    lines = ["| Comparison | Order | Type | Seeds | Diff (pts) | 95% CI | A>B / A<B / = | p sign | p t |",
             "|---|---|---|---:|---:|---|---|---:|---:|"]
    for comp, tag, kind, r in rows:
        lines.append("| {} | {} | {} | {} | {:+.2f} | [{:+.2f}, {:+.2f}] | {} | {} | {} |".format(
            comp, tag, kind, r["n"], r["diff"], r["lo"], r["hi"], r["counts"], fmt_p(r["p_sign"]), fmt_p(r["p_t"])))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Per-seed comparisons of streams.py runs (feature_runs.jsonl)")
    p.add_argument("jsonl", help="feature_runs.jsonl (or a backup copy of it)")
    p.add_argument("--dataset", required=True, help="feature name, e.g. core50_350 or icwt")
    p.add_argument("--config", default=None, help="e.g. caltech101 (as in the summary table)")
    p.add_argument("--frames", default=None, help='e.g. "sessions 3,7,10" (as in the summary table)')
    p.add_argument("--modes", default="tda", help="modes compared across orders, unpaired (default tda)")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0, help="seed of the bootstrap")
    p.add_argument("--out", default=None, help="also write the tables to this Markdown file")
    args = p.parse_args()

    data = load(args.jsonl, args.dataset, args.config, args.frames)
    if not data:
        raise SystemExit("No runs of {} match (check --dataset, --config, --frames).".format(args.dataset))
    rs = np.random.RandomState(args.seed)
    out = ["# Per-seed comparisons: {}".format(args.dataset), "",
           "Source: `{}`. Bootstrap: {} resamples. t-test p-values: {}.".format(
               args.jsonl, args.boot, "scipy" if stats else "normal approximation (scipy not found)")]
    for (backbone, config, frames) in sorted(data):
        groups = data[(backbone, config, frames)]
        out += ["", "## {} — config {} — frames {}".format(backbone, config, frames), ""]
        out.append(table(comparisons(groups, [m.strip() for m in args.modes.split(",") if m.strip()], args.boot, rs)))
    text = "\n".join(out) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print("saved", args.out)


if __name__ == "__main__":
    main()