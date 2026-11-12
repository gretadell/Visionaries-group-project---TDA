#!/usr/bin/env python3
"""Select context-subtraction parameters without reading test-session results.

Put this file in the project root, alongside context_cache.py and streams.py.
In the project's existing Python environment (Python >= 3.7):
    python select_context_params.py --plan
    python select_context_params.py

Requires the original project's dependencies, a CUDA GPU, TDA/ and
results/features/core50_350__RN50.pt containing ALL eight non-test sessions.
Optional paths: --repo PATH --features PATH --out PATH --device cpu
CPU is supported explicitly but is slower and uses different precision.

Fixed protocol: RN50, caltech101, sessions 1,2,4,5,6,8,9,11, every=1,
orders s0/s2/s3, seeds 1/2/3. 9 baselines + 72 context runs.
Ranking: equal-weight mean of per-order paired improvements over TDA.
Exact ties: ctx before ctxx, window 300 before all, strength .5 before 1.

Calls the existing inference functions directly: no method duplication, no
changes to source files, no trace writes and no changes to feature_runs.jsonl.
Uses isolated output/checkpoints. Resumes only with identical provenance.
Old shared runs are not reused because their provenance may be incomplete.
The frozen JSON and final commands are written ONLY after all 81 runs pass.
Final test runs are NOT launched. No smoothing experiments are performed.

Outputs: manifest.json, checkpoints/*.json, all_runs.jsonl, ranking.csv,
paired_differences.csv, selection_report.md, frozen_context.json,
final_evaluation_commands.sh (review before running).

Validation without GPU or project dependencies:
    python select_context_params.py --self-test
"""
import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

SESSIONS = [1, 2, 4, 5, 6, 8, 9, 11]
ORDERS = ["s0", "s2", "s3"]
SEEDS = [1, 2, 3]
DATASET = "core50_350"
BACKBONE = "RN50"
CONFIG = "caltech101"
PROTOCOL_VERSION = 1
SCRIPT_PATH = Path(__file__).resolve()


def candidates():
    return [{"mode": m, "window": w, "strength": s}
            for m in ("ctx", "ctxx") for w in (300, "all") for s in (0.5, 1.0)]


def cid(c):
    if c["mode"] == "tda":
        return "tda"
    return "{}-w{}-strength{}".format(c["mode"], c["window"], c["strength"])


def jobs():
    return [dict(c, order=o, seed=s, id="{}__{}__seed{}".format(cid(c), o, s))
            for c in [{"mode": "tda"}] + candidates() for o in ORDERS for s in SEEDS]


def digest(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".writing-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    import io
    s = io.StringIO()
    writer = csv.DictWriter(s, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, s.getvalue())


def git_info(root):
    def call(args):
        try:
            return subprocess.check_output(["git", "-C", str(root)] + args,
                                           stderr=subprocess.DEVNULL, universal_newlines=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"commit": call(["rev-parse", "HEAD"]), "status": call(["status", "--porcelain"])}


def check_sessions(meta):
    if "session" not in meta:
        raise ValueError("Feature metadata has no session column.")
    present = set(int(s) for s in meta["session"])
    missing = sorted(set(SESSIONS) - present)
    if missing:
        raise ValueError("Missing non-test sessions {}. Extract these sessions first; "
                         "do not substitute test sessions 3,7,10.".format(missing))


def validate_result(r, job, n, baseline=None):
    for k, v in job.items():
        if r.get(k) != v:
            raise ValueError("Checkpoint/job mismatch: {}".format(k))
    if r.get("n") != n:
        raise ValueError("Wrong frame count for {}".format(job["id"]))
    for field in ("acc", "clip_acc"):
        x = r.get(field)
        if not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 100:
            raise ValueError("Invalid {} for {}".format(field, job["id"]))
    if baseline is not None and abs(r["clip_acc"] - baseline["clip_acc"]) > 1e-6:
        raise ValueError("CLIP baseline changed on the same stream: " + job["id"])


def summarize(rows):
    expected = {j["id"] for j in jobs()}
    by_id = {r["id"]: r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != expected:
        raise ValueError("Selection requires exactly the 81 distinct protocol runs.")
    ranking, paired = [], []
    for c in candidates():
        line = dict(candidate=cid(c), **c)
        gains = []
        for order in ORDERS:
            ds, accs = [], []
            for seed in SEEDS:
                a = by_id["{}__{}__seed{}".format(cid(c), order, seed)]
                b = by_id["tda__{}__seed{}".format(order, seed)]
                delta = a["acc"] - b["acc"]
                ds.append(delta)
                accs.append(a["acc"])
                paired.append(dict(candidate=cid(c), order=order, seed=seed,
                                   context_acc=a["acc"], tda_acc=b["acc"], clip_acc=b["clip_acc"],
                                   context_minus_tda=delta, context_minus_clip=a["acc"] - b["clip_acc"]))
            line[order + "_acc"] = statistics.mean(accs)
            line[order + "_gain"] = statistics.mean(ds)
            line[order + "_gain_sd"] = statistics.stdev(ds)
            line[order + "_positive_seeds"] = sum(x > 0 for x in ds)
            line[order + "_seeds"] = len(ds)
            gains.append(statistics.mean(ds))
        line["score"] = statistics.mean(gains)
        ranking.append(line)
    ranking.sort(key=lambda r: (-r["score"], r["mode"] != "ctx", r["window"] != 300, r["strength"]))
    return ranking, paired


def build_report(ranking, paired):
    winner = ranking[0]
    lines = ["# Context-subtraction selection (validation only)", "",
             "CORe50 350 / RN50 / caltech101 / non-test sessions " + ",".join(map(str, SESSIONS)),
             "Seeds 1-3; all available frames, no subsampling. No test accuracies used in ranking.", "",
             "Score = equally weighted mean of the S0/S2/S3 mean paired gains over TDA.",
             "Differences and SDs are percentage points. Rank uses unrounded values.", "",
             "| Rank | Candidate | S0 gain | S2 gain | S3 gain | Score |",
             "|---:|---|---:|---:|---:|---:|"]
    for i, r in enumerate(ranking, 1):
        lines.append("| {} | {} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} |".format(
            i, r["candidate"], r["s0_gain"], r["s2_gain"], r["s3_gain"], r["score"]))
    lines += ["", "Selected: **{}** (score {:+.4f}).".format(winner["candidate"], winner["score"])]
    if winner["score"] <= 0:
        lines += ["**No candidate improves the predefined mean validation objective.** "
                  "The best-ranked candidate is recorded, not claimed to be an improvement."]
    lines += ["", "## Selected candidate: individual paired differences", "",
              "| Order | Seed | Context | TDA | Context - TDA | Context - CLIP |",
              "|---|---:|---:|---:|---:|---:|"]
    for p in paired:
        if p["candidate"] == winner["candidate"]:
            lines.append("| {} | {} | {:.4f} | {:.4f} | {:+.4f} | {:+.4f} |".format(
                p["order"], p["seed"], p["context_acc"], p["tda_acc"],
                p["context_minus_tda"], p["context_minus_clip"]))
    lookup = {(r["mode"], r["window"], r["strength"]): r for r in ranking}
    lines += ["", "## Diagnostic contrasts (descriptive)", "",
              "Positive numbers favour the first option. These are not causal tests or significance claims.", ""]
    for mode in ("ctx", "ctxx"):
        for strength in (0.5, 1.0):
            a, b = lookup[(mode, "all", strength)], lookup[(mode, 300, strength)]
            lines.append("- {} strength {}: cumulative-all minus W300 score = {:+.4f}.".format(
                mode, strength, a["score"] - b["score"]))
    for window in (300, "all"):
        for strength in (0.5, 1.0):
            delta = lookup[("ctxx", window, strength)]["s3_gain"] - lookup[("ctx", window, strength)]["s3_gain"]
            lines.append("- S3, W{} strength {}: ctxx minus ctx = {:+.4f}.".format(window, strength, delta))
    lines += ["", "## Interpretation and provenance", "",
              "- Three selection seeds are exploratory; no significance-based selection is performed.",
              "- Equal order weighting is fixed in advance. Inspect per-order regressions, not just the mean.",
              "- Window all is the cumulative mean of past frames, not a future-aware global mean.",
              "- Comparing all and 300 probes cumulative versus recent centering; it does not prove background removal.",
              "- ctxx uses pseudo-labels to exclude the current pseudo-class; its effect does not isolate objects perfectly.",
              "- Two strengths (0.5/1.0, W300) were previously explored on test sessions. Disclose this pilot.",
              "- The selected config must transfer unchanged to ViT, CORe50 128, iCWT and UCF101.",
              "- No final test runs have been executed by this script.",
              "- manifest.json records feature SHA256, source SHA256, runtime and resolved TDA config.",
              "- Existing shared results were not reused; only identical-provenance checkpoints are resumed."]
    return "\n".join(lines) + "\n"


def final_commands(winner, repo):
    # Separate results directories prevent mixing previous pilots or CPU/GPU runs.
    q = shlex.quote
    lines = ["#!/usr/bin/env bash", "# Generated only; NOT executed. Review feature names before running.",
             "# Activate the project environment and allocate a GPU first.",
             "set -euo pipefail", "cd " + q(str(repo)),
             'export TDA_HOME="$PWD"', 'export REPO_DIR="${REPO_DIR:-$PWD/TDA}"',
             '# Set FEATURE_SOURCE if features are stored elsewhere.',
             'FEATURE_SOURCE="${FEATURE_SOURCE:-$PWD/results/features}"',
             '# NEW destination: do not point this at an old experiment folder.',
             'FINAL_ROOT="${FINAL_ROOT:-$PWD/results/context_final_$(date +%Y%m%d_%H%M%S)}"', "",
             'run_dataset() {', '  local ds="$1" cfg="$2" sessions="$3"',
             '  export RESULTS_DIR="$FINAL_ROOT/$ds"',
             '  mkdir -p "$RESULTS_DIR/features"',
             '  for bb in RN50 ViT-B/16; do',
             r'    local filename="${ds}__${bb//\//-}.pt"',
             '    test -f "$FEATURE_SOURCE/$filename" || { echo "Missing $FEATURE_SOURCE/$filename"; return 1; }',
             '    ln -s "$(realpath "$FEATURE_SOURCE/$filename")" "$RESULTS_DIR/features/$filename"',
             '  done', '  local extra=()',
             '  if [[ -n "$sessions" ]]; then extra+=(--sessions "$sessions"); fi',
             '  for bb in RN50 ViT-B/16; do',
             '    python streams.py run --dataset "$ds" --backbone "$bb" --cfg-like "$cfg" '
             '--order s0,s2,s3 --mode tda --seed 1 --repeats 10 "${extra[@]}"',
             '    python context_cache.py --dataset "$ds" --backbone "$bb" --cfg-like "$cfg" '
             '--order s0,s2,s3 --seed 1 --repeats 10 --mode {} --window {} --strength {} "${{extra[@]}}"'.format(
                 winner["mode"], winner["window"], winner["strength"]),
             '  done', '  python streams.py summary --dataset "$ds"', '}', "",
             '# Nick: uncomment the datasets assigned to you.',
             '# run_dataset core50_350 caltech101 3,7,10',
             '# run_dataset icwt caltech101 ""',
             '# Chiara (verify the feature filename is core50):',
             '# run_dataset core50 caltech101 3,7,10',
             '# Greta:', '# run_dataset ucf101_video_long ucf101 ""', "",
             '# No --trace: generated commands cannot collide with previous trace files.',
             '# Extend paired_tests.py for context-vs-TDA comparisons on the final results.']
    return "\n".join(lines) + "\n"


def execute(args):
    repo = Path(args.repo).expanduser().resolve()
    official = Path(os.environ.get("REPO_DIR", str(repo / "TDA"))).expanduser().resolve()
    result_root = Path(os.environ.get("RESULTS_DIR", str(repo / "results"))).expanduser().resolve()
    feature = Path(args.features).expanduser().resolve() if args.features else result_root / "features" / "core50_350__RN50.pt"
    out = Path(args.out).expanduser().resolve() if args.out else result_root / "context_selection"
    for name in ("context_cache.py", "streams.py", "tda_replica.py"):
        if not (repo / name).is_file():
            raise ValueError("Missing {} in --repo {}".format(name, repo))
    if not (official / "configs").is_dir():
        raise ValueError("Official TDA/configs is missing. Run the project setup first.")
    if not feature.is_file():
        raise ValueError("Missing feature file: " + str(feature))
    os.environ.update(TDA_HOME=str(repo), REPO_DIR=str(official), RESULTS_DIR=str(result_root))
    sys.path.insert(0, str(repo))
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable: allocate a GPU, or explicitly use --device cpu (slower/different precision).")
    import streams
    import context_cache
    os.chdir(str(official))
    print("Loading trusted local feature file:", feature, flush=True)
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False  # project-owned feature dict; compatible with newer PyTorch
    data = torch.load(str(feature), **kwargs)
    if data.get("dataset") != "core50" or data.get("backbone") != BACKBONE:
        raise ValueError("Expected CORe50 RN50 feature metadata; verify the selected file.")
    meta = data["meta"]
    check_sessions(meta)
    n_all = len(data["labels"])
    if len(data["feats"]) != n_all or any(len(v) != n_all for v in meta.values()):
        raise ValueError("Inconsistent feature/label/metadata lengths.")
    data = streams.select_rows(data, SESSIONS, every=1)
    if set(int(s) for s in data["meta"]["session"]) != set(SESSIONS):
        raise ValueError("Session selection failed.")
    n = len(data["labels"])
    if not n:
        raise ValueError("Empty validation set.")
    cfg_args = argparse.Namespace(cfg_like=CONFIG, dataset=DATASET, overrides=None, mode="tda")
    cfg = streams.make_cfg(cfg_args, "tda")
    hashes = {"selector": digest(SCRIPT_PATH)}
    for name in ("context_cache.py", "streams.py", "tda_replica.py"):
        hashes[name] = digest(repo / name)
    official_sources = sorted(list(official.rglob("*.py")) + list((official / "configs").rglob("*.yaml")))
    official_hashes = {str(p.relative_to(official)): digest(p) for p in official_sources}
    print("Hashing features for safe resume; selected {} frames.".format(n), flush=True)
    identity = dict(protocol_version=PROTOCOL_VERSION, dataset=DATASET, backbone=BACKBONE,
                    sessions=SESSIONS, orders=ORDERS, seeds=SEEDS, every=1, candidates=candidates(),
                    config=cfg, frame_count=n, feature_sha256=digest(feature), sources=hashes,
                    official_sources=official_hashes, python=platform.python_version(), torch=str(torch.__version__),
                    device=args.device, dtype=str(data["feats"].dtype), text_dtype=str(data["text_weights"].dtype),
                    cuda=str(torch.version.cuda), gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else platform.machine(),
                    cudnn=torch.backends.cudnn.version(),
                    matmul_tf32=getattr(getattr(torch.backends.cuda, "matmul", None), "allow_tf32", None))
    fingerprint = hashlib.sha256(canonical(identity).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    # POSIX advisory lock: released automatically on interruption; no stale lock blocks resume.
    import fcntl
    with open(str(out / ".selection.lock"), "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another selection is running in this output directory.")
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old.get("fingerprint") != fingerprint:
                raise ValueError("Provenance changed (code/features/runtime/protocol). Use a NEW --out directory.")
        else:
            if (out / "checkpoints").exists():
                raise ValueError("Checkpoints without a manifest: use a new --out directory.")
            write_json(manifest_path, dict(fingerprint=fingerprint, identity=identity,
                                          feature_path=str(feature), project_git=git_info(repo),
                                          official_git=git_info(official), created=time.strftime("%Y-%m-%dT%H:%M:%S")))
        rows, baselines, started = [], {}, time.monotonic()
        for number, job in enumerate(jobs(), 1):
            cp = out / "checkpoints" / (job["id"] + ".json")
            pair = (job["order"], job["seed"])
            if cp.exists():
                r = json.loads(cp.read_text())
                if r.get("fingerprint") != fingerprint:
                    raise ValueError("Checkpoint provenance mismatch: " + str(cp))
                action = "resume"
            else:
                idx = streams.stream_order(data, job["order"], job["seed"], "core50")
                print("[{}/81] {} ({} frames)".format(number, job["id"], n), flush=True)
                if job["mode"] == "tda":
                    res = streams.run_on_features(copy.deepcopy(cfg), data, idx, args.device)
                else:
                    res = context_cache.run_context_tda(copy.deepcopy(cfg), data, idx, args.device,
                        window=None if job["window"] == "all" else job["window"],
                        exclude_class=job["mode"] == "ctxx", strength=job["strength"])
                r = dict(res, **job)
                r["fingerprint"] = fingerprint
                validate_result(r, job, n, baselines.get(pair))
                write_json(cp, r)
                action = "done"
            validate_result(r, job, n, baselines.get(pair))
            if job["mode"] == "tda":
                baselines[pair] = r
            rows.append(r)
            print("[{}/81] {} {}: {:.4f}%".format(number, action, job["id"], r["acc"]), flush=True)
        ranking, paired = summarize(rows)
        winner = ranking[0]
        write_csv(out / "ranking.csv", ranking)
        write_csv(out / "paired_differences.csv", paired)
        atomic_text(out / "all_runs.jsonl", "".join(canonical(r) + "\n" for r in rows))
        atomic_text(out / "selection_report.md", build_report(ranking, paired))
        frozen = dict(selected={k: winner[k] for k in ("mode", "window", "strength")},
                      validation_score=winner["score"], validation_improved=winner["score"] > 0,
                      fingerprint=fingerprint, protocol=identity,
                      selection_rule="Equal-weight mean of S0/S2/S3 paired gains over TDA; seeds 1-3.",
                      exact_tie_break="ctx, then 300, then 0.5",
                      prior_test_exploration="W300 strengths 0.5 and 1.0 were previously explored on test sessions.",
                      final_test_executed=False)
        write_json(out / "frozen_context.json", frozen)
        atomic_text(out / "final_evaluation_commands.sh", final_commands(winner, repo))
        print("\nSELECTED:", frozen["selected"], "score:", winner["score"])
        print("Outputs:", out, "\nElapsed this invocation: {:.1f} minutes".format((time.monotonic() - started) / 60))
        if winner["score"] <= 0:
            print("No positive mean validation gain. Do not claim an improvement.")


def self_test():
    assert len(jobs()) == 81 and len({j["id"] for j in jobs()}) == 81
    check_sessions({"session": SESSIONS + [3, 7, 10]})
    try:
        check_sessions({"session": [3, 7, 10]})
        raise AssertionError("Missing validation sessions were accepted")
    except ValueError:
        pass
    rows = []
    for j in jobs():
        baseline = 50 + j["seed"] + ORDERS.index(j["order"])
        gain = 0 if j["mode"] == "tda" else 2
        rows.append(dict(j, acc=baseline + gain, clip_acc=49.0, n=100))
    ranking, paired = summarize(rows)
    assert ranking[0]["candidate"] == "ctx-w300-strength0.5"
    assert all(r["score"] == 2 for r in ranking)
    assert len(paired) == 72 and all(p["context_minus_tda"] == 2 for p in paired)
    try:
        summarize(rows[:-1])
        raise AssertionError("Incomplete grid was ranked")
    except ValueError:
        pass
    try:
        validate_result(dict(rows[0], clip_acc=48), jobs()[0], 100, rows[0])
        raise AssertionError("Changed CLIP baseline was accepted")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        write_json(p / "test.json", rows[0])
        assert json.loads((p / "test.json").read_text()) == rows[0]
        write_csv(p / "ranking.csv", ranking)
        assert len(list(csv.DictReader((p / "ranking.csv").open()))) == 8
        script = p / "final.sh"
        atomic_text(script, final_commands(ranking[0], Path("/tmp/project with spaces")))
        subprocess.run(["bash", "-n", str(script)], check=True)
    assert "No candidate" not in build_report(ranking, paired)
    print("PASS: grid, session exclusion, paired ranking, exact ties, incomplete-grid rejection, "
          "baseline checks, output serialization and generated shell syntax. No inference tested.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=str(SCRIPT_PATH.parent))
    p.add_argument("--features", help="Trusted project feature .pt file (default RESULTS_DIR/features/core50_350__RN50.pt)")
    p.add_argument("--out", help="Isolated selection directory (default RESULTS_DIR/context_selection)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--plan", action="store_true", help="Print the fixed grid without loading features or running inference")
    p.add_argument("--self-test", action="store_true", help="Test orchestration helpers without torch/GPU")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if args.plan:
        print("VALIDATION ONLY: RN50, caltech101, sessions", SESSIONS)
        print("9 TDA + 72 context runs. No subsampling, no traces, no final tests.")
        for j in jobs():
            print(j["id"])
        return
    try:
        execute(args)
    except (ValueError, ImportError, OSError) as e:
        raise SystemExit("STOP: " + str(e))


if __name__ == "__main__":
    main()
