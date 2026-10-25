"""
experiments.py - runs the experiments of experiments.yaml and builds the report.

    python experiments.py list                          experiments defined in experiments.yaml
    python experiments.py check                         dataset sizes + equivalence test (GPU)
    python experiments.py run <name> [--backbone B]     run one experiment (GPU); --dry-run to preview
    python experiments.py report                        rebuild results/REPORT.md (no GPU needed)
    python experiments.py plan <name>                   (used by run.sh) backbone + time limit per job

On the cluster you normally use `bash run.sh ...`, which calls this file inside SLURM jobs.
Paths come from environment variables (set by run.sh, or by the Colab notebook):
    TDA_HOME     this repository             (default: the folder of this file)
    REPO_DIR     the official TDA repository (default: TDA_HOME/TDA)
    DATA_DIR     datasets                    (default: ~/tda_data)
    RESULTS_DIR  results                     (default: TDA_HOME/results)
    LOG_DIR      logs                        (default: TDA_HOME/logs)

Everything is written to RESULTS_DIR:
    runs.jsonl          one JSON line per (run, dataset): accuracy, CLIP accuracy, settings, ...
    traces/*.csv.gz     one file per (run, dataset): what happened at every test image
    dataset_check.json  test-set sizes found on disk
    REPORT.md           tables vs paper, ablations, every other experiment
    summary.csv         the numbers of REPORT.md in long format (Excel / pandas)
    shot_capacity.png   Fig. 4
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time

import yaml

# ======================================================================================
# Paths and constants
# ======================================================================================
TDA_HOME = os.environ.get("TDA_HOME", os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.environ.get("REPO_DIR", os.path.join(TDA_HOME, "TDA"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/tda_data"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(TDA_HOME, "results"))
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(TDA_HOME, "logs"))
TRACE_DIR = os.path.join(RESULTS_DIR, "traces")
RUNS_FILE = os.path.join(RESULTS_DIR, "runs.jsonl")
CHECK_FILE = os.path.join(RESULTS_DIR, "dataset_check.json")
EXPERIMENTS_FILE = os.path.join(TDA_HOME, "experiments.yaml")
REPLICA = os.path.join(TDA_HOME, "tda_replica.py")

# Dataset order of the official scripts (scripts/run_*_benchmark_*.sh). Do not change it:
# with the official stream, each dataset depends on the RNG state left by the previous ones.
ORDER = {
    "ood": ["I", "A", "V", "R", "S"],
    "cd": ["caltech101", "dtd", "eurosat", "fgvc", "food101",
           "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"],
}
ALL_DATASETS = ORDER["ood"] + ORDER["cd"]
PRETTY = {"I": "ImageNet", "A": "ImageNet-A", "V": "ImageNet-V2", "R": "ImageNet-R", "S": "ImageNet-Sketch",
          "caltech101": "Caltech101", "dtd": "DTD", "eurosat": "EuroSAT", "fgvc": "Aircraft",
          "food101": "Food101", "oxford_flowers": "Flower102", "oxford_pets": "Pets",
          "stanford_cars": "Cars", "sun397": "SUN397", "ucf101": "UCF101"}
# Size of the test split used by the paper (CoOp splits / official test sets)
EXPECTED_SIZE = {"I": 50000, "A": 7500, "V": 10000, "R": 30000, "S": 50889,
                 "caltech101": 2465, "dtd": 1692, "eurosat": 8100, "fgvc": 3333, "food101": 30300,
                 "oxford_flowers": 2463, "oxford_pets": 3669, "stanford_cars": 8041, "sun397": 19850,
                 "ucf101": 3783}
# TDA rows of Tables 1 and 3 of the paper (top-1 %)
PAPER_TDA = {
    "RN50": {"I": 61.35, "A": 30.29, "V": 55.54, "R": 62.58, "S": 38.12,
             "fgvc": 17.61, "caltech101": 89.70, "stanford_cars": 57.78, "dtd": 43.74,
             "eurosat": 42.11, "oxford_flowers": 68.74, "food101": 77.75,
             "oxford_pets": 86.18, "sun397": 62.53, "ucf101": 64.18},
    "ViT-B/16": {"I": 69.51, "A": 60.11, "V": 64.67, "R": 80.24, "S": 50.54,
                 "fgvc": 23.91, "caltech101": 94.24, "stanford_cars": 67.28, "dtd": 47.40,
                 "eurosat": 58.00, "oxford_flowers": 71.42, "food101": 86.14,
                 "oxford_pets": 88.63, "sun397": 67.62, "ucf101": 70.66},
}
# Ablations of the paper (RN50): Fig. 3 (ImageNet) and Section 4.3 (cross-domain average)
PAPER_ABLATION = {"imagenet": {"clip": 59.81, "pos": 60.82, "neg": 60.83, "tda": 61.35},
                  "cross-domain average": {"pos": 60.38, "neg": 60.11, "tda": 61.03}}
PAPER_EXPERIMENTS = ["paper_cd", "paper_ood", "ablation_components", "ablation_shots"]
N_VERIFY = 500
DEFAULTS = {"mode": "tda", "seed": 1, "order": "official"}


# ======================================================================================
# Small helpers
# ======================================================================================
def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def safe(name):
    return name.replace("/", "-")


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def fmt(x):
    return "-" if x is None else "{:.2f}".format(x)


def flag(delta):
    if delta is None:
        return ""
    a = abs(delta)
    return "✅" if a <= 0.5 else ("⚠️" if a <= 1.0 else "❌")


def run_logged(cmd, log_path, keep=("===", '"acc"', "verify", "Error", "error", "Traceback",
                                    "Processing", "TDA's test accuracy")):
    """Run a command inside the official repo; FULL output -> log_path, informative lines -> screen."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print("$", " ".join(cmd), flush=True)
    print("  full log:", log_path, flush=True)
    env = dict(os.environ, WANDB_MODE="disabled", PYTHONUNBUFFERED="1",
               PYTHONPATH=REPO_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))
    start, tail = time.time(), []
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, env=env)
        for line in p.stdout:
            log.write(line)
            log.flush()
            tail = (tail + [line])[-30:]
            last = line.split("\r")[-1]          # drop tqdm redraws
            if any(k in last for k in keep):
                print("  " + last.rstrip()[:200], flush=True)
        code = p.wait()
    print("  finished in {:.1f} min, exit code {}".format((time.time() - start) / 60, code), flush=True)
    if code != 0:
        print("".join(tail))
        raise RuntimeError("command failed (exit code {}), see {}".format(code, log_path))


# ======================================================================================
# Datasets
# ======================================================================================
def dataset_sizes(dry_run=False):
    """Load every dataset with the OFFICIAL loaders -> {name: size, or None if missing}. Saved to CHECK_FILE."""
    if dry_run and not os.path.isdir(REPO_DIR):
        print("[dry-run] official repo not found: assuming every dataset is on disk")
        return dict(EXPECTED_SIZE)
    sys.path.insert(0, REPO_DIR)
    cwd = os.getcwd()
    os.chdir(REPO_DIR)
    from datasets import build_dataset
    from datasets.imagenet import ImageNet
    key = {"A": "imagenet-a", "V": "imagenet-v", "R": "imagenet-r", "S": "imagenet-s"}
    sizes = {}
    for name in ALL_DATASETS:
        try:
            if name == "I":
                sizes[name] = len(ImageNet(DATA_DIR + "/", None).test)
            else:
                sizes[name] = len(build_dataset(key.get(name, name), DATA_DIR + "/").test)
        except Exception:
            sizes[name] = None
    os.chdir(cwd)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECK_FILE, "w") as f:
        json.dump({"data_dir": DATA_DIR, "sizes": sizes, "expected": EXPECTED_SIZE}, f, indent=2)
    return sizes


def resolve_datasets(spec, sizes):
    """Datasets of an experiment that are on disk, in the right order."""
    def expand(value):
        names = value if isinstance(value, list) else [value]
        out = []
        for n in names:
            out += ORDER[n] if n in ORDER else [n]
        for n in out:
            if n not in ALL_DATASETS:
                raise SystemExit("unknown dataset {!r}; choose from {} or cd/ood".format(n, ALL_DATASETS))
        return out

    wanted = expand(spec["datasets"])
    present = [d for d in wanted if sizes.get(d)]
    if not present and spec.get("fallback"):
        print("[info] none of {} on disk: using the fallback {}".format(wanted, spec["fallback"]))
        wanted = expand(spec["fallback"])
        present = [d for d in wanted if sizes.get(d)]
    missing = [d for d in wanted if not sizes.get(d)]
    if missing:
        print("[info] not on disk, skipped: {}".format(", ".join(missing)))
    return present


# ======================================================================================
# Experiments
# ======================================================================================
def load_experiments():
    with open(EXPERIMENTS_FILE) as f:
        exps = yaml.safe_load(f) or {}
    for name, spec in exps.items():
        if not isinstance(spec, dict) or "datasets" not in spec:
            raise SystemExit("experiments.yaml: experiment {!r} needs at least `datasets:`".format(name))
    return exps


def get_experiment(name):
    exps = load_experiments()
    if name not in exps:
        raise SystemExit("unknown experiment {!r}. Defined in experiments.yaml: {}".format(name, ", ".join(exps)))
    return exps[name]


def expand_grid(grid):
    """grid (dict, or list of dicts) -> list of settings {mode, seed, order, overrides}."""
    grids = grid if isinstance(grid, list) else [grid or {}]
    points = []
    for g in grids:
        keys = list(g)
        for k in keys:
            if k not in DEFAULTS and "." not in k:
                raise SystemExit("grid key {!r}: use mode, seed, order or a config value like "
                                 "positive.alpha".format(k))
        values = [g[k] if isinstance(g[k], list) else [g[k]] for k in keys]
        for combo in itertools.product(*values):
            pt = dict(DEFAULTS, overrides={})
            for k, v in zip(keys, combo):
                if k in DEFAULTS:
                    pt[k] = v
                else:
                    pt["overrides"][k] = v
            if pt not in points:
                points.append(pt)
    return points


def run_key(r):
    return (r["experiment"], r.get("impl", "replica"), r["dataset"], r["backbone"], r.get("mode", "tda"),
            r.get("seed", 1), r.get("order", "official"), json.dumps(r.get("overrides") or {}, sort_keys=True))


def finished():
    return {run_key(r) for r in read_jsonl(RUNS_FILE) if r.get("limit") is None}


def label_of(pt, skip=("seed",)):
    """Human-readable name of a setting, e.g. 'mode=pos positive.shot_capacity=1'."""
    parts = ["{}={}".format(k, pt[k]) for k in ("mode", "order", "seed") if k not in skip and pt[k] != DEFAULTS[k]]
    parts += ["{}={}".format(k, v) for k, v in sorted(pt["overrides"].items())]
    return " ".join(parts) or "default"


def run_official(name, seq, backbone, dry):
    log = os.path.join(LOG_DIR, "{}_official_{}.log".format(name, safe(backbone)))
    cmd = ["python", "tda_runner.py", "--config", "configs", "--wandb-log", "--datasets", seq,
           "--backbone", backbone, "--data-root", DATA_DIR + "/"]
    if dry:
        print("  [dry-run]", " ".join(cmd))
        return
    run_logged(cmd, log)
    accs, current = {}, None              # final accuracy = last value printed for each dataset
    with open(log) as f:
        for line in f:
            m = re.search(r"Processing (\S+) dataset", line)
            if m:
                current = m.group(1)
            m = re.search(r"TDA's test accuracy: (\d+\.\d+)", line)
            if m and current:
                accs[current] = float(m.group(1))
    with open(RUNS_FILE, "a") as f:
        for d, a in accs.items():
            f.write(json.dumps({"experiment": name, "impl": "official", "dataset": d, "backbone": backbone,
                                "mode": "tda", "seed": 1, "order": "official", "overrides": {}, "acc": a,
                                "sequence": seq, "limit": None, "time": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
    print("  official accuracies:", accs, flush=True)


def run_replica(name, seq, backbone, pt, dry, extra=()):
    cmd = ["python", REPLICA, "--config", "configs", "--datasets", seq, "--backbone", backbone,
           "--data-root", DATA_DIR + "/", "--experiment", name, "--mode", pt["mode"], "--seed", str(pt["seed"]),
           "--order", pt["order"], "--out", RUNS_FILE, "--trace-dir", TRACE_DIR]
    for k, v in sorted(pt["overrides"].items()):
        cmd += ["--set", "{}={}".format(k, json.dumps(v))]
    cmd += list(extra)
    if dry:
        print("  [dry-run]", " ".join(cmd))
        return
    tag = re.sub(r"[^A-Za-z0-9_.=-]+", "-", label_of(pt, skip=()))
    run_logged(cmd, os.path.join(LOG_DIR, "{}_{}_{}.log".format(name, safe(backbone), tag)))


def cmd_run(args):
    spec = get_experiment(args.name)
    sizes = dataset_sizes(args.dry_run)
    datasets = resolve_datasets(spec, sizes)
    if not datasets:
        raise SystemExit("No dataset of experiment {!r} is on disk (run the data step first).".format(args.name))
    backbones = [args.backbone] if args.backbone else spec.get("backbones", ["RN50"])
    points = expand_grid(spec.get("grid"))
    done = set() if args.force else finished()
    print("\n== Experiment {} | backbones {} | datasets {} | {} setting(s)".format(
        args.name, backbones, "/".join(datasets), len(points)), flush=True)

    for bb in backbones:
        if spec.get("official"):
            todo = [d for d in datasets if (args.name, "official", d, bb, "tda", 1, "official", "{}") not in done]
            print("\n-- {} | official code: {}".format(bb, "to run" if todo else "already done"), flush=True)
            if todo:
                run_official(args.name, "/".join(datasets), bb, args.dry_run)
        for i, pt in enumerate(points, 1):
            key = lambda d: (args.name, "replica", d, bb, pt["mode"], pt["seed"], pt["order"],
                             json.dumps(pt["overrides"], sort_keys=True))
            todo = [d for d in datasets if key(d) not in done]
            status = "already done" if not todo else "/".join(todo)
            if todo and pt["order"] == "official" and len(todo) < len(datasets):
                status += " (the official stream depends on the earlier datasets: whole sequence re-run)"
            print("\n-- {} | replica [{}/{}] {}: {}".format(bb, i, len(points), label_of(pt, skip=()), status), flush=True)
            if not todo:
                continue
            # the official stream of dataset k depends on datasets 1..k-1: re-run the whole sequence
            seq = datasets if pt["order"] == "official" else todo
            run_replica(args.name, "/".join(seq), bb, pt, args.dry_run)


def cmd_check(args):
    print("\n== 1. Test sets (official loaders) ==")
    sizes = dataset_sizes(args.dry_run)
    for name, n in sizes.items():
        if n is None:
            note = "missing"
        elif n == EXPECTED_SIZE[name]:
            note = "OK"
        else:   # the official loader drops files whose name contains "sh" (e.g. 'plushie_*' in ImageNet-R)
            note = "{:+d} vs paper split ({})".format(n - EXPECTED_SIZE[name], EXPECTED_SIZE[name])
        print("  {:16s} {:>6s}  {}".format(PRETTY[name], str(n or "-"), note))
    available = [d for d, n in sizes.items() if n]
    if not available:
        raise SystemExit("No dataset found: download them first (bash run.sh data).")

    verify_ds = next(d for d in ["dtd", "caltech101", "eurosat", "oxford_pets"] + available if d in available)
    backbones = os.environ.get("BACKBONES", "RN50 ViT-B/16").split()
    print("\n== 2. Equivalence test on the first {} images of {} ==".format(N_VERIFY, PRETTY[verify_ds]))
    for bb in backbones:
        run_replica("verify", verify_ds, bb, dict(DEFAULTS, overrides={}), args.dry_run,
                    extra=["--verify", "--limit", str(N_VERIFY)])
    if args.dry_run:
        return
    rows = [r for r in read_jsonl(RUNS_FILE) if r["experiment"] == "verify" and r["dataset"] == verify_ds]
    rows = list({r["backbone"]: r for r in rows}.values())
    ok = bool(rows) and all(r["verify_pred_mismatches"] == 0 and r["verify_max_abs_logit_diff"] == 0.0 for r in rows)
    for r in rows:
        print("  {:9s} mismatches={}  max|diff|={}".format(r["backbone"], r["verify_pred_mismatches"],
                                                          r["verify_max_abs_logit_diff"]))
    print("\n  RESULT:", "PASS - the re-implementation is identical to the official code" if ok else
          "FAIL - the two implementations differ: do not trust the full runs, look at the logs")
    sys.exit(0 if ok else 1)


def cmd_list(args):
    for name, spec in load_experiments().items():
        n = len(expand_grid(spec.get("grid")))
        print("{:22s} {:3d} setting(s)  {}".format(name, n, spec.get("description", "")))


def cmd_plan(args):
    """For run.sh: one line '<backbone> <time limit>' per job of the experiment."""
    spec = get_experiment(args.name)
    for bb in spec.get("backbones", ["RN50"]):
        print(bb, spec.get("time", args.default_time))


# ======================================================================================
# Report
# ======================================================================================
def cmd_report(args):
    rows = [r for r in read_jsonl(RUNS_FILE) if r.get("limit") is None]
    latest = {}
    for r in rows:                         # later runs overwrite earlier ones
        latest[run_key(r)] = r
    rows = list(latest.values())
    verify = [r for r in read_jsonl(RUNS_FILE) if r["experiment"] == "verify"]
    check = json.load(open(CHECK_FILE)) if os.path.isfile(CHECK_FILE) else None
    try:
        exps = load_experiments()
    except Exception:
        exps = {}
    commit = next((r.get("repo_commit") for r in rows + verify if r.get("repo_commit")), "unknown")
    default = lambda r: r.get("mode", "tda") == "tda" and r.get("seed", 1) == 1 and \
        r.get("order", "official") == "official" and not r.get("overrides")

    L = ["# TDA replication - results report", "",
         "*Generated {} · official code commit `{}` · paper: Karmanov et al., CVPR 2024*".format(
             time.strftime("%Y-%m-%d %H:%M"), commit[:10]), ""]
    csv = ["section,experiment,backbone,dataset,setting,value,std,n_seeds,clip,paper"]

    # ---------------------------------------------------------------- 1. status
    L += ["## 1. Status", ""]
    if check:
        miss = [PRETTY[d] for d, n in check["sizes"].items() if not n]
        L.append("- **Datasets on disk:** {}/15".format(15 - len(miss)) + (" (missing: {})".format(", ".join(miss)) if miss else ""))
    else:
        L.append("- **Datasets:** not checked yet (`bash run.sh check`)")
    last_v = {r["backbone"]: r for r in verify}
    if last_v:
        ok = all(r["verify_pred_mismatches"] == 0 and r["verify_max_abs_logit_diff"] == 0.0 for r in last_v.values())
        L.append("- **Equivalence test (replica vs official code):** {} ({})".format(
            "✅ PASS" if ok else "❌ FAIL", ", ".join("{}: {} mismatches".format(b, r["verify_pred_mismatches"])
                                                   for b, r in last_v.items())))
    else:
        L.append("- **Equivalence test:** not run yet")
    done_exps = sorted({r["experiment"] for r in rows})
    L.append("- **Experiments with results:** " + (", ".join("`{}`".format(e) for e in done_exps) or "none yet"))
    L += ["", "**Legend** (replica − paper): ✅ ≤ 0.5 points (expected noise) · ⚠️ 0.5–1 · ❌ > 1 "
          "(check the dataset). *Replica − official* should always be 0.00.", ""]

    # ---------------------------------------------------------------- 2-3. paper tables
    sec = 2
    for exp, bench, title in [("paper_ood", "ood", "Table 1 - OOD benchmark"),
                              ("paper_cd", "cd", "Table 3 - Cross-domain benchmark")]:
        L += ["## {}. {} (`{}`)".format(sec, title, exp), ""]
        sec += 1
        sel = [r for r in rows if r["experiment"] == exp and default(r)]
        if not sel:
            L += ["Not run yet (`bash run.sh run {}`).".format(exp), ""]
            continue
        acc = {(r.get("impl", "replica"), r["dataset"], r["backbone"]): r for r in sel}
        groups = [("Average (5)", ORDER["ood"]), ("OOD Average (4)", ORDER["ood"][1:])] if bench == "ood" \
            else [("Average (10)", ORDER["cd"])]
        for bb in sorted({r["backbone"] for r in sel}):
            L += ["**CLIP-{}**".format(bb), "",
                  "| Dataset | CLIP | Paper | Official code | Replica | Replica − Paper | Replica − Official | |",
                  "|---|---:|---:|---:|---:|---:|---:|:-:|"]
            get = lambda impl, d, k="acc": acc.get((impl, d, bb), {}).get(k)
            for d in ORDER[bench]:
                pa, o, r, c = PAPER_TDA[bb][d], get("official", d), get("replica", d), get("replica", d, "clip_acc")
                dp = None if r is None else r - pa
                do = None if (r is None or o is None) else r - o
                L.append("| {} | {} | {:.2f} | {} | {} | {} | {} | {} |".format(
                    PRETTY[d], fmt(c), pa, fmt(o), fmt(r), fmt(dp), fmt(do), flag(dp)))
                csv.append("paper,{},{},{},replica,{},,,{},{}".format(exp, bb, d, "" if r is None else r,
                                                                      "" if c is None else c, pa))
                if o is not None:
                    csv.append("paper,{},{},{},official,{},,,,{}".format(exp, bb, d, o, pa))
            for label, ds in groups:
                pa = mean([PAPER_TDA[bb][d] for d in ds])
                rs, os_ = [get("replica", d) for d in ds], [get("official", d) for d in ds]
                r = mean(rs) if None not in rs else None
                o = mean(os_) if None not in os_ else None
                dp = None if r is None else r - pa
                note = "" if r is not None else " (needs all {} datasets)".format(len(ds))
                L.append("| **{}**{} |  | **{:.2f}** | {} | {} | {} | {} | {} |".format(
                    label, note, pa, fmt(o), fmt(r), fmt(dp), fmt(None if (r is None or o is None) else r - o), flag(dp)))
            L.append("")

    # ---------------------------------------------------------------- 4. ablations
    L += ["## {}. Ablations (Fig. 3 and Fig. 4)".format(sec), ""]
    sec += 1
    comp = [r for r in rows if r["experiment"] == "ablation_components"]
    shots = [r for r in rows if r["experiment"] == "ablation_shots"]
    if not comp and not shots:
        L += ["Not run yet (`bash run.sh run ablation_components ablation_shots`).", ""]
    if comp:
        ds_used = [d for d in ALL_DATASETS if d in {r["dataset"] for r in comp}]
        on_in = ds_used == ["I"]
        ref = PAPER_ABLATION["imagenet" if on_in else "cross-domain average"]
        L += ["**Cache components** on {} (CLIP-{}){}".format(
            ", ".join(PRETTY[d] for d in ds_used), "/".join(sorted({r["backbone"] for r in comp})),
            "" if on_in else ". ImageNet not available: mean over these datasets; the paper's numbers are "
                             "the mean over all 10 cross-domain datasets."), "",
              "| Setting | Ours | Paper |", "|---|---:|---:|"]
        names = {"clip": "CLIP only", "pos": "Positive cache only", "neg": "Negative cache only", "tda": "TDA (both)"}
        for m in ["clip", "pos", "neg", "tda"]:
            vals = [r["acc"] for r in comp if r["mode"] == m]
            ours = mean(vals) if len(vals) == len(ds_used) else None
            L.append("| {} | {} | {} |".format(names[m], fmt(ours), fmt(ref.get(m))))
            csv.append("ablation,ablation_components,,{},mode={},{},,,,{}".format(
                "+".join(ds_used), m, "" if ours is None else ours, ref.get(m, "")))
        L.append("")
    if shots:
        ds_used = sorted({r["dataset"] for r in shots})
        curves = {"Positive cache": {}, "Negative cache": {}}
        for r in shots:
            name, k = ("Positive cache", r["overrides"].get("positive.shot_capacity")) if r["mode"] == "pos" \
                else ("Negative cache", r["overrides"].get("negative.shot_capacity"))
            if k is not None:
                curves[name].setdefault(k, []).append(r["acc"])
        curves = {n: {k: mean(v) for k, v in c.items() if len(v) == len(ds_used)} for n, c in curves.items()}
        ks = sorted(set(curves["Positive cache"]) | set(curves["Negative cache"]))
        L += ["**Shot capacity** on {} (paper: best at 3 for the positive cache, 2 for the negative cache)".format(
            ", ".join(PRETTY[d] for d in ds_used)), "",
              "| Shot capacity | " + " | ".join(map(str, ks)) + " |", "|---|" + "---:|" * len(ks)]
        for n, c in curves.items():
            L.append("| {} | ".format(n) + " | ".join(fmt(c.get(k)) for k in ks) + " |")
            for k, v in c.items():
                csv.append("ablation,ablation_shots,,{},{} k={},{},,,,".format("+".join(ds_used), n, k, v))
        L.append("")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for n, c in curves.items():
                if c:
                    ax.plot(sorted(c), [c[k] for k in sorted(c)], marker="o", label=n)
            ax.set_xlabel("Shot capacity")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Shot capacity")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(RESULTS_DIR, "shot_capacity.png"), dpi=150)
            L += ["![Shot capacity](shot_capacity.png)", ""]
        except Exception as e:
            L += ["(plot not created: {})".format(e), ""]

    # ---------------------------------------------------------------- 5. other experiments
    others = [e for e in done_exps if e not in PAPER_EXPERIMENTS + ["verify"]]
    L += ["## {}. Other experiments".format(sec), ""]
    sec += 1
    if not others:
        L += ["None yet. Add one to `experiments.yaml` and run it with `bash run.sh run <name>`.", ""]
    for exp in others:
        sel = [r for r in rows if r["experiment"] == exp and r.get("impl", "replica") == "replica"]
        L += ["### `{}`".format(exp), ""]
        if exps.get(exp, {}).get("description"):
            L += ["*{}*".format(exps[exp]["description"]), ""]
        L += ["Accuracy (%), mean ± std over seeds when there are several (number of seeds in brackets). "
              "*CLIP* = zero-shot CLIP on the same streams.", ""]
        for bb in sorted({r["backbone"] for r in sel}):
            cells, settings = {}, []
            for r in sel:
                if r["backbone"] != bb:
                    continue
                pt = {"mode": r.get("mode", "tda"), "seed": r.get("seed", 1), "order": r.get("order", "official"),
                      "overrides": r.get("overrides") or {}}
                s = label_of(pt)
                if s not in settings:
                    settings.append(s)
                cells.setdefault((r["dataset"], s), []).append(r)
            datasets = [d for d in ALL_DATASETS if any(k[0] == d for k in cells)]
            multi = len(datasets) > 1
            L += ["**CLIP-{}**".format(bb), "",
                  "| Setting | " + " | ".join(PRETTY[d] for d in datasets) + (" | Mean |" if multi else " |"),
                  "|---|" + "---:|" * (len(datasets) + multi)]

            def row(label, values):   # values: {dataset: [accuracies]}
                line, means = "| {} |".format(label), []
                for d in datasets:
                    v = values.get(d, [])
                    if not v:
                        line += " - |"
                        continue
                    means.append(mean(v))
                    line += " {:.2f} |".format(mean(v)) if len(v) == 1 else \
                        " {:.2f} ± {:.2f} ({}) |".format(mean(v), std(v), len(v))
                if multi:
                    line += " **{:.2f}** |".format(mean(means)) if len(means) == len(datasets) else " - |"
                return line

            clip_vals = {d: [r["clip_acc"] for s in settings for r in cells.get((d, s), [])
                             if r.get("clip_acc") is not None and r.get("mode", "tda") != "clip"] for d in datasets}
            L.append(row("*CLIP (zero-shot)*", clip_vals))
            for s in settings:
                vals = {d: [r["acc"] for r in cells.get((d, s), [])] for d in datasets}
                L.append(row(s, vals))
                for d, v in vals.items():
                    if v:
                        c = clip_vals.get(d)
                        csv.append("experiment,{},{},{},{},{:.4f},{:.4f},{},{},".format(
                            exp, bb, d, s, mean(v), std(v), len(v), round(mean(c), 4) if c else ""))
            L.append("")

    # ---------------------------------------------------------------- 6. dataset sizes
    if check:
        L += ["## {}. Test sets".format(sec), "",
              "Sizes found with the official loaders. Smaller ImageNet-A/R/Sketch sets are expected: the "
              "official `datasets/utils.py` skips files whose name contains \"sh\" (e.g. `plushie_*`), "
              "and the replication keeps this behaviour.", "",
              "| Dataset | Found | Paper split |", "|---|---:|---:|"]
        for d, n in check["sizes"].items():
            L.append("| {} | {} | {} |".format(PRETTY[d], n if n else "missing", check["expected"][d]))
        L.append("")

    L += ["---", "Raw results: `runs.jsonl` · per-image traces: `traces/` · this report as a table: `summary.csv` · "
          "plots and stream analysis: `notebooks/analysis.ipynb`", ""]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w") as f:
        f.write("\n".join(csv) + "\n")
    print("Report written:", os.path.join(RESULTS_DIR, "REPORT.md"))
    if not args.quiet:
        print("\n" + "\n".join(L))


# ======================================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("list", help="experiments defined in experiments.yaml")
    s.set_defaults(func=cmd_list)
    s = sub.add_parser("check", help="dataset sizes + equivalence test (GPU)")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_check)
    s = sub.add_parser("run", help="run one experiment of experiments.yaml (GPU)")
    s.add_argument("name")
    s.add_argument("--backbone", choices=["RN50", "ViT-B/16"], help="only this backbone")
    s.add_argument("--force", action="store_true", help="re-run even the settings that are already done")
    s.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    s.set_defaults(func=cmd_run)
    s = sub.add_parser("report", help="rebuild results/REPORT.md")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_report)
    s = sub.add_parser("plan", help="(internal) jobs of an experiment, for run.sh")
    s.add_argument("name")
    s.add_argument("--default-time", default="12:00:00")
    s.set_defaults(func=cmd_plan)
    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
