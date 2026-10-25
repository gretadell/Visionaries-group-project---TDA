# TDA replication: Efficient Test-Time Adaptation of Vision-Language Models

A reproduction of **Karmanov et al., *Efficient Test-Time Adaptation of Vision-Language Models*, CVPR 2024**
([paper](https://openaccess.thecvf.com/content/CVPR2024/html/Karmanov_Efficient_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2024_paper.html),
[official code](https://github.com/kdiAAA/TDA)) on CLIP ResNet-50 and ViT-B/16, built to be **extended** with
new experiments, stream orders and variants of the method.

It runs **two implementations side by side**:

1. the **official code** of the authors, unmodified, cloned automatically;
2. an **independent re-implementation**, [`tda_replica.py`](tda_replica.py), checked prediction by prediction
   against the official code.

It reproduces **Table 1** (out-of-distribution), **Table 3** (cross-domain) and the **ablations of Fig. 3 and 4**,
and every experiment you add to [`experiments.yaml`](experiments.yaml) ends up in the same report.

> **Status:** code complete and tested offline (equivalence of the method with the official functions on
> synthetic data, job submission with a simulated SLURM, report and analysis on simulated results). The full GPU
> runs have not been executed yet, so no result table is published here.

---

## Contents

- [Repository layout: which file does what](#repository-layout-which-file-does-what)
- [Option A: SLURM cluster (recommended)](#option-a-slurm-cluster-recommended)
- [Option B: Google Colab](#option-b-google-colab)
- [Experiments](#experiments)
- [Extending the method](#extending-the-method)
- [Results and analysis](#results-and-analysis)
- [All commands](#all-commands)
- [Datasets](#datasets)
- [How the method works](#how-the-method-works)
- [Troubleshooting](#troubleshooting)
- [Expected deviations from the paper](#expected-deviations-from-the-paper)
- [Credits](#credits)

---

## Repository layout: which file does what

```
TDA_replication/
├── README.md
├── config.sh                cluster settings (SLURM account/partition, paths, resources)
├── run.sh                   the only command on the cluster: bash run.sh <command>
├── experiments.yaml         WHAT to run: one block per experiment
├── experiments.py           runs the experiments, builds results/REPORT.md
├── tda_replica.py           the METHOD (re-implementation of TDA)
├── prepare_data.py          downloads and arranges the datasets
└── notebooks/
    ├── colab.ipynb          runs the same files on Google Colab
    └── analysis.ipynb       plots and stream analysis (no GPU needed)
```

| I want to... | Edit |
|---|---|
| set up the cluster (account, partition, where datasets go) | `config.sh`, once |
| run a new experiment (other datasets, seeds, stream orders, hyper-parameters) | `experiments.yaml` only |
| change or extend the method (new cache rule, new stream order, new variant) | `tda_replica.py` |
| change the report or add a table | `experiments.py` (section *Report*) |
| make plots or analyse what happens along the stream | `notebooks/analysis.ipynb` |

Created when you run it, and ignored by git: `TDA/` (official repository), `results/` (everything to look at),
`logs/` (full logs), and the datasets folder (`~/tda_data` by default).

You do not need Linux on your own computer: everything runs on the cluster (Linux) or on Colab. From your
computer you only use `ssh` and `scp`, which work the same in Windows PowerShell, macOS and Linux.

---

## Option A: SLURM cluster (recommended)

You need: an account on a Linux cluster with SLURM and NVIDIA GPUs, `conda` (or a `miniconda3` module), `git`,
internet access from the login node, and about **25 GB** of disk (60 GB with SUN397).

### Step 1. Connect and clone

```bash
ssh <username>@<cluster-login-address>          # e.g. ssh 1234567@slogin.unibocconi.it
git clone https://github.com/gretadell/Visionaries-group-project---TDA.git TDA_replication
cd TDA_replication
```

Some university clusters accept connections only from the campus network or the university VPN: if `ssh`
hangs, turn on the VPN. If the repository is private, `git clone` asks for your GitHub username and a
[personal access token](https://github.com/settings/tokens) instead of the password.

### Step 2. Set your SLURM account and GPU partition

```bash
sacctmgr show assoc user=$USER format=Account,Partition,QOS -P    # -> your account
sinfo -o "%P %G %l"                                              # -> a partition whose GRES column shows gpu:...
nano config.sh          # replace the two CHANGE_ME, then Ctrl+O, Enter, Ctrl+X
```

In `config.sh` you can also change where datasets are stored (`DATA_DIR`: use a scratch/work folder if your home
quota is small) and the resources of each job.

### Step 3. One-time setup (about 10 minutes)

```bash
bash run.sh setup
```

Creates the conda environment `tda` with the authors' versions (Python 3.7, PyTorch 1.12.1, CUDA 11.3), clones
the official repository into `TDA/` and downloads the CLIP weights. Safe to run again.

### Step 4. Download the datasets (1-3 hours, in the background)

```bash
bash run.sh data          # starts in the background: you can close the terminal
bash run.sh data-log      # watch progress (Ctrl+C stops watching, not the download)
```

12 of the 15 datasets download automatically; SUN397 (37 GB) with `bash run.sh data --with-sun397`; ImageNet and
Stanford Cars by hand (see [Datasets](#datasets)). Every experiment runs on the datasets that are present.

### Step 5. Quick check (one GPU job, about 15 minutes)

```bash
bash run.sh check
bash run.sh status        # wait until the job leaves the queue
cat results/REPORT.md
```

Checks the test-set sizes and runs the **equivalence test**: the official code and the re-implementation on the
same 500 images must give identical predictions. The report must show `✅ PASS`.

### Step 6. Reproduce the paper

```bash
bash run.sh all
```

Submits the check and then the four paper experiments (`paper_cd`, `paper_ood`, `ablation_components`,
`ablation_shots`), one GPU job per backbone; they start only if the check succeeds. `results/REPORT.md` is
rebuilt at the end of every job.

### Step 7. Your own experiments

```bash
bash run.sh list                  # experiments defined in experiments.yaml
bash run.sh run seeds stream_order
```

See [Experiments](#experiments) to add one.

### Step 8. Copy the results to your computer

```bash
# on YOUR computer, not on the cluster
scp -r <username>@<cluster-login-address>:~/TDA_replication/results ./tda_results
```

---

## Option B: Google Colab

1. Open [`notebooks/colab.ipynb`](notebooks/colab.ipynb) in Colab:
   `https://colab.research.google.com/github/gretadell/Visionaries-group-project---TDA/blob/main/notebooks/colab.ipynb`
2. *Runtime → Change runtime type → GPU*.
3. In the *Settings* cell set `REPO_URL`, the datasets and the experiments, then *Runtime → Run all*.

The notebook clones this repository and runs the **same files** (`prepare_data.py`, `experiments.py`,
`experiments.yaml`), so there is nothing to keep in sync. Results go to `MyDrive/TDA_replication/results/` on
Google Drive. Colab uses its own PyTorch version and sessions last a few hours: fine for the cross-domain
datasets, too short for the OOD benchmark. If the repository is private, `REPO_URL` needs a token (see the
notebook).

---

## Experiments

Every experiment is a block of [`experiments.yaml`](experiments.yaml). For example:

```yaml
stream_order:
  description: i.i.d. stream vs images grouped by class
  datasets: [dtd, eurosat, oxford_pets]     # or cd / ood = the whole benchmark
  backbones: [RN50]
  grid:                                     # every combination is one run
    order: [random, sorted]
    seed: [1, 2, 3]
  time: "08:00:00"                          # SLURM time limit per job
```

Run it with `bash run.sh run stream_order` (cluster) or `python experiments.py run stream_order` (Colab).
Preview what would run, without a GPU: `python experiments.py run stream_order --dry-run`.

What a grid can vary:

| Key | Values | Meaning |
|---|---|---|
| `mode` | `tda`, `pos`, `neg`, `clip` | full TDA, positive cache only, negative cache only, zero-shot CLIP |
| `seed` | integers | random seed (1 = the paper's) |
| `order` | `official`, `random`, `sorted` | order of the test stream, see below |
| any config value | e.g. `positive.shot_capacity`, `positive.alpha`, `positive.beta`, `negative.alpha`, `negative.entropy_threshold.lower`, `negative.mask_threshold.lower` | overrides the official `configs/<dataset>.yaml` |

A list of grids runs each of them (see `ablation_shots`). Stream orders:

- **`official`**: the authors' shuffled stream. Only this one reproduces the paper's numbers. With it, each dataset's
  stream depends on the datasets before it in the run (shared random generator), as in the official scripts.
- **`random`**: a new shuffle for every seed that depends only on (seed, dataset), so results do not change if you
  add or remove other datasets. Use it for multi-seed studies.
- **`sorted`**: images grouped by class (all of class 0, then class 1, ...): a non-i.i.d. stream, the hard case for an
  online cache.

Finished runs are never repeated: if a job hits its time limit, or you add a value to a grid, submit the same
experiment again and only the missing runs are executed (`--force` re-runs everything). Every run also records
CLIP's accuracy on the same stream and a per-image trace for the [analysis notebook](#results-and-analysis).

`experiments.yaml` already contains three examples besides the paper: `seeds` (variance over 5 shuffles),
`stream_order` (i.i.d. vs sorted) and `alpha_beta` (sensitivity to the positive-cache hyper-parameters).

---

## Extending the method

All of the method is in [`tda_replica.py`](tda_replica.py), and its header lists where to plug in a change:

| Extension | Where |
|---|---|
| a new stream order (class blocks, increasing difficulty, ...) | `make_stream()` + add the name to `ORDERS` |
| a different cache policy or read-out | `DynamicCache`, or a new class used in `run_tda()` |
| a new variant of the method | add a name to `MODES` and handle it in `run_tda()` |
| a hyper-parameter | no code: a grid key in `experiments.yaml` |

A new mode or order is immediately usable in `experiments.yaml` (`mode: [tda, myvariant]`), appears in the report
next to TDA and CLIP, and gets a trace for the analysis notebook.

After changing `tda_replica.py`, run `bash run.sh check`: the equivalence test still compares the default `tda` mode
with the official code, so it tells you whether the original method is still exact.

---

## Results and analysis

Everything is written to `results/` (on Colab: `MyDrive/TDA_replication/results/`):

| File | Content |
|---|---|
| **`REPORT.md`** | status, Table 1 and Table 3 (CLIP / paper / official code / replica / differences, ✅ ⚠️ ❌), ablations, a table for every other experiment, test-set sizes |
| `summary.csv` | the numbers of the report in long format, for Excel or a thesis |
| `shot_capacity.png` | reproduction of Fig. 4 |
| `runs.jsonl` | raw results: one JSON line per run and dataset (accuracy, CLIP accuracy, all settings) |
| `traces/*.csv.gz` | for every run and dataset, one row per test image: true class, CLIP and TDA predictions, entropy, cache updates and sizes |
| `dataset_check.json` | test-set sizes found on disk |

How to read the report: *Replica − Official* must be **0.00** everywhere; *Replica − Paper* within ±0.5 (✅) is the
normal noise of a different GPU and library versions. Above 1 point (❌), first check that dataset's test-set size.

[`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) reads `results/` and needs no GPU (only `pandas` and
`matplotlib`), so it also runs on your computer after `scp`. It contains:

1. an overview of all experiments;
2. running accuracy of CLIP and TDA along the stream, and the gain in a moving window;
3. cache dynamics: how the caches fill and how often they are updated;
4. images fixed and broken by TDA, by CLIP's entropy;
5. effect of stream order and seeds (spread over seeds, i.i.d. vs sorted);
6. per-class gains;
7. heatmaps for two-parameter grids.

Figures are saved in `results/figures/`.

---

## All commands

On the cluster, from the repository folder:

| Command | Runs on | What it does |
|---|---|---|
| `bash run.sh setup` | login node | conda env, official repo, CLIP weights (once) |
| `bash run.sh data [names]` | login node, background | download and arrange datasets (safe to re-run) |
| `bash run.sh data-log` | login node | follow the download log |
| `bash run.sh check` | GPU job | test-set sizes + equivalence test |
| `bash run.sh all` | GPU jobs | check, then the four paper experiments |
| `bash run.sh run NAME [NAME...]` | GPU jobs | experiments of `experiments.yaml`, one job per backbone |
| `bash run.sh list` | login node | experiments defined in `experiments.yaml` |
| `bash run.sh status` | login node | your queue + last progress line of each log |
| `bash run.sh report` | login node | rebuild `results/REPORT.md` |
| `bash run.sh cancel` | login node | cancel all TDA jobs |

The same steps without SLURM (Colab, or an interactive GPU session): `python prepare_data.py`,
`python experiments.py check`, `python experiments.py run NAME`, `python experiments.py report`.

---

## Datasets

Datasets follow the official [`docs/DATASETS.md`](https://github.com/kdiAAA/TDA/blob/main/docs/DATASETS.md) and the
CoOp `split_zhou_*.json` splits.

| Benchmark | Downloaded automatically | Manual |
|---|---|---|
| Cross-domain (Table 3) | Caltech101, DTD, EuroSAT, FGVC Aircraft, Food101, Flowers102, OxfordPets, UCF101, SUN397 (optional, 37 GB) | Stanford Cars |
| OOD (Table 1) | ImageNet-A, ImageNet-V2, ImageNet-R, ImageNet-Sketch | ImageNet val |

Names in `experiments.yaml`: `I A V R S caltech101 dtd eurosat fgvc food101 oxford_flowers oxford_pets
stanford_cars sun397 ucf101`, or `cd` / `ood` for a whole benchmark. For the download step (`bash run.sh data
<names>`) the ImageNet family is called `imagenet imagenet-a imagenet-v imagenet-r imagenet-s`.

**Manual datasets.** Put the files in `~/tda_data/manual/` (Colab: `MyDrive/TDA_replication/manual/`) and run the
data step again.

| Dataset | Files | Where to get them |
|---|---|---|
| ImageNet val | `ILSVRC2012_img_val.tar`, `ILSVRC2012_devkit_t12.tar.gz` | free account at [image-net.org](https://image-net.org/download-images.php) |
| Stanford Cars | `stanford_cars/cars_test.tgz`, `stanford_cars/cars_test_annos_withlabels.mat` (+ optional `car_devkit.tgz`) | the original links are dead; a Kaggle mirror of the original files |

```bash
scp ILSVRC2012_img_val.tar ILSVRC2012_devkit_t12.tar.gz <username>@<cluster-login-address>:~/tda_data/manual/
```

Some downloads come from mirrors (Caltech101 from data.caltech.edu, EuroSAT and ImageNet-Sketch from Hugging Face).
If one fails, `logs/prepare_data.log` says which file to place by hand and where.

---

## How the method works

TDA adapts CLIP **at test time, without backpropagation**. Test images arrive one at a time:

1. **Zero-shot CLIP.** `logits = 100 · f · W_c`, with `f` the normalized image feature and `W_c` the class text
   embeddings averaged over prompt templates.
2. **Multi-view (OOD datasets only).** 63 AugMix views + the original; keep the 10% with the lowest entropy and
   average their features and logits.
3. **Positive cache.** One queue per predicted class, capacity `k`. When full, a new sample replaces the
   highest-entropy entry if it has lower entropy.
4. **Negative cache.** Samples with normalized entropy in (0.2, 0.5), stored with their probability map; capacity 2.
5. **Final logits.** `clip + A_pos(f·Q_p)·L_p − A_neg(f·Q_n)·mask`, with `A(z) = α·exp(−β(1 − z))` and
   `mask` = 1 for classes with probability in (0.03, 1.0).

Eq. 4 and Eq. 6 of the paper contain a double minus sign for the negative cache; this repository follows the
official code (subtract the negative term). The Colab notebook explains the method in detail, with formulas.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: open config.sh and set SLURM_ACCOUNT...` | Step 2 |
| `ERROR: run 'bash run.sh setup' first.` | Step 3 |
| `conda not found` | the cluster has no `miniconda3` module: install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) in your home, then `bash run.sh setup` |
| `sbatch: error: invalid account or account/partition combination` | check the values with `sacctmgr show assoc user=$USER` |
| `unknown experiment '...'` | the name is not a block of `experiments.yaml` (`bash run.sh list`) |
| `--set ...: '...' is not in the config` | a grid key has a typo; valid keys are in the official `TDA/configs/<dataset>.yaml` |
| `$'\r': command not found` | Windows line endings (files copied by hand instead of `git clone`): `sed -i 's/\r$//' *.sh` |
| A dataset is `NOT ready` in the download summary | read the error in `logs/prepare_data.log`; Google Drive quota errors go away after a few hours |
| Job pending for a long time (`Resources` / `Priority`) | normal on a shared cluster; shorter `time:` limits can help |
| Job ended with `TIMEOUT` | raise `time:` of that experiment and submit it again: finished runs are skipped |

---

## Expected deviations from the paper

Even the official code does not give exactly the published numbers on a different machine. Expect about
**±0.1-0.5 points per dataset**, a little more on the OOD datasets; averages are more stable.

- TDA is online, so accuracy depends on the order of the stream. Same seed → same order, but a different GPU or
  PyTorch version can change fp16 rounding and the shuffle.
- ImageNet-A/V2/R/Sketch use random AugMix views generated by 8 parallel workers: not bit-reproducible.
- If a dataset is missing, the official streams of the datasets after it change.
- The official loader skips files whose name contains `"sh"` (e.g. ImageNet-R `plushie_*` images), so some test sets
  are slightly smaller than expected. This quirk is kept on purpose.

---

## Credits

- Method and official code: Adilbek Karmanov, Dayan Guan, Shijian Lu, Abdulmotaleb El Saddik, Eric Xing,
  [kdiAAA/TDA](https://github.com/kdiAAA/TDA).
- CLIP: [openai/CLIP](https://github.com/openai/CLIP). Datasets and splits: [CoOp](https://github.com/KaiyangZhou/CoOp).

The official code, the CLIP weights and the datasets are downloaded from their sources and are **not**
redistributed here; they keep their own licenses.

```bibtex
@inproceedings{karmanov2024efficient,
  title     = {Efficient Test-Time Adaptation of Vision-Language Models},
  author    = {Karmanov, Adilbek and Guan, Dayan and Lu, Shijian and El Saddik, Abdulmotaleb and Xing, Eric},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```
