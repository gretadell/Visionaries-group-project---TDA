# Understanding Test-Time Adaptation on Sequential Visual Streams

**Visionaries group** · Chiaradonna · Dell'Amico · Mayda · Stavropoulou · Bocconi University, Computer Vision 2026

Training-free test-time adaptation (TTA) methods such as **TDA** ([Karmanov et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Karmanov_Efficient_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2024_paper.html))
adapt CLIP at inference with a cache of confident past test samples, but they are evaluated on shuffled test sets.
Cameras and robots see sequences instead. This repository contains everything behind our report:

1. an **exact re-implementation of TDA** (same predictions as the [official code](https://github.com/kdiAAA/TDA)) and the reproduction of the paper's Tables 1 and 3;
2. a **sequential benchmark**: four stream orders (S0 shuffled, S1 class persistence, S2 session grouping, S3 session + video order) on CORe50 (crops and full frames), iCubWorld Transformations and UCF101, two CLIP backbones, 10-50 seeds, paired statistical tests;
3. **cache diagnostics**: per-frame traces, a class-balanced read-out (M2), an oracle cache and a measurement of which cache entries drive each vote;
4. **context subtraction (CTXX)**, our label-free, training-free correction that removes an online estimate of the scene from the cache features.

**Main results.** Only session grouping helps TDA (up to +4.9 points). On shuffled full-frame CORe50, TDA falls below zero-shot
CLIP although 99% of its cached labels are correct: its read-out votes by session rather than by object. CTXX improves TDA there by
+2.2 to +6.1 points in every order and with both backbones, and costs about one point where the background is part of the class.
All numbers, tables and plots: [`results_summary/`](results_summary/README.md).

---

## Repository layout

The code is a flat set of scripts, grouped here by pipeline stage. Every script has a docstring at the top with its full usage.

| Stage | File | What it does |
|---|---|---|
| **Data** | `prepare_data.py` | downloads and arranges the datasets of the paper (reproduction) |
| | `extract_features.py` | computes CLIP image features once and saves them (CORe50 128 / 350, DTD, EuroSAT); equivalence check against images |
| | `extract_icwt.py` | same for iCubWorld Transformations (sessions = relative recording day) |
| | `extract_ucf101_video.py` | same for UCF101 as video: 48 consecutive frames at 10 fps per test video, read straight from the zip |
| **Method** | `tda_replica.py` | TDA re-implemented from scratch (modes `tda`, `pos`, `neg`, `clip`, `m2`, `oracle`) |
| | `context_cache.py` | context subtraction: modes `ctx-w<W>` / `ctxx-w<W>`, `--strength` |
| **Experiments** | `experiments.py`, `experiments.yaml` | reproduction of the paper on images (Tables 1 and 3) and the report `results/REPORT.md` |
| | `streams.py` | TDA on saved features under stream orders S0-S3 (+ Stream-51 style orders), many seeds; smoothing baseline; summary table |
| | `select_context_params.py` | selection of the CTXX parameters on the **non-test** CORe50 sessions |
| **Evaluation & analysis** | `paired_tests.py` | seed-by-seed comparisons (paired bootstrap CI, sign test, t-test; Welch across orders) |
| | `analyze_traces.py` | per-frame trace analysis: cold start, error runs, cache purity, session share, class balance |
| | `retrieval_analysis.py` | which cache entries drive the vote (same session vs same class), TDA vs CTXX |
| **Running** | `run.sh`, `config.sh` | cluster entry point (SLURM): `bash run.sh <command>`; `config.sh` holds account, partition, paths |
| | `retrieval_job.sh` | SLURM job for the retrieval analysis |
| | `notebooks/` | Colab versions: `colab.ipynb` (reproduction), `streams_colab.ipynb` (stream runs from saved features), `ucf101_video_colab.ipynb` (UCF101 end to end), `analysis.ipynb` (plots, no GPU) |
| **Results** | `results_summary/` | every result of the report: tables (CSV / Markdown), paired tests, trace plots, one folder per experiment |

Created at run time and not versioned: `TDA/` (official repository, cloned by the setup), `results/` (raw runs, features, traces),
`logs/`, datasets.

---

## Setup

**Cluster (SLURM, recommended).** Needs conda (or a `miniconda3` module), git, a GPU partition, about 25 GB of disk.

```bash
git clone https://github.com/gretadell/Visionaries-group-project---TDA.git TDA_replication
cd TDA_replication
nano config.sh            # set SLURM_ACCOUNT and SLURM_PARTITION (Bocconi: partition and qos "stud")
bash run.sh setup         # conda env "tda" (Python 3.7, PyTorch 1.12.1), official repo in TDA/, CLIP weights
bash run.sh check         # GPU job: dataset sizes + equivalence test (replica vs official code must PASS)
```

**Colab.** Open a notebook from `notebooks/`, choose a T4 GPU and *Run all*. The notebooks clone this repository and use the
same scripts; results are written to `MyDrive/TDA_replication/results/`. The stream experiments only need the saved feature
files (`results/features/<dataset>__<backbone>.pt`), not the datasets.

---

## Reproducing the report

All stream experiments run on saved features, so a run takes seconds to minutes. Finished runs are skipped, so every command
can be relaunched after an interruption. `RN50` can be replaced by `ViT-B/16`.

| Report | Command | Output in `results_summary/` |
|---|---|---|
| Sec. 5.1, Tabs. 2, A1, A2: reproduction | `bash run.sh data` then `bash run.sh all` | `1_reproduction/` |
| Tab. A3: TDA config for CORe50 | `python streams.py run --dataset core50 --backbone RN50 --cfg-like <config> --sessions 1,2,4,5,6,8,9,11 --every 5 --order s0 --mode tda --repeats 5` (one run per official config) | `README.md`, section 2 |
| Features | `bash run.sh features core50 --root <CORe50 128 folder>`; `... --name core50_350` for the full frames; `bash run.sh features-icwt`; UCF101 in `notebooks/ucf101_video_colab.ipynb` | — |
| Sec. 5.2, Tab. 3, A7, A8: stream orders | `python streams.py run --dataset core50_350 --backbone RN50 --cfg-like caltech101 --sessions 3,7,10 --order s0,s1,s2,s3 --rho 0.5,0.9 --mode tda,clip,m2,oracle --repeats 50` (iCWT / UCF101: 10 repeats, `--cfg-like ucf101` for UCF101) | `3_core50_128/`, `4_core50_350/`, `5_icwt/`, `6_ucf101/` |
| Smoothing baseline (Tab. A6) | same command with `--mode smooth --window 5,15,30 --order s3,instance` | same folders |
| Sec. 5.3, Tab. 4, Fig. 3: traces | add `--trace` (10 seeds), then `python analyze_traces.py --dataset core50_350 --backbone RN50` | `*/analysis/` |
| Paired tests | `python paired_tests.py results/feature_runs.jsonl --dataset core50_350 --config caltech101 --out tests.md` | `*_tests.md`, `*/context_final/paired_ctxx_*.md` |
| Sec. 5.4, Tab. 5: CTXX selection | `python select_context_params.py` (non-test sessions only) | `7_context_selection/` |
| Sec. 5.4, Fig. 4, Tabs. A4, A5: CTXX | `python context_cache.py --dataset core50_350 --backbone RN50 --cfg-like caltech101 --sessions 3,7,10 --order s0,s2,s3 --mode ctxx --window 300 --strength 0.5 --repeats 10` | `*/context_final/` |
| Tab. A9: retrieval analysis | `sbatch retrieval_job.sh` (or `python retrieval_analysis.py run ...` then `python retrieval_analysis.py summary`) | `8_retrieval_check/` |

Tables of any set of runs: `python streams.py summary --dataset <name>` (mean and 95% CI over seeds).

---

## Notes

- **Equivalence.** `bash run.sh check` compares `tda_replica.py` with the official functions on the same 500 images; every
  extension keeps `--mode tda` as the original method, so the check stays valid after changes.
- **Same streams for every method.** A stream depends only on (seed, dataset, order), so methods are compared seed by seed.
- **No tuning on test data.** The TDA configuration and the CTXX parameters were chosen on CORe50 sessions 1, 2, 4, 5, 6, 8, 9,
  11 and frozen; the test sessions are 3, 7 and 10.
- **Windows line endings.** After copying files from Windows run `sed -i 's/\r$//' *.sh *.py` on the cluster.
- **Datasets not used.** ImageNet, Stanford Cars and SUN397 (reproduction) and Stream-51 (download link no longer available).

## Credits

TDA method and official code: Karmanov, Guan, Lu, El Saddik, Xing ([kdiAAA/TDA](https://github.com/kdiAAA/TDA)).
CLIP: [openai/CLIP](https://github.com/openai/CLIP). Datasets: CORe50 (Lomonaco & Maltoni, CC BY 4.0), iCubWorld Transformations
(Pasquale et al.), UCF101 (Soomro et al.), CoOp splits. Datasets, weights and the official code are downloaded from their sources
and not redistributed here.
