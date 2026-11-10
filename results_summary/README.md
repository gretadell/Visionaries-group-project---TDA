# Results summary — TDA on sequential streams

One place for every result we have, who produced it, with which settings, and what is still missing.
Last update: 10 November 2026.

| # | Experiment | Who | Where | Status |
|---|---|---|---|---|
| 1 | Reproduction of the paper (Tables 1 and 3) | Greta (code, Colab), Chiara (HPC) | `1_reproduction/` | ✅ done (ImageNet, Stanford Cars, SUN397 missing) |
| 2 | Hyper-parameter search on CORe50 | Chiara | `2_hparam_search/` | ✅ done → **caltech101** frozen |
| 3 | CORe50 128×128 (cropped) | Chiara | `3_core50_128/` | ✅ done (no traces) |
| 4 | CORe50 350×350 (full frame) | Nick | `4_core50_350/` | ✅ done (no traces) |
| 5 | iCubWorld Transformations (iCWT) | Nick | `5_icwt/` | ⏳ in progress |
| 6 | UCF101 as a video stream | Greta | `6_ucf101/` | ⏳ in progress |
| — | Stream-51 | — | — | ❌ dropped: official download link is dead, no mirror found |

**Common settings (CORe50, iCWT):** TDA on pre-extracted CLIP features (`extract_features.py`, `streams.py`), backbones RN50 and ViT-B/16, TDA config `caltech101`, accuracy = top-1 (%), ± = 95% CI over seeds.

**Stream orders:** S0 everything shuffled (as in the paper) · S1 class persists with probability ρ (0.5, 0.9) · S2 one session at a time, frames shuffled inside · S3 one session at a time, object videos in temporal order · instance (used for smoothing).

**Methods:** CLIP zero-shot · TDA · M2 (cache votes with the per-class mean instead of the sum) · Oracle (cache filled with true labels) · Smooth-wN (CLIP averaged over the last N frames).

---

## 1. Reproduction of the paper

Official code (commit `e697fb0c80`) + our re-implementation `tda_replica.py`, which gives **identical predictions** to the official code on every dataset (Replica − Official = 0.00). Bocconi HPC, A100, Python 3.7, PyTorch 1.12.1, official configs, single stream order as in the paper.

### Table 3 — cross-domain (accuracy %, Δ = ours − paper)

| Dataset | RN50 ours | RN50 paper | Δ | ViT ours | ViT paper | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Caltech101 | 89.53 | 89.70 | −0.17 | 94.04 | 94.24 | −0.20 |
| DTD | 43.56 | 43.74 | −0.18 | 45.04 | 47.40 | **−2.36** |
| EuroSAT | 41.07 | 42.11 | **−1.04** | 63.06 | 58.00 | **+5.06** |
| FGVC Aircraft | 17.22 | 17.61 | −0.39 | 25.02 | 23.91 | **+1.11** |
| Food101 | 77.80 | 77.75 | +0.05 | 86.18 | 86.14 | +0.04 |
| Flowers102 | 67.88 | 68.74 | −0.86 | 71.46 | 71.42 | +0.04 |
| Oxford Pets | 85.85 | 86.18 | −0.33 | 89.64 | 88.63 | **+1.01** |
| UCF101 | 63.73 | 64.18 | −0.45 | 71.05 | 70.66 | +0.39 |
| Stanford Cars | — | 57.78 | — | — | 67.28 | — |
| SUN397 | — | 62.53 | — | — | 67.62 | — |

Colab (Greta, RN50) and HPC agree within 0.25 points.

### Table 1 — out-of-distribution (accuracy %)

| Dataset | RN50 ours | RN50 paper | Δ | ViT ours | ViT paper | Δ |
|---|---:|---:|---:|---:|---:|---:|
| ImageNet-A | 30.95 | 30.29 | +0.66 | 60.30 | 60.11 | +0.19 |
| ImageNet-V2 | 55.23 | 55.54 | −0.31 | 64.45 | 64.67 | −0.22 |
| ImageNet-R | 62.82 | 62.58 | +0.24 | 80.41 | 80.24 | +0.17 |
| ImageNet-Sketch | 38.02 | 38.12 | −0.10 | 50.93 | 50.54 | +0.39 |
| **OOD average (4)** | **46.76** | **46.63** | **+0.13** | **64.02** | **63.89** | **+0.13** |
| ImageNet | — | 61.35 | — | — | 69.51 | — |

**Takeaways.** Table 1 is reproduced with both backbones (all within 0.7 points). Table 3 is reproduced with RN50; with ViT, 4 datasets deviate by 1–5 points in both directions. Since the official code gives the same numbers, the likely causes are the single stream order used in the paper and/or the published hyper-parameters (not checked yet).

**Missing:** ImageNet (needs an image-net.org account; also needed for the ablations in Fig. 3–4), Stanford Cars (official links dead, Kaggle copy possible), SUN397 (37 GB, over the 50 GB cluster quota).

---

## 2. Hyper-parameter search (CORe50 128)

CORe50 has no official TDA config, so we tried the 10 official configs on the **8 non-test sessions** (1, 2, 4, 5, 6, 8, 9, 11), one frame in five, order S0, 5 seeds. Test sessions 3, 7, 10 were never used for the choice.

| Config | RN50 | ViT-B/16 |
|---|---:|---:|
| **caltech101 (chosen)** | **80.47 ±0.18** | **87.74 ±0.20** |
| eurosat | 80.23 ±0.13 | 87.20 ±0.18 |
| ucf101 | 80.04 ±0.10 | 86.70 ±0.17 |
| dtd | 79.84 ±0.12 | 86.16 ±0.21 |
| sun397 | 79.84 ±0.12 | 86.16 ±0.21 |
| oxford_pets | 79.79 ±0.11 | 86.14 ±0.20 |
| fgvc | 79.73 ±0.10 | 85.86 ±0.19 |
| oxford_flowers | 79.41 ±0.06 | 85.32 ±0.18 |
| stanford_cars | 79.32 ±0.06 | 85.24 ±0.17 |
| food101 | 79.02 ±0.10 | 84.64 ±0.13 |
| *CLIP alone* | *77.03* | *80.37* |

caltech101 is first with both backbones (clearly with ViT; nearly tied with eurosat for RN50). The config is worth 1.5 points (RN50) and 3.1 points (ViT).

---

## 3. CORe50 128×128 (cropped around the object)

Test sessions 3, 7, 10 · 44,972 frames · 50 objects · 10 classes · 50 seeds.

| Order | RN50 TDA | RN50 M2 | RN50 Oracle | ViT TDA | ViT M2 | ViT Oracle |
|---|---:|---:|---:|---:|---:|---:|
| *CLIP (all orders)* | *73.37* | | | *79.17* | | |
| S0 | 79.35 ±0.06 | 79.35 ±0.06 | 79.22 ±0.08 | 84.81 ±0.08 | 84.81 ±0.08 | 85.07 ±0.11 |
| S1 ρ=0.5 | 79.35 ±0.07 | 79.35 ±0.07 | 79.26 ±0.09 | 84.87 ±0.09 | 84.87 ±0.09 | 84.98 ±0.11 |
| S1 ρ=0.9 | 79.31 ±0.07 | 79.31 ±0.07 | 79.28 ±0.09 | 84.84 ±0.09 | 84.84 ±0.09 | 84.99 ±0.11 |
| S2 | 80.18 ±0.11 | 80.19 ±0.11 | 80.21 ±0.12 | 86.08 ±0.17 | 86.09 ±0.17 | 85.82 ±0.14 |
| S3 | 79.68 ±0.13 | 79.62 ±0.13 | 81.29 ±0.21 | 85.49 ±0.22 | 85.45 ±0.22 | 86.29 ±0.21 |

Smoothing (CLIP averaged over the last N frames, CI ≤ ±0.03):

| Window | RN50 S3 | RN50 instance | ViT S3 | ViT instance |
|---|---:|---:|---:|---:|
| 5 | 74.86 | 74.86 | 79.73 | 79.73 |
| 15 | 75.60 | 75.63 | 80.40 | 80.41 |
| 30 | 76.56 | 76.61 | 80.02 | 80.05 |

---

## 4. CORe50 350×350 (full frame, with background)

Same protocol as section 3 · 44,999 frames (one corrupted image in the original archive excluded: `s3/o43/C_03_43_209.png`) · 50 seeds.

| Order | RN50 TDA | RN50 M2 | RN50 Oracle | ViT TDA | ViT M2 | ViT Oracle |
|---|---:|---:|---:|---:|---:|---:|
| *CLIP (all orders)* | *51.54* | | | *59.88* | | |
| S0 | 49.79 ±0.13 | 49.80 ±0.13 | 46.79 ±0.15 | 63.03 ±0.21 | 63.06 ±0.21 | 65.26 ±0.21 |
| S1 ρ=0.5 | 49.76 ±0.11 | 49.76 ±0.11 | 47.06 ±0.12 | 63.15 ±0.22 | 63.17 ±0.22 | 65.35 ±0.17 |
| S1 ρ=0.9 | 49.76 ±0.11 | 49.75 ±0.11 | 47.08 ±0.12 | 63.13 ±0.22 | 63.14 ±0.22 | 65.38 ±0.17 |
| S2 | 54.73 ±0.53 | 54.81 ±0.53 | 52.76 ±0.39 | 67.27 ±0.48 | 67.53 ±0.46 | 68.74 ±0.67 |
| S3 | 54.34 ±0.66 | 54.38 ±0.66 | 55.08 ±0.51 | 67.07 ±0.53 | 67.20 ±0.51 | 70.61 ±0.75 |

Smoothing (CI ≤ ±0.03):

| Window | RN50 S3 | RN50 instance | ViT S3 | ViT instance |
|---|---:|---:|---:|---:|
| 5 | 52.47 | 52.48 | 60.71 | 60.72 |
| 15 | 53.35 | 53.37 | 61.38 | 61.40 |
| 30 | 53.36 | 53.40 | 61.46 | 61.49 |

---

## 5. iCWT — in progress (Nick)

iCubWorld Transformations, cropped version, left camera, 20 categories, 2 relative sessions per object (first/second recording day). Protocol agreed for the two confirmation datasets (iCWT, UCF101): `--every 8`, **10 seeds with `--trace`**, smoothing windows 5 and 15. Full-frame version (~23 GB) only if needed for the background effect.

## 6. UCF101 — in progress (Greta)

UCF101 as a video stream on Colab, clips of 48 frames at 10 fps, same protocol as iCWT.

---

## Key findings so far

1. **TDA helps a lot on cropped CORe50:** about +6 points over CLIP with both backbones, for every order.
2. **Session context is what helps TDA.** S2 beats S0 by 0.8–1.3 points on the 128 crops and by **4–5 points on the 350 full frames**. S3 ≈ S2 and S1 ≈ S0: what matters is seeing one session (background, lighting) at a time, not the persistence of the object or of the label.
3. **The background hurts CLIP a lot:** about −20 points from 128 to 350. On 350 with RN50, TDA on shuffled streams (S0, S1) is even **below CLIP** (49.8 vs 51.5).
4. **M2 changes nothing** anywhere (max difference +0.26): the coverage bias it targets does not show up on CORe50.
5. **Cache label noise matters only sometimes.** The oracle is close to TDA on the 128 crops except in S3 (+1.6 RN50, +0.8 ViT), clearly above TDA with ViT on 350 (+1.5 to +3.5), and *below* TDA with RN50 on 350 S0/S1/S2 (to be understood).
6. **TDA beats video smoothing** clearly: smoothing gains at most +3.2 (RN50, 128) over CLIP.

## Still to do

- [ ] iCWT: extraction, final runs, summary, trace analysis (Nick)
- [ ] UCF101: final runs and trace analysis (Greta)
- [ ] CORe50 traces: sections 3 and 4 were run **without** `--trace`. Re-run with 10 seeds and `--trace` if we want cache-dynamics plots for CORe50 (Chiara for 128, Nick for 350?)
- [ ] Explain the oracle results: why is the oracle below TDA in some settings?
- [ ] Paired confidence intervals (bootstrap on seed-wise differences) for order comparisons, e.g. S2 − S0
- [ ] Decide whether to add ImageNet / Stanford Cars to the reproduction (ask the professor)
- [ ] Decide on iCWT full frame
- [ ] Write the report

## Extension ideas (to discuss)

1. **Session-aware cache:** since session context drives the gains, try resetting the cache at each session change, or one cache per session.
2. **Temporally consistent cache insertion:** add a frame to the cache only if neighbouring frames agree on the prediction, to keep repeated errors out in video streams (S3), where the oracle shows label noise costs accuracy. Close to Nick's M3.
3. **When does TDA hurt?** On 350 RN50 S0, TDA is below CLIP: test a stricter confidence (entropy) threshold for cache insertion.
4. **Background vs object:** 128 vs 350 isolates the role of background; a background-masked or center-cropped variant could confirm it.

---

## Reproducing on the Bocconi cluster (notes)

- Login: `ssh <student ID>@slogin.hpc.unibocconi.it` (VPN GlobalProtect if the Wi-Fi does not reach the cluster).
- `config.sh`: `SLURM_ACCOUNT="<student ID>"`, `SLURM_PARTITION="stud"`, `SLURM_QOS="stud"`, `MEM="64G"`.
- After every `git pull`: `sed -i 's/\r$//' *.sh *.py *.yaml` (Windows line endings) and set 24 h instead of 48 h in `experiments.yaml`.
- Before `bash run.sh setup`: `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` (same for `/pkgs/r`).
- Environment: `module load miniconda3` then `eval "$(conda shell.bash hook)" && conda activate tda`.
- Limits: 24 h per job, one job running at a time, at most 4 jobs queued per user, 50 GB home quota.
