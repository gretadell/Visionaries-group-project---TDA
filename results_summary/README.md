# Results summary — TDA on sequential streams

One place for every result we have, who produced it, with which settings, and what is still missing.
Last update: 12 November 2026.

| # | Experiment | Who | Where | Status |
|---|---|---|---|---|
| 1 | Reproduction of the paper (Tables 1 and 3) | Greta (code, Colab), Chiara (HPC) | `1_reproduction/` | ✅ done (ImageNet, Stanford Cars, SUN397 missing) |
| 2 | Hyper-parameter search on CORe50 | Chiara | `2_hparam_search/` | ✅ done → **caltech101** frozen |
| 3 | CORe50 128×128 (cropped) | Chiara | `3_core50_128/` | ✅ done, 50 seeds, paired tests (no traces) |
| 4 | CORe50 350×350 (full frame) | Nick | `4_core50_350/` | ✅ done, 50 seeds, paired tests, traces (10 seeds) |
| 5 | iCubWorld Transformations (iCWT) | Nick | `5_icwt/` | ✅ done, 10 seeds with traces, paired tests |
| 6 | UCF101 as a video stream | Greta | `6_ucf101/` | ✅ done, 10 seeds with traces, paired tests |
| — | Stream-51 | — | — | ❌ dropped: official download link is dead, no mirror found |

**Statistics.** Besides the per-mode 95% CI, `paired_tests.py` (Nick) compares modes **seed by seed** (same order and same seed = exactly the same stream: paired bootstrap CI, sign test, paired t-test) and compares orders as **independent samples** (Welch). Files: `*_tests.md` in each folder.

**Traces.** `analyze_traces.py` reads the per-frame traces (`--trace`) and reports cold start, error runs, cache purity, share of the cache from the current session and size of the largest class in the cache. Files: `analysis/<dataset>__<backbone>/` (summary table + 5 plots).

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

**From the traces** (10 seeds with `--trace`, Nick; `4_core50_350/analysis/`):

- **The cache is clean except in temporal order:** 98–99% correct pseudo-labels in S0/S1/S2 with both backbones, 92% (RN50) and 95% (ViT) in S3. So TDA falling below CLIP with RN50 on S0/S1 is **not** caused by wrong labels in the cache.
- **S2 fills the cache with the current session:** 57% of the cache comes from the current session in S2, against 33% (= chance with 3 sessions) in S0.
- **S2 helps most at the start of the stream:** gain over CLIP in the first 5% of frames +10.4 (S2) vs −2.7 (S0) with RN50; +11.6 vs +2.4 with ViT.
- **The cache is perfectly balanced:** the largest class takes 10% of the cache (= 1/10 classes), which explains why M2 has almost nothing to correct.
- **Error runs in S3 come from the video itself:** 78% of TDA errors are in runs of 10+ frames, but CLIP alone is already at 81% (RN50).

---

## 5. iCWT — iCubWorld Transformations (Nick)

Cropped version, left camera, 20 categories, one frame in eight · 2 sessions per object (first and second recording day, "relative" days so that every session contains all categories) · TDA config `caltech101` · **10 seeds with `--trace`** · smoothing windows 5 and 15 (same protocol as UCF101). Files: `5_icwt/`.

| Order | RN50 TDA | RN50 M2 | RN50 Oracle | ViT TDA | ViT M2 | ViT Oracle |
|---|---:|---:|---:|---:|---:|---:|
| *CLIP (all orders)* | *50.15* | | | *65.20* | | |
| S0 | 52.95 ±0.18 | 52.95 ±0.18 | 55.90 ±0.21 | 69.99 ±0.21 | 69.99 ±0.20 | 70.47 ±0.25 |
| S1 ρ=0.5 | 53.20 ±0.34 | 53.20 ±0.34 | 56.08 ±0.33 | 70.32 ±0.22 | 70.32 ±0.22 | 70.84 ±0.23 |
| S1 ρ=0.9 | 53.15 ±0.36 | 53.14 ±0.36 | 56.12 ±0.31 | 70.30 ±0.23 | 70.30 ±0.23 | 70.85 ±0.23 |
| S2 | 52.94 ±0.15 | 52.94 ±0.15 | 56.08 ±0.11 | 70.31 ±0.15 | 70.32 ±0.15 | 70.73 ±0.17 |
| S3 | 52.82 ±0.29 | 52.80 ±0.29 | 56.86 ±0.24 | 70.20 ±0.18 | 70.19 ±0.18 | 71.04 ±0.22 |

Smoothing (CI ≤ ±0.10):

| Window | RN50 S3 | RN50 instance | ViT S3 | ViT instance |
|---|---:|---:|---:|---:|
| 5 | 56.64 | 56.67 | 69.44 | 69.46 |
| 15 | 51.43 | 51.55 | 60.71 | 60.95 |

**From the traces:**

- **Session order does not matter here:** S2 raises the share of the cache from the current session (67% vs 50% = chance with 2 sessions) but accuracy does not change (S2 − S0: −0.01 RN50, +0.32 ViT).
- **The cache is noisier with RN50:** 89% correct pseudo-labels (S0), 86% in S3; ViT 98% and 97%. Accordingly the oracle gains +2.9 to +4.0 points with RN50 and only +0.4 to +0.8 with ViT.
- **Short smoothing beats TDA with RN50:** averaging CLIP over 5 frames gives +6.5 over CLIP, against +2.7 for TDA (10/10 seeds). With ViT TDA stays ahead (−0.76 for smoothing). Longer windows (15) hurt.
- **The cache is perfectly balanced** (largest class = 5% = 1/20 classes), as on CORe50.

## 6. UCF101 as a video stream (Greta)

UCF101 test list 1 · **10 videos per class**, spread over groups (1,010 videos) · **48 consecutive frames at 10 fps** per video (≈5 s around the middle) · 46,294 frames · 101 classes · groups g01–g07 used as sessions (same actor and background) · official UCF101 TDA config (`ucf101`) and prompt *"a photo of a person doing {}."* · **10 seeds with `--trace`** · Colab T4. Files: `6_ucf101/`.

| Order | RN50 TDA | RN50 M2 | RN50 Oracle | ViT TDA | ViT M2 | ViT Oracle |
|---|---:|---:|---:|---:|---:|---:|
| *CLIP (all orders)* | *63.28* | | | *67.45* | | |
| S0 | 66.26 ±0.08 | 66.27 ±0.08 | 70.07 ±0.06 | 71.06 ±0.11 | 71.08 ±0.11 | 74.39 ±0.10 |
| S1 ρ=0.5 | 66.24 ±0.11 | 66.25 ±0.11 | 70.09 ±0.10 | 71.05 ±0.09 | 71.06 ±0.09 | 74.32 ±0.05 |
| S1 ρ=0.9 | 66.09 ±0.09 | 66.07 ±0.09 | 70.17 ±0.11 | 70.96 ±0.11 | 70.95 ±0.11 | 74.46 ±0.07 |
| S2 | 67.66 ±0.37 | 67.67 ±0.38 | 74.54 ±0.50 | 71.92 ±0.25 | 71.96 ±0.25 | 77.94 ±0.69 |
| S3 | 66.66 ±0.35 | 66.58 ±0.37 | 74.41 ±0.59 | 71.21 ±0.18 | 71.21 ±0.18 | 77.82 ±0.39 |

Smoothing (CI ≤ ±0.07):

| Window | RN50 S3 | RN50 instance | ViT S3 | ViT instance |
|---|---:|---:|---:|---:|
| 5 | 63.58 | 63.60 | 66.88 | 66.89 |
| 15 | 58.70 | 58.73 | 61.14 | 61.13 |

**From the traces** (RN50; ViT is similar):

- **S2 doubles the share of the cache coming from the current session** (33% vs 16% in S0), and the gain is largest at the start of the stream (+7.5 vs +1.8 points over CLIP in the first 5% of frames).
- **Cache purity drops in temporal order:** 87% correct pseudo-labels in S0, 81% in S3.
- **Error runs in S3 come mostly from the video itself:** 72% of TDA errors are in runs of 10+ frames, but CLIP alone is already at 69%.

---

## Key findings so far

Differences from the paired tests (seed by seed for modes, Welch for orders). n.s. = 95% CI includes 0. (x/N) = seeds in which the first term is better.

| Comparison | CORe50 128 RN50 | CORe50 128 ViT | CORe50 350 RN50 | CORe50 350 ViT | iCWT RN50 | iCWT ViT | UCF101 RN50 | UCF101 ViT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TDA − CLIP, S0 | +5.98 | +5.64 | **−1.75** (0/50) | +3.16 | +2.80 | +4.79 | +2.98 | +3.61 |
| TDA: S2 − S0 | +0.83 | +1.27 | **+4.93** | **+4.23** | −0.01 n.s. | +0.32 | +1.41 | +0.87 |
| TDA: S3 − S2 | −0.51 | −0.59 | −0.38 n.s. | −0.19 n.s. | −0.12 n.s. | −0.11 n.s. | −1.00 | −0.72 |
| TDA: S1 (ρ=0.5) − S0 | −0.00 n.s. | +0.06 n.s. | −0.04 n.s. | +0.11 n.s. | +0.25 n.s. | +0.33 | −0.02 n.s. | −0.01 n.s. |
| M2 − TDA, S2 | +0.01 | +0.00 | +0.08 | +0.26 | +0.00 | +0.00 | +0.01 | +0.03 |
| Oracle − TDA, S3 | +1.61 | +0.80 | +0.73 | +3.54 | +4.04 | +0.84 | +7.76 | +6.62 |
| best Smooth − TDA, S3 | −3.12 | −5.09 | −0.98 (23/50) | −5.62 | **+3.82** (10/10) | −0.76 | −3.08 | −4.32 |

Seeds: 50 on CORe50, 10 on iCWT and UCF101.

1. **TDA beats CLIP almost everywhere** (+2.7 to +7.4 points, in all or almost all seeds). The one exception is RN50 on CORe50 350 with shuffled streams (S0, S1), where TDA is **below CLIP in all 50 seeds**. Since the cache is 99% correct there, the cause is not label noise: more likely the cache retrieves neighbours by background rather than by object.
2. **Session context helps, when sessions look different.** S2 beats S0 on CORe50 (about +1 on the 128 crops, **+4–5 on the 350 full frames**) and on UCF101 (+0.9 to +1.4); on iCWT it is ≈ 0 (−0.01 RN50, +0.32 ViT). In every dataset S2 fills the cache with the current session (CORe50 350: 57% vs 33%; UCF101: 33% vs 16%; iCWT: 67% vs 50%), but this only pays off where the background changes between sessions. On iCWT the two sessions are two recording days of the same setup, so they probably look alike (hypothesis to check).
3. **Label persistence (S1) does not matter**, and **temporal order (S3) is never better than S2**: slightly worse on CORe50 128 and UCF101 (−0.5 to −1.0), not significant elsewhere.
4. **M2 has (almost) nothing to correct.** The traces show a perfectly balanced cache (largest class = 1/number of classes). M2 − TDA is within ±0.1 everywhere, except a small systematic gain on CORe50 350 S2 (+0.26 with ViT).
5. **Cache label noise matters in temporal order and on harder datasets.** Purity drops in S3 (CORe50 350: 99% → 92–95%; UCF101: 87% → 81%; iCWT RN50: 89% → 86%), and the oracle gains most in S3, on UCF101 (+6.6 to +7.8) and on iCWT with RN50 (+3 to +4). Open point: the oracle is **below** TDA with RN50 on CORe50 350 S0/S1/S2 (−2 to −3), which label noise cannot explain since TDA's cache is already 99% correct.
6. **TDA does not create long error runs:** in S3 the share of errors in runs of 10+ frames is about the same for TDA and CLIP (CORe50 350: 78% vs 81%; UCF101: 72% vs 69%). Error runs come from the video itself.
7. **Video smoothing is a strong baseline only with short windows on iCWT.** TDA beats smoothing on CORe50 and UCF101 (all seeds except CORe50 350 RN50), but on iCWT with RN50 averaging CLIP over 5 frames beats TDA by +3.8 (10/10 seeds). Windows of 15+ frames are worse than 5 on iCWT and UCF101.
8. **S2 helps most at the start of the stream (cold start):** on CORe50 350 the gain over CLIP in the first 5% of frames is +10.4 in S2 against −2.7 in S0 (RN50); on UCF101 +7.5 against +1.8.

## Still to do

- [x] iCWT: final runs, paired tests, trace analysis (Nick)
- [x] UCF101: final runs, paired tests, trace analysis (Greta)
- [x] Paired tests on all four datasets (`paired_tests.py`)
- [x] Traces on CORe50 350 (Nick, 10 seeds)
- [ ] Traces on CORe50 128 (10 seeds with `--trace`), to have the cache analysis on both resolutions (Chiara)
- [ ] Context subtraction (Greta's proposal): code, UCF101 pilot, then CORe50 and iCWT if the pilot works
- [ ] Explain why the oracle is below TDA with RN50 on CORe50 350 (S0/S1/S2)
- [ ] Explain why short smoothing beats TDA with RN50 on iCWT
- [ ] iCWT full frame: now useful to test finding 2 (does adding the background bring the session effect back?)
- [ ] Decide whether to add ImageNet / Stanford Cars to the reproduction (ask the professor)
- [ ] Write the report

## Extension ideas (to discuss)

0. **Context subtraction (proposed by Greta, in progress):** keep a running average of the last frames' features, which estimates the current context (background, lighting), and subtract it before the cache lookup. No labels, no training, one extra hyper-parameter (length of the average). Caveat: in S3 the recent frames are mostly the same object, so the average may estimate the object rather than the background.
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
