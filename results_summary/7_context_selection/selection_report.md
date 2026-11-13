# Context-subtraction selection (validation only)

CORe50 350 / RN50 / caltech101 / non-test sessions 1,2,4,5,6,8,9,11
Seeds 1-3; all available frames, no subsampling. No test accuracies used in ranking.

Score = equally weighted mean of the S0/S2/S3 mean paired gains over TDA.
Differences and SDs are percentage points. Rank uses unrounded values.

| Rank | Candidate | S0 gain | S2 gain | S3 gain | Score |
|---:|---|---:|---:|---:|---:|
| 1 | ctxx-w300-strength0.5 | +3.4269 | +7.7736 | +3.2417 | +4.8141 |
| 2 | ctx-w300-strength0.5 | +3.2925 | +7.8583 | +2.0519 | +4.4009 |
| 3 | ctx-w300-strength1.0 | +3.5883 | +6.3997 | +0.7333 | +3.5738 |
| 4 | ctxx-w300-strength1.0 | +3.7267 | +6.1831 | +0.7414 | +3.5504 |
| 5 | ctxx-wall-strength0.5 | +3.4117 | +1.8017 | +0.9361 | +2.0498 |
| 6 | ctx-wall-strength0.5 | +3.2875 | +1.6528 | +0.8422 | +1.9275 |
| 7 | ctxx-wall-strength1.0 | +3.7247 | +1.1764 | -0.0164 | +1.6282 |
| 8 | ctx-wall-strength1.0 | +3.5900 | +0.9700 | -0.0800 | +1.4933 |

Selected: **ctxx-w300-strength0.5** (score +4.8141).

## Selected candidate: individual paired differences

| Order | Seed | Context | TDA | Context - TDA | Context - CLIP |
|---|---:|---:|---:|---:|---:|
| s0 | 1 | 55.6017 | 51.9783 | +3.6233 | -1.3717 |
| s0 | 2 | 56.1267 | 52.8700 | +3.2567 | -0.8467 |
| s0 | 3 | 55.7525 | 52.3517 | +3.4008 | -1.2208 |
| s2 | 1 | 63.8417 | 57.0675 | +6.7742 | +6.8683 |
| s2 | 2 | 62.9225 | 55.0233 | +7.8992 | +5.9492 |
| s2 | 3 | 62.8100 | 54.1625 | +8.6475 | +5.8367 |
| s3 | 1 | 62.4042 | 61.0092 | +1.3950 | +5.4308 |
| s3 | 2 | 61.8950 | 58.3608 | +3.5342 | +4.9217 |
| s3 | 3 | 61.1367 | 56.3408 | +4.7958 | +4.1633 |

## Diagnostic contrasts (descriptive)

Positive numbers favour the first option. These are not causal tests or significance claims.

- ctx strength 0.5: cumulative-all minus W300 score = -2.4734.
- ctx strength 1.0: cumulative-all minus W300 score = -2.0805.
- ctxx strength 0.5: cumulative-all minus W300 score = -2.7643.
- ctxx strength 1.0: cumulative-all minus W300 score = -1.9221.
- S3, W300 strength 0.5: ctxx minus ctx = +1.1897.
- S3, W300 strength 1.0: ctxx minus ctx = +0.0081.
- S3, Wall strength 0.5: ctxx minus ctx = +0.0939.
- S3, Wall strength 1.0: ctxx minus ctx = +0.0636.

## Interpretation and provenance

- Three selection seeds are exploratory; no significance-based selection is performed.
- Equal order weighting is fixed in advance. Inspect per-order regressions, not just the mean.
- Window all is the cumulative mean of past frames, not a future-aware global mean.
- Comparing all and 300 probes cumulative versus recent centering; it does not prove background removal.
- ctxx uses pseudo-labels to exclude the current pseudo-class; its effect does not isolate objects perfectly.
- Two strengths (0.5/1.0, W300) were previously explored on test sessions. Disclose this pilot.
- The selected config must transfer unchanged to ViT, CORe50 128, iCWT and UCF101.
- No final test runs have been executed by this script.
- manifest.json records feature SHA256, source SHA256, runtime and resolved TDA config.
- Existing shared results were not reused; only identical-provenance checkpoints are resumed.
