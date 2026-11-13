# Per-seed comparisons: icwt

Source: `icwt/feature_runs.jsonl`. Bootstrap: 10000 resamples. t-test p-values: scipy.

## RN50 — config caltech101 — frames all

| Comparison | Order | Type | Seeds | Diff (pts) | 95% CI | A>B / A<B / = | p sign | p t |
|---|---|---|---:|---:|---|---|---:|---:|
| ctxx-w300 [context.strength=0.5] − TDA | s0 | paired | 10 | +0.45 | [+0.34, +0.54] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s0 | paired | 10 | +3.25 | [+3.13, +3.37] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s0 | paired | 10 | +2.80 | [+2.63, +2.98] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s2 | paired | 10 | +0.50 | [+0.43, +0.56] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s2 | paired | 10 | +3.29 | [+3.18, +3.39] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s2 | paired | 10 | +2.79 | [+2.65, +2.93] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s3 | paired | 10 | +0.40 | [+0.27, +0.54] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s3 | paired | 10 | +3.07 | [+2.90, +3.25] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s3 | paired | 10 | +2.67 | [+2.39, +2.95] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA: S2 − S0 |  | unpaired | 10 vs 10 | -0.01 | [-0.24, +0.20] |  |  | 0.914 |
| TDA: S3 − S2 |  | unpaired | 10 vs 10 | -0.12 | [-0.43, +0.19] |  |  | 0.486 |
| TDA: S3 − S0 |  | unpaired | 10 vs 10 | -0.13 | [-0.46, +0.20] |  |  | 0.461 |

## ViT-B/16 — config caltech101 — frames all

| Comparison | Order | Type | Seeds | Diff (pts) | 95% CI | A>B / A<B / = | p sign | p t |
|---|---|---|---:|---:|---|---|---:|---:|
| ctxx-w300 [context.strength=0.5] − TDA | s0 | paired | 10 | -0.93 | [-1.02, -0.84] | 0 / 10 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s0 | paired | 10 | +3.86 | [+3.74, +3.98] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s0 | paired | 10 | +4.79 | [+4.60, +4.99] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s2 | paired | 10 | -1.02 | [-1.09, -0.95] | 0 / 10 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s2 | paired | 10 | +4.09 | [+4.00, +4.19] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s2 | paired | 10 | +5.11 | [+4.97, +5.26] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s3 | paired | 10 | -1.05 | [-1.15, -0.94] | 0 / 10 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s3 | paired | 10 | +3.95 | [+3.85, +4.05] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s3 | paired | 10 | +5.00 | [+4.84, +5.17] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA: S2 − S0 |  | unpaired | 10 vs 10 | +0.32 | [+0.08, +0.56] |  |  | 0.025 |
| TDA: S3 − S2 |  | unpaired | 10 vs 10 | -0.11 | [-0.33, +0.11] |  |  | 0.353 |
| TDA: S3 − S0 |  | unpaired | 10 vs 10 | +0.21 | [-0.06, +0.46] |  |  | 0.146 |
