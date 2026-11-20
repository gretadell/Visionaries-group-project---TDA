# Per-seed comparisons: core50

Source: `results/context_final_20261112_191902/core50/feature_runs.jsonl`. Bootstrap: 10000 resamples. t-test p-values: scipy.

## RN50 — config caltech101 — frames sessions 3,7,10

| Comparison | Order | Type | Seeds | Diff (pts) | 95% CI | A>B / A<B / = | p sign | p t |
|---|---|---|---:|---:|---|---|---:|---:|
| ctxx-w300 [context.strength=0.5] − TDA | s0 | paired | 10 | -1.37 | [-1.43, -1.31] | 0 / 10 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s0 | paired | 10 | +4.60 | [+4.54, +4.68] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s0 | paired | 10 | +5.97 | [+5.87, +6.07] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s2 | paired | 10 | -0.91 | [-1.00, -0.79] | 0 / 10 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s2 | paired | 10 | +6.04 | [+5.87, +6.18] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s2 | paired | 10 | +6.94 | [+6.76, +7.11] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s3 | paired | 10 | -2.29 | [-2.53, -2.05] | 0 / 10 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s3 | paired | 10 | +4.05 | [+3.78, +4.31] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s3 | paired | 10 | +6.33 | [+6.10, +6.58] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA: S2 − S0 |  | unpaired | 10 vs 10 | +0.98 | [+0.77, +1.17] |  |  | <0.001 |
| TDA: S3 − S2 |  | unpaired | 10 vs 10 | -0.61 | [-0.90, -0.31] |  |  | 0.001 |
| TDA: S3 − S0 |  | unpaired | 10 vs 10 | +0.37 | [+0.11, +0.63] |  |  | 0.023 |

## ViT-B/16 — config caltech101 — frames sessions 3,7,10

| Comparison | Order | Type | Seeds | Diff (pts) | 95% CI | A>B / A<B / = | p sign | p t |
|---|---|---|---:|---:|---|---|---:|---:|
| ctxx-w300 [context.strength=0.5] − TDA | s0 | paired | 10 | -1.11 | [-1.18, -1.05] | 0 / 10 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s0 | paired | 10 | +4.42 | [+4.30, +4.54] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s0 | paired | 10 | +5.53 | [+5.37, +5.69] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s2 | paired | 10 | -0.97 | [-1.05, -0.89] | 0 / 10 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s2 | paired | 10 | +6.07 | [+5.72, +6.42] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s2 | paired | 10 | +7.04 | [+6.69, +7.43] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s3 | paired | 10 | -1.98 | [-2.16, -1.77] | 0 / 10 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s3 | paired | 10 | +4.57 | [+4.16, +4.98] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s3 | paired | 10 | +6.55 | [+6.06, +7.00] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA: S2 − S0 |  | unpaired | 10 vs 10 | +1.52 | [+1.12, +1.93] |  |  | <0.001 |
| TDA: S3 − S2 |  | unpaired | 10 vs 10 | -0.50 | [-1.12, +0.09] |  |  | 0.146 |
| TDA: S3 − S0 |  | unpaired | 10 vs 10 | +1.02 | [+0.50, +1.50] |  |  | 0.003 |
