# Per-seed comparisons: core50_350

Source: `core50_350/feature_runs.jsonl`. Bootstrap: 10000 resamples. t-test p-values: scipy.

## RN50 — config caltech101 — frames sessions 3,7,10

| Comparison | Order | Type | Seeds | Diff (pts) | 95% CI | A>B / A<B / = | p sign | p t |
|---|---|---|---:|---:|---|---|---:|---:|
| ctxx-w300 [context.strength=0.5] − TDA | s0 | paired | 10 | +3.80 | [+3.71, +3.89] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s0 | paired | 10 | +1.89 | [+1.70, +2.09] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s0 | paired | 10 | -1.91 | [-2.12, -1.70] | 0 / 10 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s2 | paired | 10 | +6.08 | [+5.70, +6.43] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s2 | paired | 10 | +9.34 | [+8.52, +10.18] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s2 | paired | 10 | +3.27 | [+2.16, +4.35] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s3 | paired | 10 | +3.25 | [+2.66, +3.90] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s3 | paired | 10 | +6.55 | [+5.46, +7.54] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s3 | paired | 10 | +3.30 | [+1.81, +4.66] | 8 / 2 / 0 | 0.109 | 0.002 |
| TDA: S2 − S0 |  | unpaired | 10 vs 10 | +5.17 | [+4.04, +6.32] |  |  | <0.001 |
| TDA: S3 − S2 |  | unpaired | 10 vs 10 | +0.04 | [-1.75, +1.82] |  |  | 0.969 |
| TDA: S3 − S0 |  | unpaired | 10 vs 10 | +5.21 | [+3.74, +6.58] |  |  | <0.001 |

## ViT-B/16 — config caltech101 — frames sessions 3,7,10

| Comparison | Order | Type | Seeds | Diff (pts) | 95% CI | A>B / A<B / = | p sign | p t |
|---|---|---|---:|---:|---|---|---:|---:|
| ctxx-w300 [context.strength=0.5] − TDA | s0 | paired | 10 | +2.18 | [+2.02, +2.31] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s0 | paired | 10 | +4.87 | [+4.54, +5.18] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s0 | paired | 10 | +2.69 | [+2.25, +3.11] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s2 | paired | 10 | +4.69 | [+3.95, +5.42] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s2 | paired | 10 | +11.65 | [+11.18, +12.14] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s2 | paired | 10 | +6.96 | [+5.90, +8.06] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − TDA | s3 | paired | 10 | +2.52 | [+1.69, +3.31] | 10 / 0 / 0 | 0.002 | <0.001 |
| ctxx-w300 [context.strength=0.5] − CLIP | s3 | paired | 10 | +8.58 | [+8.13, +9.07] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA − CLIP | s3 | paired | 10 | +6.06 | [+4.94, +7.29] | 10 / 0 / 0 | 0.002 | <0.001 |
| TDA: S2 − S0 |  | unpaired | 10 vs 10 | +4.27 | [+3.10, +5.48] |  |  | <0.001 |
| TDA: S3 − S2 |  | unpaired | 10 vs 10 | -0.90 | [-2.50, +0.68] |  |  | 0.312 |
| TDA: S3 − S0 |  | unpaired | 10 vs 10 | +3.37 | [+2.18, +4.67] |  |  | <0.001 |
