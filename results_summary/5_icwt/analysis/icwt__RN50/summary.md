| Mode | Order | Seeds | acc | clip | fixed | broken | start_gain | rest_gain | err_run | err_in_long | purity | same_session | same_object | top_class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip | s0 | 10 | 50.15 | 50.15 | 0.00 | 0.00 | 0.00 | 0.00 | 1.99 | 1.11 |  |  |  |  |
| clip | s1 rho=0.5 | 10 | 50.15 | 50.15 | 0.00 | 0.00 | 0.00 | 0.00 | 2.19 | 3.21 |  |  |  |  |
| clip | s1 rho=0.9 | 10 | 50.15 | 50.15 | 0.00 | 0.00 | 0.00 | 0.00 | 2.38 | 11.14 |  |  |  |  |
| clip | s2 | 10 | 50.15 | 50.15 | 0.00 | 0.00 | 0.00 | 0.00 | 1.99 | 0.98 |  |  |  |  |
| clip | s3 | 10 | 50.15 | 50.15 | 0.00 | 0.00 | 0.00 | 0.00 | 3.79 | 42.11 |  |  |  |  |
| m2 | s0 | 10 | 52.95 | 50.15 | 7.14 | 4.33 | 4.87 | 2.70 | 1.89 | 0.60 | 88.68 | 49.97 | 0.07 | 5.01 |
| m2 | s1 rho=0.5 | 10 | 53.20 | 50.15 | 7.40 | 4.34 | 4.92 | 2.96 | 2.04 | 1.71 | 89.02 | 49.98 | 0.07 | 5.02 |
| m2 | s1 rho=0.9 | 10 | 53.14 | 50.15 | 7.36 | 4.37 | 4.16 | 2.93 | 2.20 | 7.01 | 88.61 | 49.98 | 0.07 | 5.03 |
| m2 | s2 | 10 | 52.94 | 50.15 | 7.10 | 4.31 | 5.10 | 2.67 | 1.88 | 0.55 | 88.65 | 66.94 | 0.09 | 5.01 |
| m2 | s3 | 10 | 52.80 | 50.15 | 7.04 | 4.39 | 4.92 | 2.53 | 3.80 | 41.90 | 86.19 | 65.49 | 0.31 | 5.05 |
| oracle | s0 | 10 | 55.90 | 50.15 | 9.73 | 3.98 | 10.66 | 5.49 | 1.79 | 0.35 | 100.00 | 49.98 | 0.07 | 5.01 |
| oracle | s1 rho=0.5 | 10 | 56.08 | 50.15 | 9.88 | 3.95 | 11.04 | 5.66 | 1.94 | 1.34 | 100.00 | 49.97 | 0.07 | 5.02 |
| oracle | s1 rho=0.9 | 10 | 56.12 | 50.15 | 9.93 | 3.95 | 12.70 | 5.62 | 2.09 | 6.19 | 100.00 | 49.98 | 0.08 | 5.08 |
| oracle | s2 | 10 | 56.08 | 50.15 | 9.92 | 3.99 | 11.20 | 5.65 | 1.78 | 0.31 | 100.00 | 66.48 | 0.09 | 5.01 |
| oracle | s3 | 10 | 56.86 | 50.15 | 10.58 | 3.87 | 18.74 | 6.07 | 3.45 | 36.07 | 100.00 | 65.23 | 0.33 | 5.17 |
| tda | s0 | 10 | 52.95 | 50.15 | 7.13 | 4.33 | 4.81 | 2.70 | 1.89 | 0.61 | 88.68 | 49.97 | 0.07 | 5.01 |
| tda | s1 rho=0.5 | 10 | 53.20 | 50.15 | 7.39 | 4.34 | 4.89 | 2.96 | 2.04 | 1.72 | 89.02 | 49.98 | 0.07 | 5.02 |
| tda | s1 rho=0.9 | 10 | 53.15 | 50.15 | 7.36 | 4.36 | 4.33 | 2.93 | 2.20 | 7.01 | 88.61 | 49.98 | 0.07 | 5.03 |
| tda | s2 | 10 | 52.94 | 50.15 | 7.09 | 4.31 | 5.08 | 2.67 | 1.88 | 0.55 | 88.65 | 66.94 | 0.09 | 5.01 |
| tda | s3 | 10 | 52.82 | 50.15 | 7.04 | 4.37 | 5.24 | 2.53 | 3.80 | 41.93 | 86.19 | 65.49 | 0.31 | 5.05 |

Column notes (all in %, mean over seeds, except err_run):
- acc / clip: accuracy of the mode and of CLIP alone on the same stream.
- fixed: frames CLIP gets wrong and the mode gets right; broken: CLIP right, mode wrong. acc - clip = fixed - broken.
- start_gain / rest_gain: (mode - CLIP) accuracy in the first 5% of the stream / in the rest (cold start).
- err_run: mean length of a run of consecutive errors; err_in_long: share of errors inside runs of 10 or more
  (long runs = an error that keeps repeating, e.g. a wrong pseudo-label propagating along a video).
- purity: share of samples in the positive cache whose pseudo-label is the true label.
- same_session / same_object: share of the positive cache coming from the session / the object (video) of the current frame.
- top_class: share of the positive cache taken by its largest class.
