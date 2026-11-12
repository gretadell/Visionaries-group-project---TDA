| Mode | Order | Seeds | acc | clip | fixed | broken | start_gain | rest_gain | err_run | err_in_long | purity | same_session | same_object | top_class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip | s0 | 10 | 65.20 | 65.20 | 0.00 | 0.00 | 0.00 | 0.00 | 1.54 | 0.03 |  |  |  |  |
| clip | s1 rho=0.5 | 10 | 65.20 | 65.20 | 0.00 | 0.00 | 0.00 | 0.00 | 1.68 | 0.38 |  |  |  |  |
| clip | s1 rho=0.9 | 10 | 65.20 | 65.20 | 0.00 | 0.00 | 0.00 | 0.00 | 1.83 | 4.44 |  |  |  |  |
| clip | s2 | 10 | 65.20 | 65.20 | 0.00 | 0.00 | 0.00 | 0.00 | 1.53 | 0.07 |  |  |  |  |
| clip | s3 | 10 | 65.20 | 65.20 | 0.00 | 0.00 | 0.00 | 0.00 | 3.02 | 31.42 |  |  |  |  |
| m2 | s0 | 10 | 69.99 | 65.20 | 7.22 | 2.43 | 5.06 | 4.78 | 1.43 | 0.03 | 97.93 | 49.96 | 0.08 | 5.01 |
| m2 | s1 rho=0.5 | 10 | 70.32 | 65.20 | 7.54 | 2.42 | 5.23 | 5.11 | 1.51 | 0.05 | 98.39 | 49.97 | 0.08 | 5.02 |
| m2 | s1 rho=0.9 | 10 | 70.29 | 65.20 | 7.52 | 2.43 | 5.11 | 5.09 | 1.60 | 0.60 | 98.09 | 49.98 | 0.08 | 5.04 |
| m2 | s2 | 10 | 70.32 | 65.20 | 7.51 | 2.40 | 5.56 | 5.09 | 1.42 | 0.02 | 98.18 | 67.13 | 0.10 | 5.01 |
| m2 | s3 | 10 | 70.19 | 65.20 | 7.65 | 2.67 | 4.97 | 4.98 | 2.66 | 23.57 | 96.78 | 65.80 | 0.32 | 5.08 |
| oracle | s0 | 10 | 70.47 | 65.20 | 7.87 | 2.61 | 6.71 | 5.19 | 1.42 | 0.01 | 100.00 | 49.93 | 0.08 | 5.01 |
| oracle | s1 rho=0.5 | 10 | 70.84 | 65.20 | 8.19 | 2.55 | 6.86 | 5.57 | 1.52 | 0.09 | 100.00 | 49.93 | 0.08 | 5.02 |
| oracle | s1 rho=0.9 | 10 | 70.85 | 65.20 | 8.19 | 2.55 | 7.93 | 5.52 | 1.62 | 2.00 | 100.00 | 49.93 | 0.09 | 5.08 |
| oracle | s2 | 10 | 70.73 | 65.20 | 8.12 | 2.59 | 7.33 | 5.43 | 1.41 | 0.01 | 100.00 | 67.22 | 0.10 | 5.01 |
| oracle | s3 | 10 | 71.04 | 65.20 | 8.65 | 2.81 | 12.86 | 5.47 | 2.65 | 23.79 | 100.00 | 64.98 | 0.34 | 5.17 |
| tda | s0 | 10 | 69.99 | 65.20 | 7.22 | 2.43 | 4.97 | 4.78 | 1.43 | 0.03 | 97.93 | 49.96 | 0.08 | 5.01 |
| tda | s1 rho=0.5 | 10 | 70.32 | 65.20 | 7.53 | 2.42 | 5.14 | 5.11 | 1.51 | 0.05 | 98.39 | 49.97 | 0.08 | 5.02 |
| tda | s1 rho=0.9 | 10 | 70.30 | 65.20 | 7.52 | 2.42 | 5.24 | 5.09 | 1.60 | 0.61 | 98.09 | 49.98 | 0.08 | 5.04 |
| tda | s2 | 10 | 70.31 | 65.20 | 7.51 | 2.40 | 5.48 | 5.09 | 1.42 | 0.02 | 98.18 | 67.13 | 0.10 | 5.01 |
| tda | s3 | 10 | 70.20 | 65.20 | 7.65 | 2.66 | 5.25 | 4.98 | 2.66 | 23.59 | 96.78 | 65.80 | 0.32 | 5.08 |

Column notes (all in %, mean over seeds, except err_run):
- acc / clip: accuracy of the mode and of CLIP alone on the same stream.
- fixed: frames CLIP gets wrong and the mode gets right; broken: CLIP right, mode wrong. acc - clip = fixed - broken.
- start_gain / rest_gain: (mode - CLIP) accuracy in the first 5% of the stream / in the rest (cold start).
- err_run: mean length of a run of consecutive errors; err_in_long: share of errors inside runs of 10 or more
  (long runs = an error that keeps repeating, e.g. a wrong pseudo-label propagating along a video).
- purity: share of samples in the positive cache whose pseudo-label is the true label.
- same_session / same_object: share of the positive cache coming from the session / the object (video) of the current frame.
- top_class: share of the positive cache taken by its largest class.
