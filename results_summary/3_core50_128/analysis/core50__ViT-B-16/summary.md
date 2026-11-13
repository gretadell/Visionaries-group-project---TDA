| Mode | Order | Seeds | acc | clip | fixed | broken | start_gain | rest_gain | err_run | err_in_long | purity | same_session | same_object | top_class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip | s0 | 10 | 79.17 | 79.17 | 0.00 | 0.00 | 0.00 | 0.00 | 1.26 | 0.00 |  |  |  |  |
| clip | s1 rho=0.5 | 10 | 79.17 | 79.17 | 0.00 | 0.00 | 0.00 | 0.00 | 1.40 | 0.06 |  |  |  |  |
| clip | s1 rho=0.9 | 10 | 79.17 | 79.17 | 0.00 | 0.00 | 0.00 | 0.00 | 1.52 | 0.99 |  |  |  |  |
| clip | s2 | 10 | 79.17 | 79.17 | 0.00 | 0.00 | 0.00 | 0.00 | 1.26 | 0.00 |  |  |  |  |
| clip | s3 | 10 | 79.17 | 79.17 | 0.00 | 0.00 | 0.00 | 0.00 | 5.53 | 60.20 |  |  |  |  |
| m2 | s0 | 10 | 84.70 | 79.17 | 7.86 | 2.33 | 7.46 | 5.43 | 1.18 | 0.00 | 99.95 | 33.34 | 2.02 | 10.01 |
| m2 | s1 rho=0.5 | 10 | 84.92 | 79.17 | 8.12 | 2.37 | 7.14 | 5.68 | 1.24 | 0.00 | 99.92 | 33.34 | 2.02 | 10.02 |
| m2 | s1 rho=0.9 | 10 | 84.91 | 79.17 | 8.11 | 2.37 | 7.47 | 5.65 | 1.29 | 0.03 | 99.83 | 33.33 | 2.03 | 10.04 |
| m2 | s2 | 10 | 86.22 | 79.17 | 9.42 | 2.37 | 8.67 | 6.96 | 1.16 | 0.00 | 99.98 | 56.12 | 2.02 | 10.01 |
| m2 | s3 | 10 | 85.70 | 79.17 | 9.11 | 2.58 | 2.57 | 6.74 | 4.72 | 53.42 | 96.15 | 49.99 | 3.89 | 10.68 |
| oracle | s0 | 10 | 84.96 | 79.17 | 8.62 | 2.83 | 8.50 | 5.65 | 1.18 | 0.00 | 100.00 | 33.35 | 2.02 | 10.01 |
| oracle | s1 rho=0.5 | 10 | 85.03 | 79.17 | 8.80 | 2.94 | 8.12 | 5.75 | 1.24 | 0.00 | 100.00 | 33.35 | 2.02 | 10.02 |
| oracle | s1 rho=0.9 | 10 | 85.06 | 79.17 | 8.81 | 2.92 | 9.06 | 5.73 | 1.30 | 0.03 | 100.00 | 33.34 | 2.03 | 10.07 |
| oracle | s2 | 10 | 85.99 | 79.17 | 9.60 | 2.77 | 9.68 | 6.67 | 1.16 | 0.00 | 100.00 | 54.01 | 2.02 | 10.01 |
| oracle | s3 | 10 | 86.49 | 79.17 | 9.97 | 2.65 | 19.40 | 6.69 | 4.69 | 53.42 | 100.00 | 48.89 | 3.99 | 11.71 |
| tda | s0 | 10 | 84.69 | 79.17 | 7.86 | 2.33 | 7.39 | 5.43 | 1.18 | 0.00 | 99.95 | 33.34 | 2.02 | 10.01 |
| tda | s1 rho=0.5 | 10 | 84.91 | 79.17 | 8.12 | 2.37 | 7.05 | 5.68 | 1.24 | 0.00 | 99.92 | 33.34 | 2.02 | 10.02 |
| tda | s1 rho=0.9 | 10 | 84.90 | 79.17 | 8.11 | 2.37 | 7.43 | 5.65 | 1.29 | 0.03 | 99.83 | 33.33 | 2.03 | 10.04 |
| tda | s2 | 10 | 86.21 | 79.17 | 9.42 | 2.37 | 8.55 | 6.96 | 1.16 | 0.00 | 99.98 | 56.12 | 2.02 | 10.01 |
| tda | s3 | 10 | 85.71 | 79.17 | 9.10 | 2.55 | 2.98 | 6.74 | 4.73 | 53.50 | 96.15 | 49.99 | 3.89 | 10.68 |

Column notes (all in %, mean over seeds, except err_run):
- acc / clip: accuracy of the mode and of CLIP alone on the same stream.
- fixed: frames CLIP gets wrong and the mode gets right; broken: CLIP right, mode wrong. acc - clip = fixed - broken.
- start_gain / rest_gain: (mode - CLIP) accuracy in the first 5% of the stream / in the rest (cold start).
- err_run: mean length of a run of consecutive errors; err_in_long: share of errors inside runs of 10 or more
  (long runs = an error that keeps repeating, e.g. a wrong pseudo-label propagating along a video).
- purity: share of samples in the positive cache whose pseudo-label is the true label.
- same_session / same_object: share of the positive cache coming from the session / the object (video) of the current frame.
- top_class: share of the positive cache taken by its largest class.
