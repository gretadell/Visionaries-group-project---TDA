| Mode | Order | Seeds | acc | clip | fixed | broken | start_gain | rest_gain | err_run | err_in_long | purity | same_session | same_object | top_class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip | s0 | 10 | 67.45 | 67.45 | 0.00 | 0.00 | 0.00 | 0.00 | 1.48 | 0.02 |  |  |  |  |
| clip | s1 rho=0.5 | 10 | 67.45 | 67.45 | 0.00 | 0.00 | 0.00 | 0.00 | 1.91 | 1.82 |  |  |  |  |
| clip | s1 rho=0.9 | 10 | 67.45 | 67.45 | 0.00 | 0.00 | 0.00 | 0.00 | 2.50 | 20.07 |  |  |  |  |
| clip | s2 | 10 | 67.45 | 67.45 | 0.00 | 0.00 | 0.00 | 0.00 | 1.48 | 0.03 |  |  |  |  |
| clip | s3 | 10 | 67.45 | 67.45 | 0.00 | 0.00 | 0.00 | 0.00 | 6.55 | 67.82 |  |  |  |  |
| m2 | s0 | 10 | 71.08 | 67.45 | 7.03 | 3.40 | 2.76 | 3.67 | 1.41 | 0.01 | 89.36 | 15.09 | 0.12 | 1.01 |
| m2 | s1 rho=0.5 | 10 | 71.06 | 67.45 | 6.98 | 3.37 | 2.22 | 3.68 | 1.78 | 1.01 | 89.69 | 15.05 | 0.12 | 1.03 |
| m2 | s1 rho=0.9 | 10 | 70.95 | 67.45 | 6.90 | 3.40 | 0.72 | 3.65 | 2.27 | 16.12 | 88.48 | 15.05 | 0.13 | 1.06 |
| m2 | s2 | 10 | 71.96 | 67.46 | 7.44 | 2.94 | 7.37 | 4.35 | 1.40 | 0.03 | 86.96 | 32.12 | 0.24 | 1.02 |
| m2 | s3 | 10 | 71.21 | 67.46 | 6.88 | 3.13 | 0.64 | 3.91 | 7.36 | 71.89 | 84.04 | 27.05 | 0.59 | 1.24 |
| oracle | s0 | 10 | 74.39 | 67.46 | 9.26 | 2.33 | 13.58 | 6.58 | 1.35 | 0.02 | 100.00 | 14.90 | 0.12 | 1.01 |
| oracle | s1 rho=0.5 | 10 | 74.32 | 67.46 | 9.20 | 2.35 | 13.66 | 6.50 | 1.69 | 0.64 | 100.00 | 14.88 | 0.12 | 1.03 |
| oracle | s1 rho=0.9 | 10 | 74.46 | 67.46 | 9.31 | 2.31 | 14.51 | 6.61 | 2.11 | 13.71 | 100.00 | 14.88 | 0.14 | 1.11 |
| oracle | s2 | 10 | 77.94 | 67.46 | 12.23 | 1.75 | 27.40 | 9.59 | 1.32 | 0.01 | 100.00 | 31.94 | 0.23 | 1.01 |
| oracle | s3 | 10 | 77.82 | 67.46 | 12.34 | 1.98 | 29.12 | 9.38 | 5.89 | 64.48 | 100.00 | 26.46 | 0.64 | 1.42 |
| tda | s0 | 10 | 71.06 | 67.45 | 7.01 | 3.40 | 2.53 | 3.66 | 1.41 | 0.01 | 89.36 | 15.09 | 0.12 | 1.01 |
| tda | s1 rho=0.5 | 10 | 71.05 | 67.45 | 6.96 | 3.36 | 2.09 | 3.68 | 1.78 | 1.00 | 89.69 | 15.05 | 0.12 | 1.03 |
| tda | s1 rho=0.9 | 10 | 70.96 | 67.45 | 6.88 | 3.37 | 1.02 | 3.64 | 2.27 | 16.23 | 88.48 | 15.05 | 0.13 | 1.06 |
| tda | s2 | 10 | 71.92 | 67.45 | 7.41 | 2.94 | 6.89 | 4.35 | 1.40 | 0.02 | 86.97 | 32.10 | 0.24 | 1.02 |
| tda | s3 | 10 | 71.21 | 67.45 | 6.87 | 3.12 | 1.43 | 3.88 | 7.40 | 71.98 | 84.07 | 27.02 | 0.59 | 1.24 |

Column notes (all in %, mean over seeds, except err_run):
- acc / clip: accuracy of the mode and of CLIP alone on the same stream.
- fixed: frames CLIP gets wrong and the mode gets right; broken: CLIP right, mode wrong. acc - clip = fixed - broken.
- start_gain / rest_gain: (mode - CLIP) accuracy in the first 5% of the stream / in the rest (cold start).
- err_run: mean length of a run of consecutive errors; err_in_long: share of errors inside runs of 10 or more
  (long runs = an error that keeps repeating, e.g. a wrong pseudo-label propagating along a video).
- purity: share of samples in the positive cache whose pseudo-label is the true label.
- same_session / same_object: share of the positive cache coming from the session / the object (video) of the current frame.
- top_class: share of the positive cache taken by its largest class.
