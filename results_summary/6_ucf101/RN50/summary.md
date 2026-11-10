| Mode | Order | Seeds | acc | clip | fixed | broken | start_gain | rest_gain | err_run | err_in_long | purity | same_session | same_object | top_class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip | s0 | 10 | 63.28 | 63.28 | 0.00 | 0.00 | 0.00 | 0.00 | 1.58 | 0.08 |  |  |  |  |
| clip | s1 rho=0.5 | 10 | 63.28 | 63.28 | 0.00 | 0.00 | 0.00 | 0.00 | 1.97 | 2.13 |  |  |  |  |
| clip | s1 rho=0.9 | 10 | 63.28 | 63.28 | 0.00 | 0.00 | 0.00 | 0.00 | 2.46 | 21.49 |  |  |  |  |
| clip | s2 | 10 | 63.28 | 63.28 | 0.00 | 0.00 | 0.00 | 0.00 | 1.59 | 0.09 |  |  |  |  |
| clip | s3 | 10 | 63.28 | 63.28 | 0.00 | 0.00 | 0.00 | 0.00 | 6.46 | 69.15 |  |  |  |  |
| m2 | s0 | 10 | 66.27 | 63.28 | 6.14 | 3.16 | 1.95 | 3.04 | 1.51 | 0.02 | 87.04 | 15.57 | 0.12 | 1.01 |
| m2 | s1 rho=0.5 | 10 | 66.25 | 63.28 | 6.04 | 3.07 | 1.97 | 3.02 | 1.85 | 1.43 | 87.00 | 15.53 | 0.12 | 1.03 |
| m2 | s1 rho=0.9 | 10 | 66.07 | 63.28 | 5.97 | 3.19 | 0.03 | 2.93 | 2.28 | 16.04 | 85.79 | 15.51 | 0.14 | 1.06 |
| m2 | s2 | 10 | 67.67 | 63.28 | 7.16 | 2.77 | 7.93 | 4.20 | 1.49 | 0.08 | 84.65 | 32.69 | 0.24 | 1.02 |
| m2 | s3 | 10 | 66.58 | 63.28 | 6.63 | 3.33 | 0.50 | 3.44 | 7.09 | 71.47 | 80.82 | 27.96 | 0.58 | 1.20 |
| oracle | s0 | 10 | 70.07 | 63.28 | 9.53 | 2.74 | 14.86 | 6.36 | 1.43 | 0.01 | 100.00 | 15.42 | 0.12 | 1.01 |
| oracle | s1 rho=0.5 | 10 | 70.09 | 63.28 | 9.52 | 2.71 | 14.46 | 6.41 | 1.71 | 0.73 | 100.00 | 15.39 | 0.12 | 1.03 |
| oracle | s1 rho=0.9 | 10 | 70.17 | 63.28 | 9.58 | 2.69 | 15.10 | 6.46 | 2.03 | 10.53 | 100.00 | 15.39 | 0.14 | 1.11 |
| oracle | s2 | 10 | 74.54 | 63.28 | 13.26 | 2.01 | 30.86 | 10.22 | 1.40 | 0.03 | 100.00 | 32.78 | 0.24 | 1.01 |
| oracle | s3 | 10 | 74.41 | 63.28 | 13.30 | 2.16 | 33.94 | 9.93 | 5.61 | 63.24 | 100.00 | 27.23 | 0.66 | 1.42 |
| tda | s0 | 10 | 66.26 | 63.28 | 6.13 | 3.15 | 1.78 | 3.04 | 1.51 | 0.02 | 87.04 | 15.57 | 0.12 | 1.01 |
| tda | s1 rho=0.5 | 10 | 66.24 | 63.28 | 6.03 | 3.07 | 1.80 | 3.02 | 1.85 | 1.44 | 87.00 | 15.53 | 0.12 | 1.03 |
| tda | s1 rho=0.9 | 10 | 66.09 | 63.28 | 5.96 | 3.15 | 0.55 | 2.93 | 2.28 | 16.09 | 85.79 | 15.51 | 0.14 | 1.06 |
| tda | s2 | 10 | 67.66 | 63.28 | 7.13 | 2.75 | 7.52 | 4.22 | 1.49 | 0.08 | 84.65 | 32.69 | 0.24 | 1.02 |
| tda | s3 | 10 | 66.66 | 63.28 | 6.64 | 3.26 | 2.12 | 3.44 | 7.11 | 71.58 | 80.82 | 27.96 | 0.58 | 1.20 |

Column notes (all in %, mean over seeds, except err_run):
- acc / clip: accuracy of the mode and of CLIP alone on the same stream.
- fixed: frames CLIP gets wrong and the mode gets right; broken: CLIP right, mode wrong. acc - clip = fixed - broken.
- start_gain / rest_gain: (mode - CLIP) accuracy in the first 5% of the stream / in the rest (cold start).
- err_run: mean length of a run of consecutive errors; err_in_long: share of errors inside runs of 10 or more
  (long runs = an error that keeps repeating, e.g. a wrong pseudo-label propagating along a video).
- purity: share of samples in the positive cache whose pseudo-label is the true label.
- same_session / same_object: share of the positive cache coming from the session / the object (video) of the current frame.
- top_class: share of the positive cache taken by its largest class.
