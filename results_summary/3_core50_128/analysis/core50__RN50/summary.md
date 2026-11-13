| Mode | Order | Seeds | acc | clip | fixed | broken | start_gain | rest_gain | err_run | err_in_long | purity | same_session | same_object | top_class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip | s0 | 10 | 73.37 | 73.37 | 0.00 | 0.00 | 0.00 | 0.00 | 1.36 | 0.01 |  |  |  |  |
| clip | s1 rho=0.5 | 10 | 73.37 | 73.37 | 0.00 | 0.00 | 0.00 | 0.00 | 1.50 | 0.05 |  |  |  |  |
| clip | s1 rho=0.9 | 10 | 73.37 | 73.37 | 0.00 | 0.00 | 0.00 | 0.00 | 1.62 | 0.38 |  |  |  |  |
| clip | s2 | 10 | 73.37 | 73.37 | 0.00 | 0.00 | 0.00 | 0.00 | 1.36 | 0.01 |  |  |  |  |
| clip | s3 | 10 | 73.37 | 73.37 | 0.00 | 0.00 | 0.00 | 0.00 | 5.82 | 64.76 |  |  |  |  |
| m2 | s0 | 10 | 79.34 | 73.37 | 7.79 | 1.82 | 5.69 | 5.98 | 1.26 | 0.00 | 99.95 | 33.35 | 2.02 | 10.01 |
| m2 | s1 rho=0.5 | 10 | 79.41 | 73.37 | 7.91 | 1.86 | 5.65 | 6.06 | 1.34 | 0.00 | 99.94 | 33.34 | 2.02 | 10.02 |
| m2 | s1 rho=0.9 | 10 | 79.40 | 73.37 | 7.89 | 1.86 | 5.78 | 6.04 | 1.41 | 0.12 | 99.83 | 33.35 | 2.03 | 10.04 |
| m2 | s2 | 10 | 80.32 | 73.37 | 8.76 | 1.81 | 8.17 | 6.89 | 1.25 | 0.00 | 99.93 | 57.20 | 2.02 | 10.01 |
| m2 | s3 | 10 | 79.66 | 73.37 | 8.36 | 2.07 | -1.72 | 6.71 | 5.24 | 59.65 | 95.65 | 50.71 | 3.92 | 10.50 |
| oracle | s0 | 10 | 79.28 | 73.37 | 7.99 | 2.07 | 6.68 | 5.87 | 1.26 | 0.00 | 100.00 | 33.35 | 2.02 | 10.01 |
| oracle | s1 rho=0.5 | 10 | 79.41 | 73.37 | 8.12 | 2.09 | 6.42 | 6.02 | 1.35 | 0.00 | 100.00 | 33.35 | 2.02 | 10.02 |
| oracle | s1 rho=0.9 | 10 | 79.43 | 73.37 | 8.13 | 2.07 | 7.29 | 5.99 | 1.42 | 0.12 | 100.00 | 33.36 | 2.03 | 10.07 |
| oracle | s2 | 10 | 80.25 | 73.37 | 8.79 | 1.91 | 9.19 | 6.76 | 1.25 | 0.00 | 100.00 | 56.38 | 2.02 | 10.01 |
| oracle | s3 | 10 | 81.68 | 73.37 | 9.97 | 1.66 | 23.00 | 7.53 | 4.94 | 57.64 | 100.00 | 50.93 | 4.11 | 11.71 |
| tda | s0 | 10 | 79.34 | 73.37 | 7.79 | 1.82 | 5.65 | 5.98 | 1.26 | 0.00 | 99.95 | 33.35 | 2.02 | 10.01 |
| tda | s1 rho=0.5 | 10 | 79.41 | 73.37 | 7.91 | 1.86 | 5.61 | 6.06 | 1.34 | 0.00 | 99.94 | 33.34 | 2.02 | 10.02 |
| tda | s1 rho=0.9 | 10 | 79.40 | 73.37 | 7.89 | 1.86 | 5.81 | 6.04 | 1.41 | 0.12 | 99.83 | 33.35 | 2.03 | 10.04 |
| tda | s2 | 10 | 80.32 | 73.37 | 8.76 | 1.81 | 8.06 | 6.89 | 1.25 | 0.00 | 99.93 | 57.20 | 2.02 | 10.01 |
| tda | s3 | 10 | 79.71 | 73.37 | 8.37 | 2.04 | -1.18 | 6.73 | 5.24 | 59.61 | 95.65 | 50.71 | 3.92 | 10.50 |

Column notes (all in %, mean over seeds, except err_run):
- acc / clip: accuracy of the mode and of CLIP alone on the same stream.
- fixed: frames CLIP gets wrong and the mode gets right; broken: CLIP right, mode wrong. acc - clip = fixed - broken.
- start_gain / rest_gain: (mode - CLIP) accuracy in the first 5% of the stream / in the rest (cold start).
- err_run: mean length of a run of consecutive errors; err_in_long: share of errors inside runs of 10 or more
  (long runs = an error that keeps repeating, e.g. a wrong pseudo-label propagating along a video).
- purity: share of samples in the positive cache whose pseudo-label is the true label.
- same_session / same_object: share of the positive cache coming from the session / the object (video) of the current frame.
- top_class: share of the positive cache taken by its largest class.
