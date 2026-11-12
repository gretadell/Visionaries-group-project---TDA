| Mode | Order | Seeds | acc | clip | fixed | broken | start_gain | rest_gain | err_run | err_in_long | purity | same_session | same_object | top_class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip | s0 | 10 | 51.54 | 51.54 | 0.00 | 0.00 | 0.00 | 0.00 | 1.94 | 0.65 |  |  |  |  |
| clip | s1 rho=0.5 | 10 | 51.54 | 51.54 | 0.00 | 0.00 | 0.00 | 0.00 | 2.47 | 7.19 |  |  |  |  |
| clip | s1 rho=0.9 | 10 | 51.54 | 51.54 | 0.00 | 0.00 | 0.00 | 0.00 | 3.18 | 34.53 |  |  |  |  |
| clip | s2 | 10 | 51.54 | 51.54 | 0.00 | 0.00 | 0.00 | 0.00 | 1.94 | 0.81 |  |  |  |  |
| clip | s3 | 10 | 51.54 | 51.54 | 0.00 | 0.00 | 0.00 | 0.00 | 11.08 | 81.24 |  |  |  |  |
| m2 | s0 | 10 | 49.63 | 51.54 | 10.04 | 11.95 | -2.84 | -1.87 | 2.02 | 1.01 | 99.08 | 33.34 | 2.02 | 10.03 |
| m2 | s1 rho=0.5 | 10 | 49.63 | 51.54 | 10.07 | 11.98 | -3.29 | -1.84 | 2.20 | 2.71 | 99.17 | 33.34 | 2.02 | 10.03 |
| m2 | s1 rho=0.9 | 10 | 49.58 | 51.54 | 10.04 | 12.00 | -4.31 | -1.84 | 2.39 | 8.20 | 98.97 | 33.35 | 2.04 | 10.04 |
| m2 | s2 | 10 | 54.91 | 51.54 | 12.03 | 8.66 | 12.07 | 2.91 | 1.87 | 1.15 | 98.02 | 57.34 | 2.02 | 10.05 |
| m2 | s3 | 10 | 54.93 | 51.54 | 11.58 | 8.19 | 3.03 | 3.41 | 9.59 | 77.96 | 92.34 | 51.84 | 3.93 | 10.37 |
| oracle | s0 | 10 | 46.73 | 51.54 | 7.62 | 12.44 | 0.37 | -5.09 | 2.14 | 1.71 | 100.00 | 33.35 | 2.02 | 10.01 |
| oracle | s1 rho=0.5 | 10 | 46.94 | 51.54 | 7.85 | 12.45 | -0.31 | -4.83 | 2.47 | 6.14 | 100.00 | 33.35 | 2.02 | 10.02 |
| oracle | s1 rho=0.9 | 10 | 46.97 | 51.54 | 7.86 | 12.43 | 1.45 | -4.89 | 2.83 | 19.90 | 100.00 | 33.35 | 2.04 | 10.08 |
| oracle | s2 | 10 | 53.11 | 51.54 | 10.59 | 9.02 | 15.02 | 0.86 | 1.95 | 1.67 | 100.00 | 56.91 | 2.02 | 10.01 |
| oracle | s3 | 10 | 55.73 | 51.54 | 12.64 | 8.46 | 45.23 | 2.02 | 10.04 | 78.89 | 100.00 | 50.20 | 4.16 | 11.71 |
| tda | s0 | 10 | 49.64 | 51.54 | 10.02 | 11.92 | -2.71 | -1.87 | 2.02 | 1.01 | 99.08 | 33.34 | 2.02 | 10.03 |
| tda | s1 rho=0.5 | 10 | 49.63 | 51.54 | 10.05 | 11.96 | -3.44 | -1.84 | 2.20 | 2.73 | 99.17 | 33.34 | 2.02 | 10.03 |
| tda | s1 rho=0.9 | 10 | 49.59 | 51.54 | 10.02 | 11.98 | -4.24 | -1.84 | 2.39 | 8.30 | 98.97 | 33.35 | 2.04 | 10.04 |
| tda | s2 | 10 | 54.81 | 51.54 | 11.93 | 8.67 | 10.43 | 2.89 | 1.87 | 1.15 | 98.02 | 57.34 | 2.02 | 10.05 |
| tda | s3 | 10 | 54.85 | 51.54 | 11.49 | 8.19 | 2.53 | 3.34 | 9.65 | 78.12 | 92.34 | 51.84 | 3.93 | 10.37 |

Column notes (all in %, mean over seeds, except err_run):
- acc / clip: accuracy of the mode and of CLIP alone on the same stream.
- fixed: frames CLIP gets wrong and the mode gets right; broken: CLIP right, mode wrong. acc - clip = fixed - broken.
- start_gain / rest_gain: (mode - CLIP) accuracy in the first 5% of the stream / in the rest (cold start).
- err_run: mean length of a run of consecutive errors; err_in_long: share of errors inside runs of 10 or more
  (long runs = an error that keeps repeating, e.g. a wrong pseudo-label propagating along a video).
- purity: share of samples in the positive cache whose pseudo-label is the true label.
- same_session / same_object: share of the positive cache coming from the session / the object (video) of the current frame.
- top_class: share of the positive cache taken by its largest class.
