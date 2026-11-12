| Mode | Order | Seeds | acc | clip | fixed | broken | start_gain | rest_gain | err_run | err_in_long | purity | same_session | same_object | top_class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip | s0 | 10 | 59.88 | 59.88 | 0.00 | 0.00 | 0.00 | 0.00 | 1.67 | 0.19 |  |  |  |  |
| clip | s1 rho=0.5 | 10 | 59.88 | 59.88 | 0.00 | 0.00 | 0.00 | 0.00 | 2.13 | 3.78 |  |  |  |  |
| clip | s1 rho=0.9 | 10 | 59.88 | 59.88 | 0.00 | 0.00 | 0.00 | 0.00 | 2.72 | 33.51 |  |  |  |  |
| clip | s2 | 10 | 59.88 | 59.88 | 0.00 | 0.00 | 0.00 | 0.00 | 1.67 | 0.25 |  |  |  |  |
| clip | s3 | 10 | 59.88 | 59.88 | 0.00 | 0.00 | 0.00 | 0.00 | 10.14 | 79.42 |  |  |  |  |
| m2 | s0 | 10 | 62.60 | 59.88 | 11.64 | 8.92 | 3.07 | 2.70 | 1.60 | 0.11 | 99.92 | 33.34 | 2.02 | 10.03 |
| m2 | s1 rho=0.5 | 10 | 63.03 | 59.88 | 11.93 | 8.78 | 2.77 | 3.17 | 1.71 | 0.35 | 99.91 | 33.35 | 2.02 | 10.03 |
| m2 | s1 rho=0.9 | 10 | 63.03 | 59.88 | 11.93 | 8.78 | 2.69 | 3.18 | 1.82 | 1.68 | 99.76 | 33.35 | 2.03 | 10.06 |
| m2 | s2 | 10 | 67.20 | 59.88 | 13.75 | 6.43 | 13.07 | 7.03 | 1.52 | 0.16 | 99.93 | 57.00 | 2.02 | 10.16 |
| m2 | s3 | 10 | 66.06 | 59.88 | 12.70 | 6.52 | 3.61 | 6.32 | 8.15 | 73.11 | 95.46 | 52.30 | 3.87 | 10.60 |
| oracle | s0 | 10 | 65.23 | 59.88 | 12.84 | 7.49 | 6.58 | 5.29 | 1.54 | 0.05 | 100.00 | 33.35 | 2.02 | 10.01 |
| oracle | s1 rho=0.5 | 10 | 65.35 | 59.88 | 12.89 | 7.41 | 6.11 | 5.44 | 1.67 | 0.30 | 100.00 | 33.35 | 2.02 | 10.02 |
| oracle | s1 rho=0.9 | 10 | 65.37 | 59.88 | 12.92 | 7.42 | 7.45 | 5.39 | 1.80 | 1.48 | 100.00 | 33.34 | 2.03 | 10.08 |
| oracle | s2 | 10 | 68.40 | 59.88 | 14.40 | 5.87 | 16.61 | 8.10 | 1.51 | 0.14 | 100.00 | 54.81 | 2.01 | 10.01 |
| oracle | s3 | 10 | 69.71 | 59.88 | 15.51 | 5.67 | 39.30 | 8.29 | 7.51 | 70.54 | 100.00 | 50.82 | 4.10 | 11.71 |
| tda | s0 | 10 | 62.56 | 59.88 | 11.60 | 8.91 | 2.40 | 2.70 | 1.60 | 0.11 | 99.92 | 33.34 | 2.02 | 10.03 |
| tda | s1 rho=0.5 | 10 | 63.01 | 59.88 | 11.90 | 8.77 | 2.34 | 3.17 | 1.71 | 0.36 | 99.91 | 33.35 | 2.02 | 10.03 |
| tda | s1 rho=0.9 | 10 | 63.01 | 59.88 | 11.91 | 8.77 | 2.41 | 3.18 | 1.83 | 1.89 | 99.76 | 33.35 | 2.03 | 10.06 |
| tda | s2 | 10 | 66.84 | 59.88 | 13.39 | 6.43 | 11.59 | 6.72 | 1.52 | 0.16 | 99.93 | 57.00 | 2.02 | 10.16 |
| tda | s3 | 10 | 65.94 | 59.88 | 12.55 | 6.49 | 3.79 | 6.18 | 8.27 | 73.48 | 95.46 | 52.30 | 3.87 | 10.60 |

Column notes (all in %, mean over seeds, except err_run):
- acc / clip: accuracy of the mode and of CLIP alone on the same stream.
- fixed: frames CLIP gets wrong and the mode gets right; broken: CLIP right, mode wrong. acc - clip = fixed - broken.
- start_gain / rest_gain: (mode - CLIP) accuracy in the first 5% of the stream / in the rest (cold start).
- err_run: mean length of a run of consecutive errors; err_in_long: share of errors inside runs of 10 or more
  (long runs = an error that keeps repeating, e.g. a wrong pseudo-label propagating along a video).
- purity: share of samples in the positive cache whose pseudo-label is the true label.
- same_session / same_object: share of the positive cache coming from the session / the object (video) of the current frame.
- top_class: share of the positive cache taken by its largest class.
