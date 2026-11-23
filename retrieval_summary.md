# Retrieval analysis: which cache entries drive the prediction

| Dataset | Backbone | Order | Mode | Seeds | acc | session_share | session_mass | session_bias | class_share | class_mass | class_bias | trap_mass | object_mass | top1_session | top1_class | broken_session_mass | broken_class_mass | broken_trap_mass |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core50_350 | RN50 | s0 | ctxx | 3 | 53.4 | 33.3 | 72.7 | 39.4 | 10.0 | 17.1 | 7.1 | 60.5 | 4.2 | 99.7 | 31.7 | 70.9 | 8.0 | 69.4 |
| core50_350 | RN50 | s0 | tda | 3 | 49.6 | 33.3 | 52.0 | 18.7 | 10.0 | 12.8 | 2.8 | 45.3 | 2.8 | 99.8 | 31.0 | 51.7 | 9.1 | 51.0 |
| core50_350 | RN50 | s2 | ctxx | 3 | 61.2 | 58.8 | 66.8 | 8.0 | 10.0 | 20.1 | 10.2 | 53.1 | 5.2 | 83.7 | 56.7 | 66.4 | 13.6 | 59.3 |
| core50_350 | RN50 | s2 | tda | 3 | 55.0 | 58.8 | 71.1 | 12.4 | 10.0 | 12.9 | 2.9 | 61.9 | 2.9 | 99.8 | 42.9 | 62.8 | 9.3 | 60.3 |
| core50_350 | RN50 | s3 | ctxx | 3 | 56.9 | 47.5 | 52.8 | 5.4 | 11.8 | 20.1 | 8.3 | 40.7 | 7.7 | 70.2 | 50.6 | 52.8 | 13.3 | 46.1 |
| core50_350 | RN50 | s3 | tda | 3 | 53.6 | 47.5 | 56.7 | 9.2 | 11.8 | 14.9 | 3.1 | 47.2 | 5.4 | 93.1 | 40.1 | 48.2 | 10.6 | 45.5 |
| core50_350 | ViT-B/16 | s0 | ctxx | 3 | 64.6 | 33.3 | 66.2 | 32.8 | 10.0 | 19.2 | 9.2 | 53.7 | 5.0 | 97.4 | 36.3 | 68.8 | 7.8 | 67.9 |
| core50_350 | ViT-B/16 | s0 | tda | 3 | 62.3 | 33.3 | 47.3 | 14.0 | 10.0 | 13.2 | 3.2 | 41.0 | 3.0 | 98.0 | 34.8 | 48.6 | 9.0 | 48.0 |
| core50_350 | ViT-B/16 | s2 | ctxx | 3 | 71.1 | 57.5 | 62.3 | 4.8 | 10.0 | 22.1 | 12.1 | 47.9 | 6.0 | 79.7 | 64.1 | 66.0 | 10.6 | 59.0 |
| core50_350 | ViT-B/16 | s2 | tda | 3 | 66.3 | 57.5 | 65.5 | 8.0 | 10.0 | 13.3 | 3.4 | 56.6 | 3.1 | 98.3 | 47.2 | 59.1 | 9.6 | 55.9 |
| core50_350 | ViT-B/16 | s3 | ctxx | 3 | 69.0 | 50.8 | 56.3 | 5.6 | 11.8 | 21.5 | 9.7 | 42.9 | 8.3 | 73.0 | 58.2 | 61.0 | 10.6 | 55.2 |
| core50_350 | ViT-B/16 | s3 | tda | 3 | 67.6 | 50.8 | 58.7 | 7.9 | 11.8 | 15.3 | 3.5 | 48.7 | 5.4 | 91.2 | 48.5 | 53.6 | 10.3 | 50.5 |

All values in %, mean over seeds. *_share: plain composition of the positive cache; *_mass: the same share
weighted by TDA's affinity with the current frame (what actually votes); *_bias = mass - share.
trap_mass: weight on cached samples from the same session but a different true class.
top1_*: the strongest cached neighbour is from the same session / class. broken_*: frames CLIP got right and
the method got wrong. Chance for session_share in a shuffled stream: 1 / number of sessions (CORe50 test: 33%).
