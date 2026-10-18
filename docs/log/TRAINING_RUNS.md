# Training run log (exp_oct)

Generated 2026-10-09 from each run's `done.json` just before the model weights were deleted.
All weights (`*.ckpt`) under `outputs/` were removed after this log was written; metrics, configs, `train.log`, eval CSVs and logits were kept in each run folder.
Full per-run settings and exact commands: `TRAINING_RUNS.csv` (columns `overrides`, `command`). Narrative results: `docs/EXPERIMENTS_OCT.md`.

| run | arch | seed | steps | min | train set (lmdb) | note | real test | words | cov@99 | conf-err | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_s42 | None | 42 | 6570 | 0.0 | master_lmdb |  | 94.2% (357/379) | 37.3% | 38.5% | 8 | cd6ef72+uncommitted |
| baseline_s43 | None | 43 | 6570 | 18.0 | master_lmdb |  | 94.2% (357/379) | 42.3% | 55.4% | 3 | cd6ef72+uncommitted |
| baseline_s44 | None | 44 | 6570 | 19.0 | master_lmdb |  | 95.5% (362/379) | 40.7% | 54.6% | 7 | cd6ef72+uncommitted |
| crnn_s42 | crnn | 42 | 6570 | 9.0 | master_lmdb | crnn_ctc_pretrained | 88.4% (335/379) | 24.3% | 41.2% | 6 | 3e86353+uncommitted |
| crnn_s43 | crnn | 43 | 6570 | 9.0 | master_lmdb | crnn_ctc_pretrained | 88.1% (334/379) | 23.3% | 39.3% | 6 | 3e86353+uncommitted |
| crnn_s44 | crnn | 44 | 6570 | 8.0 | master_lmdb | crnn_ctc_pretrained | 87.9% (333/379) | 23.7% | 38.0% | 8 | 3e86353+uncommitted |
| ls005_s42 | None | 42 | 6570 | 20.0 | master_lmdb | label_smoothing_0.05 | 95.0% (360/379) | 23.3% | 16.6% | 4 | ead9580+uncommitted |
| ls005_s43 | None | 43 | 6570 | 20.0 | master_lmdb | label_smoothing_0.05 | 94.5% (358/379) | 26.0% | 4.2% | 4 | ead9580+uncommitted |
| ls005_s44 | None | 44 | 6570 | 20.0 | master_lmdb | label_smoothing_0.05 | 94.5% (358/379) | 27.3% | 8.7% | 4 | ead9580+uncommitted |
| ls010_s42 | None | 42 | 6570 | 20.0 | master_lmdb | label_smoothing_0.1 | 93.9% (356/379) | 18.7% | 3.4% | 4 | ead9580+uncommitted |
| pad_s42 | None | 42 | 6570 | 25.0 | master_lmdb | padding | 93.9% (356/379) | 11.0% | 40.1% | 7 | 3ffeeca+uncommitted |
| pad_s43 | None | 43 | 6570 | 20.0 | master_lmdb | padding | 93.9% (356/379) | 14.3% | 43.8% | 7 | 3ffeeca+uncommitted |
| pad_s44 | None | 44 | 6570 | 39.0 | master_lmdb | padding | 94.5% (358/379) | 12.3% | 42.5% | 6 | 3ffeeca+uncommitted |
| replay_p10_s42 | None | 42 | 6570 | 18.0 | replay_p10 | replay_10 | 93.7% (355/379) | 87.3% | 60.2% | 7 | 3dfea8b+uncommitted |
| replay_p20_s42 | None | 42 | 6570 | 18.0 | replay_p20 | replay_20 | 93.9% (356/379) | 89.3% | 55.4% | 6 | 3dfea8b+uncommitted |
| replay_p20_s43 | None | 43 | 6570 | 19.0 | replay_p20 | replay_20 | 92.9% (352/379) | 88.0% | 44.9% | 10 | 3dfea8b+uncommitted |
| replay_p20_s44 | None | 44 | 6570 | 20.0 | replay_p20 | replay_20 | 93.9% (356/379) | 87.7% | 64.1% | 7 | 3dfea8b+uncommitted |
| replay_p30_s42 | None | 42 | 6570 | 18.0 | replay_p30 | replay_30 | 93.9% (356/379) | 89.0% | 61.7% | 10 | 3dfea8b+uncommitted |
| replay_p30_s43 | None | 43 | 6570 | 19.0 | replay_p30 | replay_30 | 95.3% (361/379) | 90.0% | 40.4% | 9 | 3dfea8b+uncommitted |
| replay_p30_s44 | None | 44 | 6570 | 19.0 | replay_p30 | replay_30 | 94.5% (358/379) | 89.3% | 50.7% | 8 | 3dfea8b+uncommitted |
| rm2_s42 | parseq | 42 | 8445 | 25.0 | replay30_marks2 | replay30_plus_marked_all_formats_4000 | 94.5% (358/379) | 90.3% | 58.8% | 7 | c5dd623+uncommitted |
| rm2_s43 | parseq | 43 | 8445 | 23.0 | replay30_marks2 | replay30_plus_marked_all_formats_4000 | 93.9% (356/379) | 89.0% | 57.5% | 7 | c5dd623+uncommitted |
| rm2_s44 | parseq | 44 | 8445 | 23.0 | replay30_marks2 | replay30_plus_marked_all_formats_4000 | 95.3% (361/379) | 88.3% | 59.9% | 9 | c5dd623+uncommitted |
| rm6k_s42 | parseq | 42 | 9375 | 27.0 | replay30_marks6k | replay30_plus_marked_NNa_6000 | 94.7% (359/379) | 89.0% | 62.3% | 6 | 5f227df+uncommitted |
| rm6k_s43 | parseq | 43 | 9375 | 27.0 | replay30_marks6k | replay30_plus_marked_NNa_6000 | 93.4% (354/379) | 89.7% | 59.4% | 5 | 5f227df+uncommitted |
| rm6k_s44 | parseq | 44 | 9375 | 28.0 | replay30_marks6k | replay30_plus_marked_NNa_6000 | 94.5% (358/379) | 89.7% | 49.6% | 5 | 5f227df+uncommitted |
| ss3k_s42 | parseq | 42 | 7995 | 22.0 | replay30_ssl3k | replay30_plus_3056_real_SSL-HWD_crops | 92.9% (352/379) | 88.3% | 39.6% | 9 | 6938c31+uncommitted |
| ss3k_s43 | parseq | 43 | 7995 | 22.0 | replay30_ssl3k | replay30_plus_3056_real_SSL-HWD_crops | 93.7% (355/379) | 88.3% | 48.3% | 9 | 6938c31+uncommitted |
| ss3k_s44 | parseq | 44 | 7995 | 23.0 | replay30_ssl3k | replay30_plus_3056_real_SSL-HWD_crops | 93.7% (355/379) | 89.0% | 44.6% | 9 | 6938c31+uncommitted |
