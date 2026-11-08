# Frozen-checkpoint test results

Checkpoint: /home/ubuntu/Deep Learning/Results/training/20260930T110113414620Z/final.pt
SHA256: c98eadb790c9398d8bcb28ab7c1f8d1e780b16d80c5d8abefcdfcc4b30e7131a
Epoch: 23

| Set | Fused windows | Single accuracy | Fused accuracy | Fused macro F1 |
|---|---:|---:|---:|---:|
| S1 | 1636 | 95.94% | 100.00% | 1.0000 |
| S2 | 7924 | 93.55% | 99.87% | 0.9982 |
| S3 | 3095 | 84.71% | 99.77% | 0.9978 |
| S4 | 6111 | 84.30% | 96.25% | 0.9620 |
| S5 | 3062 | 78.78% | 94.35% | 0.9434 |
| S6 | 6149 | 88.42% | 99.69% | 0.9968 |
| S7 | 2872 | 84.14% | 95.51% | 0.9578 |

S1 uses the reserved chronological test portions. S2-S7 use complete external sessions.
All preprocessing and fusion settings are frozen. No retraining or checkpoint selection uses these results.
Window stride 30 creates correlated overlapping samples. Whole-recording centering follows the reference.
