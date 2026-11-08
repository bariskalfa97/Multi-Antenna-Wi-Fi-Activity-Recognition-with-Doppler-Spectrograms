# Frozen-checkpoint test results

Checkpoint: /home/ubuntu/Deep Learning/Results/contrastive/20260930T123247844416Z/final.pt
SHA256: dd5af05b52b969731f60c540c102ca1c16f0aafcb1a5c0c4dc5f079596248684
Epoch: 25

| Set | Fused windows | Single accuracy | Fused accuracy | Fused macro F1 |
|---|---:|---:|---:|---:|
| S1 | 1636 | 98.09% | 100.00% | 1.0000 |
| S2 | 7924 | 90.83% | 94.76% | 0.9315 |
| S3 | 3095 | 84.04% | 99.48% | 0.9950 |
| S4 | 6111 | 78.78% | 87.01% | 0.8605 |
| S5 | 3062 | 75.54% | 81.22% | 0.7982 |
| S6 | 6149 | 90.72% | 99.54% | 0.9955 |
| S7 | 2872 | 72.75% | 80.40% | 0.7851 |

S1 uses the reserved chronological test portions. S2-S7 use complete external sessions.
All preprocessing and fusion settings are frozen. No retraining or checkpoint selection uses these results.
Window stride 30 creates correlated overlapping samples. Whole-recording centering follows the reference.
