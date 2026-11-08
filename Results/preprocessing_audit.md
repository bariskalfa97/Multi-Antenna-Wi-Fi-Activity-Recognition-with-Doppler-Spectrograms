# SHARP preprocessing audit — 2026-09-26

Reference commit: 5bc1788896e800218cd167734f6c11a05b88e983 of https://github.com/francescamen/SHARP.

## Confirmed mismatch and correction
The reference subtracts each bin's temporal mean over the complete antenna recording before windowing:
`centered = raw - raw.mean(axis=0, keepdims=True)`.
Both CSI_doppler_create_dataset_train.py and CSI_doppler_create_dataset_test.py perform this step. Our original loader omitted it. Source/train.py now defaults to --preprocessing trace_mean; --preprocessing raw reproduces the old input transform. Raw files are unchanged. Signed centered values are retained without clipping, logarithms or re-normalization.

Sources:
- https://github.com/francescamen/SHARP/blob/5bc1788896e800218cd167734f6c11a05b88e983/Python_code/CSI_doppler_create_dataset_train.py
- https://github.com/francescamen/SHARP/blob/5bc1788896e800218cd167734f6c11a05b88e983/Python_code/CSI_doppler_create_dataset_test.py
- https://github.com/francescamen/SHARP/blob/5bc1788896e800218cd167734f6c11a05b88e983/Python_code/dataset_utility.py
- https://github.com/francescamen/SHARP/blob/5bc1788896e800218cd167734f6c11a05b88e983/Python_code/CSI_network.py
- https://github.com/francescamen/SHARP/blob/5bc1788896e800218cd167734f6c11a05b88e983/Python_code/network_utility.py

## Other checks and differences
- Stored Doppler values are linear powers normalized per time row; reference generation clips below 10^-1.2. A log transform is used for plots, not the classifier loader.
- Reference input layout is time x bins x channel; our channel x time x bins is the equivalent PyTorch layout. No bin flip or additional FFT shift is needed.
- Diagram's filters, strides, padding, dropout, output logits and parameter count agree with the reference. Flatten ordering differs across frameworks, so pretrained dense weights would require permutation if imported; we train from scratch.
- Baseline E/L/W/R/J labels agree. Numeric repetition suffixes are included; ambiguous LOS is excluded by our strict parser, whereas reference code reads a single character at a fixed filename position. No LOS file occurs in the training/validation splits used here.
- Reference example uses stride 30. We retain 170 training / 340 validation for this controlled comparison. Reference window range excludes the final exactly fitting start; ours includes it.
- Reference uses Adam learning rate 0.0001; ours retains 0.001 to isolate preprocessing.
- Reference splits individual traces chronologically 60/20/20 with 31-row gaps. We retain separate sessions: S1a,S1b train; S1c validation. This is not exact paper reproduction.
- Reference uses multi-antenna decision fusion; our reported comparison is single-antenna.
- Full-trace centering uses future samples within each recording. It is an offline transform, not causal streaming preprocessing. With session-disjoint splits, no training/validation recording is shared and labels are not used to compute means. Online use would need a separately validated causal method.

## Controlled comparison
Both runs: seed 42, Glorot weights, deterministic cuDNN setting, 5 epochs, Adam 0.001, batch 32, identical sessions/windows, no test data loading. Best validation macro-F1 epoch is reported for each run.

| Input | Epoch | Validation accuracy | Macro F1 | Sitting predicted empty |
|---|---:|---:|---:|---:|
| Raw | 5 | 48.62% | 0.4457 | 160/228 |
| Trace mean removed | 2 | 73.35% | 0.7325 | 7/228 |

Run directories: Results/training/audit_raw_20260926 and Results/training/audit_centered_20260926.

This is a single-seed, short validation comparison, not a test-set performance claim. Walking/running/jumping confusion remains. No tuning or evaluation on S7a was performed.

## Validation and usage
Exact NumPy equality checks passed for all 24 validation traces against the reference mean subtraction formula, including window shape, orientation and label consistency. Per-bin residual means are below 1e-6. Both runs passed initial gradient checks. Both manifests contain only train/val.

Training now saves best.pt (lowest validation loss) and best_f1.pt (highest validation macro F1). Patience remains 20 and early stopping still monitors loss. Test data is only loaded when --evaluate-test is explicitly supplied; it evaluates best.pt. For tuning:

    python3 Source/train.py --epochs 30 --patience 20

Use --preprocessing raw only for comparisons to the original pipeline. Old checkpoints require their original preprocessing and should not be evaluated with the corrected transform.
