# SHARP reference settings

Default command on the server:

```sh
python3 '/home/ubuntu/Deep Learning/Source/train.py'
```

Reference: https://github.com/francescamen/SHARP/tree/5bc1788896e800218cd167734f6c11a05b88e983
Local pinned reference and pre-change source backup:
`Results/reproduction_setup/20260930T105538Z/`

- Same PyTorch CNN: 1x340x100 input, 128535 parameters, five classes, dropout 0.2.
- S1a/S1b/S1c each split chronologically: first floor(0.6*N) rows train; skip 31 rows; next floor(0.2*N) rows validation; skip 31 rows; remainder S1 test.
- Whole-recording per-bin mean subtraction before splitting matches the reference. This intentionally reads statistics from all temporal portions of the recording, including the held-out portions; it is an offline, noncausal reference preprocessing choice.
- Window length 340 and stride 30 in all partitions. Match the reference exclusive window endpoint exactly.
- Adam learning rate 0.0001, betas (0.9,0.999), epsilon 1e-7. ReferenceAdam implements the Keras epsilon placement in the update, which differs from simply passing the same epsilon to torch.optim.Adam.
- Batch size 32; repeated streams before batching give full batches. Training continues its shuffled stream across epochs; validation starts at the first validation sample for each evaluation.
- Maximum 25 epochs, early stopping on validation loss, patience 3, minimum improvement 0.
- Glorot uniform weights and zero biases; cross-entropy from logits.
- Use final.pt for reproduction: the published script saves the final model over its accuracy checkpoint. We retain best.pt and best_f1.pt separately as diagnostics rather than overwrite them.
- final_validation.json counts each validation window exactly once. Epoch monitoring includes repeated tail examples as in the reference. Best checkpoint metadata flags repeated monitoring metrics; fusion only directly checks a saved finite validation confusion matrix.
- validate_fusion.py reads the checkpoint's validation partition and stride. Three/four agreeing labels win, otherwise sum raw logits. No softmax fusion change.
- Default does not evaluate test partitions. --evaluate-test explicitly enables the reserved S1 portion and external S2a/S2b/S3a/S4a/S4b/S5a/S6a/S6b/S7a sessions. Reported training-script test metrics remain single antenna.
- Outputs remain in Results; Report is untouched.

Full dataset window counts (individual antennas): train 21648, validation 6692, reserved S1 test 6544. Test count is derived from lengths only, with no test predictions.

After a full training run, use its printed output directory:

```sh
python3 '/home/ubuntu/Deep Learning/Source/validate_fusion.py' --checkpoint '/home/ubuntu/Deep Learning/Results/training/RUN_DIRECTORY/final.pt'
```

Previous session-held-out settings remain available explicitly:

```sh
python3 '/home/ubuntu/Deep Learning/Source/train.py' --protocol session --epochs 30 --lr 0.001 --stride 170 --patience 20
```

## Verification

`python3 Source/check_sharp_alignment.py` checks actual window tensors against the pinned reference window function on synthetic recordings, chronological gaps, full-batch repetition, independent NumPy Adam updates including tiny gradients, and all 625 antenna-vote patterns.

A one-epoch, two-windows-per-trace GPU smoke run succeeded under Results/reproduction_setup/smoke_sharp_alignment. Its saved finite validation confusion matrix was reproduced through grouped four-antenna inference using the new temporal partition. These smoke scores are not benchmark results. No full reference training run has been started.

## Reproduction limits

The scientific settings and supported reference behaviors are aligned. This remains a PyTorch port, not a bitwise TensorFlow rerun: seed 42 is retained, and initialization draws, shuffle sequences, dropout masks and floating-point kernels differ across frameworks. Precomputed Doppler traces are used; their raw-CSI extraction pipeline was not rerun. The reference's final-checkpoint behavior is reproduced under a separate filename to preserve diagnostic checkpoints.
