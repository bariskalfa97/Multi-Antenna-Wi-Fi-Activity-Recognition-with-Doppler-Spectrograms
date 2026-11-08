# Multi-Antenna Wi-Fi Activity Recognition with Doppler Spectrograms

This project classifies human activity from Doppler traces collected on four aligned Wi-Fi antenna streams. It includes a multi-branch CNN, a supervised contrastive pretraining pipeline, evaluation and antenna-fusion scripts, and experiment outputs.

## Method

Each input is a single-antenna window with shape `1 × 340 × 100` (time × Doppler bins). The loader reads normalized power traces from NumPy arrays serialized in `.txt` files. It subtracts each Doppler bin's temporal mean over the full recording, then extracts 340-row windows at a stride of 30. The default SHARP protocol makes chronological 60/20/20 train, validation, and test partitions within sessions, with 31-row gaps between partitions to prevent overlapping windows.

The `ActivityCNN` processes each window through three parallel branches: max pooling, a shallow convolution, and a deeper convolutional branch. Their feature maps are concatenated, fused with a 1×1 convolution, flattened, and classified with a dense layer. The five labels are empty, sitting, walking, running, and jumping.

`contrastive_train.py` adds domain-aware supervised contrastive pretraining before classifier fine-tuning. Positive pairs include aligned antenna views of an event and same-activity examples from different training sessions; session `S1c` is held out for validation. `validate_fusion.py` and `evaluate_test.py` support fusion validation and optional held-out test evaluation. Single-antenna metrics are reported before decision fusion unless otherwise stated.

## Repository contents

- `Source/train.py` — trace loading, preprocessing, windowing, baseline training, and evaluation.
- `Source/architecture.py` — three-branch `ActivityCNN` implementation.
- `Source/contrastive_train.py` — supervised contrastive pretraining and fine-tuning.
- `Source/evaluate_test.py`, `Source/validate_fusion.py` — held-out evaluation and antenna-fusion validation.
- `Source/analyze_jumping.py`, `Source/explore_doppler.ipynb` — activity-specific analysis and Doppler exploration.
- `Source/check_sharp_alignment.py` — checks alignment with the SHARP reference protocol.
- `Results/` — training, evaluation, contrastive-learning, and fusion outputs, plus analysis notes.
- `WiFi_HAR_Report.pdf` — project report.

## Data and execution

The Wi-Fi traces are not included in this repository. The training scripts expect session directories such as `S1a`, `S1b`, and `S1c`, containing four aligned antenna streams per recording. Files must follow the naming pattern `S1a_W_stream_0.txt` and contain finite, normalized power arrays with 100 Doppler bins and at least 340 time rows. Pass the `doppler_traces` directory to the scripts with `--data-root`; run `python Source/train.py --help` or `python Source/contrastive_train.py --help` for available options.

The default preprocessing subtracts the mean using the entire recording, including samples later in time. It therefore matches the offline reference setup but is not suitable as-is for causal, real-time inference.

The code uses Python, NumPy, and PyTorch. The repository does not include a dependency lockfile or requirements file.

## Reference

The baseline protocol follows SHARP, which studies activity recognition using commodity Wi-Fi access points:

F. Meneghello, D. Garlisi, N. Dal Fabbro, I. Tinnirello, and M. Rossi, “SHARP: Environment and Person Independent Activity Recognition With Commodity IEEE 802.11 Access Points,” *IEEE Transactions on Mobile Computing*, vol. 22, no. 10, pp. 6160–6175, 2023. [https://doi.org/10.1109/TMC.2022.3185681](https://doi.org/10.1109/TMC.2022.3185681)
