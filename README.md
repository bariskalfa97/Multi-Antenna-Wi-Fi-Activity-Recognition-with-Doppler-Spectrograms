# Multi-Antenna Wi-Fi Activity Recognition with Doppler Spectrograms

A research project exploring human activity recognition from Wi-Fi channel measurements using Doppler spectrograms captured across multiple antennas.

> **Repository note:** This README draft is based on the project title and public research context. The linked repository contents were not accessible during drafting, so implementation-specific details (scripts, dependencies, dataset, activity labels, and reported results) are intentionally not guessed.

## Overview

Human movement changes the wireless channel over time. Those changes can be represented in the Doppler domain, where motion-related frequency shifts become visible as time-varying spectrogram patterns. Using measurements from multiple antennas can provide complementary views of the same activity.

This project studies a pipeline that connects multi-antenna Wi-Fi measurements to activity labels:

1. Load and prepare Wi-Fi channel measurements.
2. Extract Doppler representations for the available antenna channels.
3. Organize the resulting spectrograms as model inputs.
4. Train and evaluate an activity-recognition method.

The exact signal format, preprocessing, model architecture, and evaluation protocol should be documented from the repository implementation and its accompanying study.

## Repository Contents

See the source folders and notebooks in this repository for the current implementation. Add a file-by-file guide here once the project structure is confirmed.

## Data

Document the dataset source, access instructions, activity classes, participant and recording counts, antenna configuration, and train/validation/test split here. Do not commit private or restricted data to the repository.

## Requirements

The runtime and package requirements depend on the implementation. Check the repository's dependency file (for example, `requirements.txt` or `environment.yml`) before setting up an environment.

## Getting Started

After confirming the repository's actual dependency and entry-point files:

1. Clone the repository.
2. Create and activate an environment using the documented Python/runtime version.
3. Install the dependencies listed by the project.
4. Obtain and place the dataset as described in the data documentation.
5. Run the preprocessing, training, and evaluation entry points documented by the project.

The exact commands are omitted until the scripts and their arguments can be verified.

## Method and Evaluation

Describe the implemented method here, including:

- CSI or other channel-measurement representation and sampling details
- Doppler transform and spectrogram parameters
- How antenna channels are combined or modeled
- Activity labels and dataset split strategy
- Model architecture, training settings, and evaluation metrics
- Baselines and reproducibility details

Report results with the evaluation split and metric definitions. Avoid presenting a single accuracy number without its protocol and dataset context.

## Citation

For background on Wi-Fi human activity recognition using Doppler information, see the SHARP paper:

> F. Meneghello, D. Garlisi, N. Dal Fabbro, I. Tinnirello, and M. Rossi, “SHARP: Environment and Person Independent Activity Recognition With Commodity IEEE 802.11 Access Points,” *IEEE Transactions on Mobile Computing*, vol. 22, no. 10, pp. 6160–6175, 2023. [doi:10.1109/TMC.2022.3185681](https://doi.org/10.1109/TMC.2022.3185681).

```bibtex
@article{meneghello2023sharp,
  author  = {Meneghello, Francesca and Garlisi, Domenico and Dal Fabbro, Nicol\'o and Tinnirello, Ilenia and Rossi, Michele},
  title   = {{SHARP}: Environment and Person Independent Activity Recognition With Commodity {IEEE} 802.11 Access Points},
  journal = {IEEE Transactions on Mobile Computing},
  year    = {2023},
  volume  = {22},
  number  = {10},
  pages   = {6160--6175},
  doi     = {10.1109/TMC.2022.3185681}
}
```

If this repository directly uses or reproduces SHARP, describe that connection here; the linked repository contents were unavailable for verification while drafting this README.

## License

See the repository's `LICENSE` file for the applicable terms. If no license file is present, clarify the intended license before reusing or redistributing the code.

## Acknowledgments

Add verified acknowledgments, funding information, and upstream tools or datasets here.
