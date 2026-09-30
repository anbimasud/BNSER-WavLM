# Freezing-Depth Sensitivity Experiment

This directory contains the reviewer-facing implementation of the freezing-depth sensitivity analysis for the BNSER-WavLM study.

## Purpose

The study uses a 12-layer freezing configuration as the reference WavLM-PEFT setting. To examine sensitivity to the freezing boundary, the first **6, 8, 16, and 18 Transformer layers** were additionally evaluated on the same speaker-independent test partition.

This experiment is a **freezing-depth sensitivity analysis within the full WavLM-PEFT configuration**. It is not a component-wise ablation of augmentation, class weighting, or gradient accumulation.

The 12-layer result is the **pre-existing WavLM-PEFT reference run reported in the manuscript**. It is not retrained by `freezing_depth.py`. The additional depths are trained once on the same predefined Train/Validation/Test split.

## Fixed experimental configuration

All additional freezing-depth runs retain the other WavLM-PEFT settings reported in the manuscript:

- Backbone: `microsoft/wavlm-large`
- WavLM Transformer layers: 24
- Input sampling rate: 16 kHz
- Input duration: 4 seconds
- Dataset partition: predefined speaker-independent Train/Validation/Test split
- Train / Validation / Test samples: 1,610 / 345 / 345
- Test speakers: 4 held-out speakers
- Optimizer: AdamW
- Learning rate: `1e-5`
- Maximum epochs: 50
- Early-stopping patience: 5 epochs
- Random seed: 42
- Physical batch size: 4
- Gradient accumulation: 4 steps
- Effective batch size: 16
- Training augmentation: 35% of training samples
- Augmentation type allocation: 40% additive Gaussian noise, 30% pitch shift, 30% time stretch
- Pitch shift range: ±0.5 semitones
- Time-stretch range: ±5%
- Noise scale: 0.5–2% of the current waveform peak amplitude
- Class-weighted cross-entropy: Neutral = 1.4; all other classes = 1.0
- Validation and test audio: no augmentation

The validation-loss aggregation follows the Model-3 reference implementation (`batch_mean`). The original four-step accumulation behavior is preserved: an incomplete final accumulation group is not stepped.

## Evaluated freezing depths

| Frozen Transformer layers | Status |
|---:|---|
| 6 | Additional sensitivity run |
| 8 | Additional sensitivity run |
| 12 | Existing WavLM-PEFT reference run; not retrained here |
| 16 | Additional sensitivity run |
| 18 | Additional sensitivity run |

For a freezing depth `d`, layers `0` through `d-1` are frozen and the remaining WavLM Transformer layers remain trainable. The convolutional feature encoder and classification head are not removed from the model.

## Why this experiment is separate from the primary training notebooks

The primary training directory contains the three reported model configurations:

```text
experiments/training/
├── model1_wavlm_ft/
├── model2_wavlm_aug/
└── model3_wavlm_peft/
```

This directory contains the additional reviewer-requested sensitivity analysis. It does not maintain a second implementation of the dataset pipeline, augmentation, model construction, or training loop. Instead, `freezing_depth.py` imports those components from:

```text
experiments/training/common/
```

This keeps the repository's training logic in one place and reduces the risk of implementation drift between the primary WavLM-PEFT model and the freezing-depth experiments.

## Running the experiment

From the repository root, set the local path to the BNSER dataset. The dataset itself is not redistributed in this repository.

```bash
export BNSER_DATA_DIR=/path/to/FInal_2300_Data_Spk_wise_split
python experiments/freezing_depth/freezing_depth.py
```

The expected dataset structure is:

```text
FInal_2300_Data_Spk_wise_split/
├── Train/
├── Val/
└── Test/
```

The script validates that the predefined split contains 1,610 training, 345 validation, and 345 test samples before any training begins. It does **not** create a new random train/validation/test split.

### Optional verification against the existing 12-layer reference

If the prediction CSV from the reported 12-layer WavLM-PEFT run is available locally, pass it to the script:

```bash
python experiments/freezing_depth/freezing_depth.py \
    --reference-predictions /path/to/model3/test_predictions.csv
```

The script then compares the speaker, emotion, and audio-file identity of every additional run with the existing 12-layer reference test set.

## Outputs

The script creates one isolated directory for each additional freezing depth:

```text
experiments/freezing_depth/runs/
├── freeze_6/
├── freeze_8/
├── freeze_16/
└── freeze_18/
```

Each run contains the machine-readable training and evaluation artifacts needed for local verification, including the best checkpoint, training history, test predictions, class-wise metrics, and experiment configuration.

The reviewer-facing summary files are:

```text
freezing_depth_results.csv
same_test_set_verification.csv
```

These summary files are generated from the completed runs rather than manually entering model results.

## Reproducibility and interpretation

The four additional freezing-depth configurations are trained once using the
same predefined speaker-independent partition as the pre-existing 12-layer
reference. The 12-layer result is not retrained by `freezing_depth.py`; its
test-set identity is verified against the additional runs through
`same_test_set_verification.csv`.

The sensitivity analysis is interpreted descriptively. The 12-layer configuration
is retained as a performance–parameter-efficiency compromise among the evaluated
settings, not as an accuracy optimum.

## Relation to the manuscript

The implementation corresponds to the manuscript's description that:

> the pre-existing 12-layer WavLM-PEFT reference was supplemented by evaluations freezing the first 6, 8, 16, and 18 layers on the same speaker-independent test set.

The code is intentionally limited to this sensitivity analysis and does not alter the reported primary model definitions.
