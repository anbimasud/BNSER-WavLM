# Training Experiments

Reviewer-facing training implementation for the three WavLM-Large configurations reported in the BNSER Noakhali dialect speech emotion recognition study.

## Structure

The notebooks are intentionally short experiment drivers. Shared scientific code is under `common/`; model-specific settings are under `configs/`; each model directory contains its notebook and a concise README.

```text
training/
├── README.md
├── configs/
├── common/
├── model1_wavlm_ft/
├── model2_wavlm_aug/
├── model3_wavlm_peft/
└── outputs/
```

## Dataset

The speech corpus is not redistributed. Set:

```bash
export BNSER_DATA_DIR=/path/to/FInal_2300_Data_Spk_wise_split
```

The root must contain the pre-defined speaker-independent folders `Train/`, `Val/`, and `Test/`. The code does not create a new split.

Expected sizes are 1,610 training, 345 validation, and 345 test samples.

## Shared protocol

All configurations use WavLM-Large, six emotion classes, 16-kHz mono audio, fixed four-second inputs, AdamW, learning rate `1e-5`, seed 42, maximum 50 epochs, and five-epoch early stopping patience. Physical batch size is 4.

The fixed-length preprocessing follows the original model-specific experimental implementations:

- **WavLM-FT:** `librosa.util.fix_length(..., size=64000)` is used for training, validation, and test inputs.
- **WavLM-FT+Aug:** training-time augmentation is applied first, followed by a random four-second crop for training samples; validation and test samples use a centered four-second crop.
- **WavLM-PEFT:** the same preprocessing order as WavLM-FT+Aug is used: training-time augmentation followed by a random four-second training crop, with centered four-second validation/test crops.

Validation loss is averaged across validation batches, matching the original training implementations.

## Model-specific protocol

**WavLM-FT:** full fine-tuning, no augmentation, standard cross-entropy.

**WavLM-FT+Aug:** full fine-tuning plus training-only augmentation. Each training sample has 0.35 probability of augmentation; selected transformations are 40% noise, 30% pitch shift, and 30% time stretch. Validation and test inputs are unaugmented.

**WavLM-PEFT:** same augmentation, first 12 of 24 Transformer layers frozen, Neutral loss weight 1.4, four-step gradient accumulation (effective batch size 16), weighted cross-entropy. The three Model-3 components are introduced jointly and are not interpreted as isolated causal effects.

## Reproducibility boundaries

The code preserves the documented scientific configuration and fixed split. Exact decimal reproduction can vary with CUDA/PyTorch/Transformers/librosa versions, hardware, and stochastic execution. The manuscript reports single-run results using seed 42.

Checkpoints and local outputs are ignored by Git. The repository does not redistribute the underlying speech data.
