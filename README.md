# BNSER-WavLM

Reproducibility repository for the study **Parameter-Efficient Fine-Tuning of WavLM for Low-Resource Noakhali Bangla Dialect Speech Emotion Recognition With Controlled Audio Augmentation**.

This repository contains the code, configuration documentation, analysis scripts, machine-readable result tables, and supplementary evidence associated with the revised BNSER-WavLM manuscript. It is intended to make the reported methodology and analyses inspectable and reproducible without overstating what can be concluded from the available data.

> **Reproducibility boundary.** The repository does not redistribute the underlying speech corpus or the archived evaluation directories used by some post-hoc analyses. Exact reruns of the affected analyses therefore require the corresponding data/evaluation artifacts and the necessary permissions.

## 1. Study overview

The Bangla Noakhali Dialect Speech Emotion Recognition (BNSER) corpus contains **2,300 recordings from 25 native Noakhali speakers** across six emotion classes: Angry, Disgust, Fear, Happy, Neutral, and Sad.

The corpus contains:

- **1,800 scripted recordings**; and
- **500 drama-derived recordings**.

The experiments use a strict speaker-independent partition:

| Split | Speakers | Samples |
|---|---:|---:|
| Training | 18 | 1,610 |
| Validation | 3 | 345 |
| Test | 4 | 345 |
| **Total** | **25** | **2,300** |

The fixed test set contains 345 recordings from four held-out speakers. All 50 Neutral test recordings come from one held-out speaker and the Neutral class is represented only by scripted recordings. These characteristics are treated as evaluation limitations rather than being interpreted as evidence of source-independent emotion recognition.

For the five non-Neutral classes, the test set contains 250 scripted and 45 drama-derived recordings. Source-stratified results are therefore reported as **source-associated differences**, not as causal effects of recording or performance style.

Detailed corpus statistics are provided in [`data/dataset_statistics.md`](data/dataset_statistics.md).

## 2. Model configurations

All three configurations use **WavLM-Large** with 24 Transformer encoder layers for six-class BNSER emotion classification.

| Configuration | Trainable parameters | Training augmentation | Layer freezing | Loss | Additional components |
|---|---:|---|---|---|---|
| **WavLM-FT** | 315.7M | None | None | Cross-entropy | — |
| **WavLM-FT+Aug** | 315.7M | 35% | None | Cross-entropy | Controlled noise/pitch/time-stretch augmentation |
| **WavLM-PEFT** | 164.6M | 35% | First 12 layers | Weighted cross-entropy | Neutral weight 1.4; gradient accumulation 4 |

For WavLM-PEFT, the first 12 encoder layers are frozen and the upper 12 are trainable. The trainable-parameter count decreases from 315.7M to 164.6M, corresponding to a **47.88% reduction in trainable parameters during optimization**.

This reduction must not be interpreted as a reduction in inference depth or inference computation: the frozen layers remain part of the same 24-layer forward architecture and execute during inference.

The three Model-3 components—layer freezing, Neutral-class weighting, and gradient accumulation—were introduced jointly. The study therefore does **not** claim an isolated causal contribution for any one of them.

Training configuration documentation is available under [`experiments/training/`](experiments/training/).

## 3. Training-time augmentation

WavLM-FT+Aug and WavLM-PEFT apply controlled acoustic augmentation **only during training** with a 35% per-sample selection probability.

Among selected augmented samples, the implemented transformations are:

| Transformation | Probability among augmented samples | Range |
|---|---:|---|
| Noise addition | 40% | 0.5–2.0% of signal level |
| Pitch shift | 30% | −0.5 to +0.5 semitones |
| Time stretch | 30% | 0.95–1.05 rate |

Validation and test inference use the original, non-augmented inputs.

The supplied archive does **not** contain completed human listening-test data for perceptual validation of augmentation. Accordingly, the repository does not claim independent perceptual confirmation that the transformations preserve perceived emotional content.

## 4. Main reported test performance

The following values are the reported point estimates on the fixed 345-sample speaker-independent test set.

| Model | Accuracy | Weighted F1 | Balanced Accuracy |
|---|---:|---:|---:|
| WavLM-FT | 75.07% | 0.7568 | 0.7531 |
| WavLM-FT+Aug | 82.61% | 0.8257 | 0.8184 |
| WavLM-PEFT | 88.70% | 0.8872 | 0.8881 |

The observed WavLM-PEFT minus WavLM-FT accuracy difference is **13.62 percentage points** (306/345 versus 259/345). This is an observed difference on the evaluated test set; it is not presented as a population-level superiority estimate.


## 5. Bootstrap uncertainty analysis

A non-parametric bootstrap analysis uses **10,000 sample-level resamples** of the fixed 345-instance test set with 95% percentile intervals. The reported intervals are conditional on this evaluated test set and do not quantify alternative speaker partitions or independent training-run variability.

### Reported 95% CIs

| Model | Accuracy | Weighted F1 | Balanced Accuracy |
|---|---|---|---|
| WavLM-FT | 75.07% (70.43–79.42) | 0.7568 (0.7118–0.8001) | 0.7531 (0.7097–0.7951) |
| WavLM-FT+Aug | 82.61% (78.55–86.38) | 0.8257 (0.7853–0.8644) | 0.8184 (0.7800–0.8559) |
| WavLM-PEFT | 88.70% (85.22–91.88) | 0.8872 (0.8530–0.9188) | 0.8881 (0.8541–0.9195) |

Paired bootstrap accuracy differences on the same test instances are:

| Comparison | Difference | 95% CI |
|---|---:|---:|
| WavLM-FT+Aug − WavLM-FT | +7.54 pp | 2.90 to 12.46 pp |
| WavLM-PEFT − WavLM-FT+Aug | +6.09 pp | 1.74 to 10.43 pp |
| WavLM-PEFT − WavLM-FT | +13.62 pp | 8.70 to 18.55 pp |

The manuscript and repository distinguish these fixed-test-set intervals from uncertainty arising from repeated training runs or alternative speaker partitions.

The analysis implementation is in [`experiments/bootstrap_ci/`](experiments/bootstrap_ci/). The archived manuscript-facing summary is in [`results/bootstrap_results.csv`](results/bootstrap_results.csv).

## 6. Freezing-depth sensitivity

The manuscript additionally evaluates freezing-depth sensitivity within the combined WavLM-PEFT configuration on the same 345-sample test set. The evaluated depths are 6, 8, 12, 16, and 18 frozen layers.

| Frozen layers | Trainable parameters | Trainable-parameter reduction | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 240.13M | 23.94% | 91.88% | 91.59% | 91.92% | 92.04% |
| 8 | 214.94M | 31.92% | 89.57% | 89.26% | 89.57% | 89.77% |
| 12 | 164.55M | 47.88% | 88.70% | 88.81% | 88.82% | 88.72% |
| 16 | 114.16M | 63.84% | 61.74% | 61.24% | 61.54% | 61.51% |
| 18 | 88.97M | 71.82% | 61.74% | 60.76% | 59.85% | 60.29% |

The 12-layer configuration is the pre-existing WavLM-PEFT reference run; the additional depths were evaluated once on the same fixed split. The sensitivity analysis therefore supports describing 12 layers as a **performance–parameter-efficiency compromise among the evaluated settings**, not as a globally optimal freezing depth.

## 7. Source-stratified evaluation

Source-stratified evaluation is restricted to the 295 non-Neutral test recordings for which both source types are represented: 250 scripted and 45 drama-derived.

| Model | Scripted | Drama-derived | Scripted − drama-derived |
|---|---:|---:|---:|
| WavLM-FT | 80.00% (200/250) | 48.89% (22/45) | 31.11 pp |
| WavLM-FT+Aug | 92.00% (230/250) | 66.67% (30/45) | 25.33 pp |
| WavLM-PEFT | 92.80% (232/250) | 64.44% (29/45) | 28.36 pp |

These values demonstrate a source-associated performance difference within the evaluated corpus. They do **not** isolate a causal effect of performance style or recording source because source is entangled with speaker composition and acquisition/recording conditions.

The machine-readable summary is [`results/source_stratified_results.csv`](results/source_stratified_results.csv), with the corresponding analysis under [`experiments/source_stratified/`](experiments/source_stratified/).

## 8. Inference-efficiency benchmark

Inference efficiency was evaluated separately from trainable-parameter efficiency using a common benchmark protocol:

- NVIDIA Tesla T4
- 16 kHz input
- fixed 4.0-second waveform
- batch size 1
- `eval()` mode
- gradient computation disabled with `torch.inference_mode()`
- 20 warm-up iterations
- 300 CUDA-synchronized timed iterations
- augmentation disabled during inference

| Model | Mean latency (ms) | Median (ms) | P95 (ms) | Throughput (samples/s) | RTF | Peak GPU memory (GB) |
|---|---:|---:|---:|---:|---:|---:|
| WavLM-FT | 63.22 | 63.28 | 65.00 | 15.82 | 0.0158 | 1.2586 |
| WavLM-FT+Aug | 65.66 | 65.70 | 67.65 | 15.23 | 0.0164 | 1.2586 |
| WavLM-PEFT | 66.75 | 66.73 | 68.29 | 14.98 | 0.0167 | 1.2586 |

The latency values are benchmark observations under this protocol. The difference between WavLM-FT and WavLM-FT+Aug is **not attributed to augmentation**, because augmentation is disabled at inference and the reported benchmark measures the model forward path.

For WavLM-PEFT, forward-path verification confirms that **all 24 WavLM Transformer layers execute during inference**, including the 12 frozen layers. Therefore, the study does not claim an inference-speed or inference-memory reduction caused by layer freezing.

The machine-readable benchmark table is [`results/efficiency_results.csv`](results/efficiency_results.csv), and the benchmark implementation/evidence is under [`experiments/inference_efficiency/`](experiments/inference_efficiency/).

## 9. Annotation reliability

The BNSER corpus contains 2,300 speech samples. Inter-annotator agreement was assessed on 2,200 annotated samples. Three native Noakhali-dialect annotators independently labelled the corpus. The revised manuscript reports pairwise Cohen's κ values of:

- Annotator 1 vs Annotator 2: **0.805011**
- Annotator 1 vs Annotator 3: **0.802409**
- Annotator 2 vs Annotator 3: **0.831207**
- Arithmetic mean: **0.812876 ≈ 0.813**

The previously disputed **90.56%** agreement figure is not used in the revised reporting. The repository does not fabricate a new agreement calculation from unavailable raw annotator label vectors.

The Anotation summary is [`results/kappa_results.xlsx`](results/kappa_results.xlsx), and the corresponding analysis is under [`analysis/kappa_analysis/`](analysis/kappa_analysis/).

## 10. Confusion-matrix analysis

[`analysis/confusion_matrix/`](analysis/confusion_matrix/) contains the confusion-matrix analysis and machine-readable matrices.


## 11. WavLM backbone interpretation

WavLM-Large is treated as a **task-motivated backbone choice**, not as a demonstrated universal optimum for Noakhali SER. The revised manuscript explicitly notes that wav2vec 2.0, HuBERT, XLS-R, and IndicWav2Vec are relevant alternative self-supervised speech encoders, but no matched BNSER benchmark among these backbones was conducted in the present study.

Accordingly, the repository does not claim that WavLM is superior to those alternatives.

## 12. Reproducibility scope

The repository separates several concepts that should not be conflated:

- **trainable-parameter efficiency** vs. inference efficiency;
- **source-associated differences** vs. causal source or performance-style effects;
- **fixed-test-set bootstrap uncertainty** vs. training-run or population uncertainty;
- **combined Model-3 configuration effects** vs. isolated component effects; and
- documented augmentation settings vs. independent perceptual validation.

The supplied training notebooks and analysis scripts are intended to document the experimental procedures. Exact reruns may additionally require the BNSER speech data, archived prediction/evaluation artifacts, and the original computational environment.

## 13. Repository structure

```text
BNSER-WavLM/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── dataset_statistics.md
├── experiments/
│   ├── training/
│   │   ├── common/
│   │   ├── configs/
│   │   ├── model1_wavlm_ft/
│   │   ├── model2_wavlm_aug/
│   │   ├── model3_wavlm_peft/
│   │   ├── outputs/
│   │   └── README.md
│   ├── bootstrap_ci/
│   ├── source_stratified/
│   ├── freezing_depth/
│   └── inference_efficiency/
├── analysis/
│   ├── confusion_matrix/
│   └── kappa_analysis/
├── results/
│   ├── test_predictions.csv
│   ├── bootstrap_results.csv
│   ├── source_stratified_results.csv
│   ├── efficiency_results.csv
│   ├── layer_execution_verification.csv
│   ├── kappa_results.xlsx
│   ├── pairwise_kappa.csv
│   └── kappa_mean.csv
└── supplementary/
    └── merged_annotations.csv
```

## 14. Software environment

The reported experiments were conducted in a Google Colab/NVIDIA GPU environment using PyTorch and Hugging Face Transformers. See [`requirements.txt`](requirements.txt) for the repository-level Python dependencies.

The exact CUDA/PyTorch build should be selected to match the available NVIDIA driver and hardware. The benchmark hardware reported in the manuscript is an NVIDIA Tesla T4.
