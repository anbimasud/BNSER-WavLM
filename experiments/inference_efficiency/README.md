# Inference Efficiency Benchmark

This directory contains the standardized inference-only benchmark used to
measure computational behavior for the three BNSER WavLM configurations:

- **WavLM-FT** — full fine-tuning baseline.
- **WavLM-FT+Aug** — full fine-tuning with training-time augmentation.
- **WavLM-PEFT** — the combined PEFT configuration with the first 12 WavLM-Large Transformer layers frozen, Neutral-class weighting, and gradient accumulation during training.

The benchmark is designed to address the computational-efficiency concerns
raised during peer review without attributing inference-speed changes to
training-only configuration changes.

## 1. What this benchmark measures

The script measures only inference-time behavior:

- mean latency;
- median latency;
- P95 latency;
- throughput;
- real-time factor (RTF); and
- peak GPU memory allocated during the timed inference block.

Training, back-propagation, optimizer updates, gradient accumulation, loss
calculation, and data augmentation are not part of the timing measurement.

## 2. Standardized protocol

All three configurations are evaluated under the same protocol:

| Item | Setting |
|---|---|
| GPU | NVIDIA Tesla T4 |
| Sampling rate | 16 kHz |
| Input duration | 4.0 s |
| Input samples | 64,000 |
| Batch size | 1 |
| Input used for timing | Deterministic zero waveform with the standardized input shape |
| Model mode | `eval()` |
| Gradient computation | Disabled with `torch.inference_mode()` |
| Warm-up | 20 forward passes |
| Timed inference | 300 forward passes |
| Synchronization | CUDA synchronized before and after each timed forward |
| Latency | Mean, median, P95 |
| Throughput | `1000 / mean_latency_ms` |
| RTF | `mean_latency_seconds / input_duration_seconds` |
| GPU memory | `torch.cuda.max_memory_allocated()` after warm-up reset |
| Inference augmentation | Disabled |

The warm-up passes are excluded from the timing statistics. Peak GPU memory
statistics are reset after warm-up so that model-loading and warm-up memory do
not become part of the reported inference peak.

## 3. Model reconstruction and checkpoint verification

The benchmark reconstructs the same downstream architecture represented by
the BNSER checkpoints:

```text
WavLM-Large (24 Transformer layers)
        |
        v
Temporal mean pooling
        |
        v
1024-dimensional representation
        |
        v
Linear projector: 1024 -> 256
        |
        v
Linear classifier: 256 -> 6
```

The checkpoints contain a `model_state` dictionary whose keys use the
`backbone.` hierarchy. The script reconstructs that hierarchy directly using
`WavLMConfig` and `WavLMModel`; it does not instantiate
`WavLMForSequenceClassification` and then leave its downstream head newly
initialized.

Before loading a checkpoint, the script checks:

1. exact model/checkpoint key-set equality;
2. exact tensor-shape equality; and
3. successful `strict=True` state-dict loading.

The run stops if any key or shape mismatch is detected. This is intentional:
a benchmark must not silently continue with partially initialized downstream
weights.

## 4. Model-specific configuration

### WavLM-FT

All model parameters are trainable. No inference-time augmentation is used.

Expected total/trainable parameters:

```text
315,717,062 total
315,717,062 trainable
0 frozen
```

### WavLM-FT+Aug

The architecture and trainable-parameter count are the same as WavLM-FT.
The augmentation described in the manuscript is a **training-time** procedure
and is disabled for this inference benchmark.

Expected total/trainable parameters:

```text
315,717,062 total
315,717,062 trainable
0 frozen
```

### WavLM-PEFT

The first 12 of the 24 WavLM-Large Transformer layers are frozen during
optimization. The upper 12 layers and classification components remain
trainable.

Expected parameter counts:

```text
315,717,062 total
164,550,822 trainable
151,166,240 frozen
```

The corresponding trainable-parameter reduction is approximately 47.88%.

This reduction is a **training-time parameter-efficiency measure**. It must
not be described as a 47.88% reduction in inference computation or inference
depth.

## 5. Forward-path verification for Model 3

The benchmark separately registers forward hooks on the 24 WavLM-Large
Transformer encoder layers and runs the same 300-forward-pass workload.

The hooks are **not active during the latency benchmark**. This separation is
important because hooks themselves introduce Python-side overhead and should
not be allowed to make Model 3 incomparable with Models 1 and 2.

The verification requires:

```text
24 Transformer layers
300 executions per layer
```

Thus a successful verification establishes that all 24 encoder layers remain
in the Model 3 inference forward path.

The verification is written to:

```text
results/layer_execution_verification.csv
```

The file records the layer index, observed execution count, expected count,
and verification status.

## 6. Why augmentation is disabled during inference

WavLM-FT+Aug and WavLM-PEFT use augmentation during training according to the
manuscript-defined procedure. That augmentation is not part of validation,
test, or inference.

Accordingly, the inference benchmark does not apply noise, pitch shift, time
stretch, or any other training augmentation. The comparison therefore measures
the deployed inference architectures under the same input and execution
protocol.

The benchmark should not be described as measuring an effect of augmentation
at inference time.

## 7. Output files

After a successful run, the script creates:

```text
experiments/inference_efficiency/
├── README.md
├── inference_efficiency.py
├── benchmark_config.yaml
├── raw/
│   ├── model1_wavlm_ft_timings.csv
│   ├── model2_wavlm_ft_aug_timings.csv
│   ├── model3_wavlm_peft_timings.csv
│   └── run_metadata.json
└── results/
    ├── efficiency_results.csv
    └── layer_execution_verification.csv
```

Each timing file contains the 300 individual latency observations. The summary
file is calculated directly from those observations and is not reconstructed
from manuscript values.

## 8. Reproducibility and environment

The benchmark is intended to be run on the same hardware and software
environment used for the reported standardized measurements. The script
requires the NVIDIA Tesla T4 and stops if another GPU is detected.

The checkpoint paths are environment-specific and are therefore stored in
`benchmark_config.yaml` rather than embedded throughout the Python code.
Update only those paths when running in another environment; do not change the
model architecture or benchmark settings without rerunning and documenting a
new measurement.

The script records the GPU name, PyTorch version, Python version, CUDA version,
Transformers version, input specification, timing configuration, and model
architecture in `raw/run_metadata.json`.

## 9. Interpretation for the manuscript

The benchmark supports a distinction between two different concepts:

- **Trainable-parameter efficiency:** Model 3 updates substantially fewer
  parameters during optimization because the first 12 Transformer layers are
  frozen.
- **Inference efficiency:** measured directly from the complete forward pass
  under the standardized T4 protocol.

Because all 24 Transformer layers execute during Model 3 inference, the
trainable-parameter reduction should not be presented as an equivalent
reduction in inference depth, inference computation, or inference latency.

Likewise, any small empirical latency difference between WavLM-FT and
WavLM-FT+Aug should be reported as an observed benchmark result rather than
attributed causally to training-time augmentation.

## 10. Relation to the revised manuscript

The standardized measurements replace the earlier computational-efficiency
measurements that used a different or insufficiently controlled protocol.
Historical latency, throughput, memory, or CPU-utilization values should not
be mixed with the standardized results.

The final manuscript should use only the results generated by this standardized
protocol after a successful run, together with the corresponding raw timing
files and Model 3 layer-verification file.
