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

The training pipeline uses Hugging Face `WavLMForSequenceClassification`
with the `microsoft/wavlm-large` configuration. The shared training loop saves
the selected model directly with:

```text
torch.save(model.state_dict(), output_dir / "best_model.pt")
```

Therefore, each benchmark checkpoint is a **raw PyTorch state dictionary**;
it is not wrapped inside a `model_state` field. The state-dict keys follow the
native `WavLMForSequenceClassification` hierarchy (for example, `wavlm.*`,
`projector.*`, and `classifier.*`).

The benchmark reconstructs the exact Hugging Face architecture from
`WavLMConfig` and `WavLMForSequenceClassification`, then loads the raw
state dictionary with strict validation.

The benchmark configuration explicitly records the manuscript-aligned
architecture dimensions:

- 1024-dimensional WavLM hidden representation;
- 24 Transformer layers;
- 256-dimensional classifier projection; and
- 6 output classes.

Before model construction, the script checks these values against the
loaded Hugging Face `WavLMConfig`, including
`classifier_proj_size == 256`. The benchmark stops if the configured
architecture and the actual Hugging Face architecture disagree.

Before loading a checkpoint, the script checks:

1. exact model/checkpoint key-set equality;
2. exact tensor-shape equality; and
3. successful `strict=True` state-dict loading.

The run stops if any key or shape mismatch is detected. This is intentional:
a benchmark must not silently continue with partially initialized or randomly
initialized model weights.

For Model 3, the first 12 WavLM Transformer layers are frozen **after** the
trained checkpoint is loaded. This reproduces the training-time trainable/frozen
parameter configuration for parameter-count reporting while leaving the full
24-layer forward architecture intact for inference.

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

The verification is written to the repository-level canonical results directory:

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

The repository uses one canonical location for manuscript-facing benchmark
results. After a successful run, the output layout is:

```text
experiments/inference_efficiency/
├── README.md
├── inference_efficiency.py
├── benchmark_config.yaml
└── inference_raw/
    ├── model1_wavlm_ft_timings.csv
    ├── model2_wavlm_ft_aug_timings.csv
    ├── model3_wavlm_peft_timings.csv
    └── run_metadata.json

results/
├── efficiency_results.csv
└── layer_execution_verification.csv
```

The `results/` files are the canonical machine-readable outputs used for
manuscript-facing evidence. The committed `results/efficiency_results.csv` is
an **archived reported-measurement artifact**: its schema is now explicitly locked into the benchmark script, but the original raw 300-iteration timing
files and run metadata were not preserved in the repository, so the exact
historical script-run identity cannot be reconstructed from the CSV alone.
This distinction is intentional and avoids claiming a numerical rerun that was
not performed during repository audit.

When the benchmark is rerun on the required NVIDIA Tesla T4, the script writes
the same canonical 23-column summary schema, plus the 300 individual timing
observations and run metadata. The metadata records the runtime versions,
checkpoint/timing paths, and SHA-256 hashes of the benchmark script and
configuration used for that new run.

The output directory is configured explicitly as `../../results` in
`benchmark_config.yaml`, relative to the `experiments/inference_efficiency/`
directory containing the benchmark script. This avoids maintaining a second
benchmark-specific `results/` directory with potentially conflicting copies.

## 8. Reproducibility and environment

The benchmark is intended to be run on the same hardware and software
environment used for the reported standardized measurements. The script
requires the NVIDIA Tesla T4 and stops if another GPU is detected.

The checkpoint location is environment-specific and is represented by one
checkpoint-root directory plus a filename for each model. By default, the
resolver uses the repository-local `./checkpoints` directory. The YAML values
therefore contain only `Model_WavLM_FT_m1.pt`, `Model_WavLM_FT_Aug_m2.pt`, and
`Model_WavLM_PEFT_m3.pt`; do **not** prepend `checkpoints/` to those values.
For another environment, set `BNSER_CHECKPOINT_ROOT` to the directory that
actually contains the three checkpoint files. Absolute checkpoint paths are
rejected. This prevents accidental `checkpoints/checkpoints/...` resolution.

Update only the checkpoint-root environment setting when running in another
environment; do not change the model architecture or benchmark settings
without rerunning and documenting a new measurement.

The script records the GPU name, PyTorch version, Python version, CUDA version,
Transformers version, input specification, timing configuration, canonical CSV
schema, checkpoint/timing provenance, and SHA-256 hashes of the benchmark
script/configuration used for that run. The complete metadata is stored in
`inference_raw/run_metadata.json`.

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
