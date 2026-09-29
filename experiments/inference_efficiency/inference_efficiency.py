"""
Standardized inference-efficiency benchmark for the BNSER-WavLM models.

This script measures inference-time behavior only. It does not perform
training, augmentation, gradient computation, or optimizer updates.

The three configurations use the same WavLM-Large inference architecture:

    WavLM-Large (24 Transformer layers)
        -> temporal mean pooling
        -> 1024 -> 256 projector
        -> 256 -> 6 classifier

Model 3 freezes the first 12 Transformer layers during optimization, but the
frozen layers remain in the inference forward path. The script therefore
reports trainable-parameter counts separately from measured inference
latency/throughput and independently verifies that all 24 Transformer layers
execute during timed inference.

IMPORTANT REPRODUCIBILITY NOTE
------------------------------
The benchmark uses the exact checkpoint format produced by the BNSER training
experiments: ``torch.save(model.state_dict(), ...)``. The model is reconstructed
from the same Hugging Face ``WavLMForSequenceClassification`` configuration and
loaded with strict=True after exact key/shape validation. This preserves the
training checkpoint key hierarchy and avoids silently benchmarking a newly
initialized downstream classification head.

The benchmark input is a deterministic zero waveform with the same shape as
the manuscript test-time input: batch size 1, 16 kHz, 4 seconds (64,000
samples). This is a shape-controlled inference benchmark; it is not an
accuracy evaluation.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import WavLMConfig, WavLMForSequenceClassification


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
CONFIG_PATH = ROOT / "benchmark_config.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark configuration not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("benchmark_config.yaml must contain a YAML mapping.")

    return config


CONFIG = load_config(CONFIG_PATH)


def resolve_checkpoint_path(checkpoint_value: str) -> Path:
    """Resolve a checkpoint filename relative to the checkpoint directory.

    ``BNSER_CHECKPOINT_ROOT`` may point to any local/cloud-mounted directory
    that *contains* the benchmark checkpoints. If it is not set, the default
    is the repository-local ``./checkpoints`` directory. Configuration values
    therefore contain filenames such as ``Model_WavLM_FT_m1.pt`` rather than
    ``checkpoints/Model_WavLM_FT_m1.pt``.
    """
    checkpoint_root_env = str(CONFIG.get("checkpoint_root_env", "BNSER_CHECKPOINT_ROOT"))
    checkpoint_root = Path(
        os.environ.get(checkpoint_root_env, REPO_ROOT / "checkpoints")
    ).expanduser()
    checkpoint_path = Path(checkpoint_value).expanduser()

    if checkpoint_path.is_absolute():
        raise ValueError(
            f"Checkpoint paths must be relative to the configured checkpoint root; "
            f"got absolute path: {checkpoint_path}"
        )

    return checkpoint_root / checkpoint_path


RAW_DIR = ROOT / CONFIG["outputs"]["timing_directory"]
RESULTS_DIR = ROOT / CONFIG["outputs"]["directory"]
RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fixed manuscript-aligned constants
# ---------------------------------------------------------------------------

MODEL_NAMES = ["WavLM-FT", "WavLM-FT+Aug", "WavLM-PEFT"]

EXPECTED_CLASS_ORDER = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
]

EXPECTED_TOTAL_PARAMS = 315_717_062
EXPECTED_M3_TRAINABLE_PARAMS = 164_550_822
EXPECTED_M3_FROZEN_PARAMS = 151_166_240
EXPECTED_M3_TRAINABLE_REDUCTION = 47.88
EXPECTED_TRANSFORMER_LAYERS = 24

# Canonical manuscript-facing summary schema.  Keep this explicit so the
# benchmark implementation cannot silently drift from the archived evidence
# table.  Run-specific checkpoint and raw-timing paths belong in the metadata
# record rather than in this summary CSV.
CANONICAL_RESULT_COLUMNS = [
    "model",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "throughput_samples_per_s",
    "rtf",
    "peak_gpu_memory_allocated_gb",
    "total_parameters",
    "trainable_parameters",
    "frozen_parameters",
    "gpu",
    "pytorch_version",
    "python_version",
    "sample_rate_hz",
    "input_duration_seconds",
    "input_samples",
    "batch_size",
    "warmup_iterations",
    "timed_iterations",
    "eval_mode",
    "inference_mode",
    "cuda_synchronized",
    "augmentation_during_inference",
]


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the standardized benchmark. "
            "Run on the specified NVIDIA Tesla T4 environment."
        )


def verify_gpu(config: Dict[str, Any]) -> str:
    gpu_name = torch.cuda.get_device_name(0)
    required = config["hardware"]["required_gpu_name"]

    print(f"Detected GPU: {gpu_name}")

    if required not in gpu_name:
        raise RuntimeError(
            f"GPU mismatch. Required '{required}', detected '{gpu_name}'. "
            "Do not mix benchmark results from a different GPU into the "
            "manuscript evidence without rerunning the standardized protocol."
        )

    return gpu_name


def count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def validate_parameter_counts(model_name: str, model: nn.Module) -> Dict[str, Any]:
    total, trainable, frozen = count_parameters(model)

    if total != EXPECTED_TOTAL_PARAMS:
        raise RuntimeError(
            f"{model_name}: unexpected total parameter count: {total:,}. "
            f"Expected {EXPECTED_TOTAL_PARAMS:,}."
        )

    if model_name in ("WavLM-FT", "WavLM-FT+Aug"):
        if trainable != EXPECTED_TOTAL_PARAMS or frozen != 0:
            raise RuntimeError(
                f"{model_name}: expected all {EXPECTED_TOTAL_PARAMS:,} "
                "parameters to be trainable."
            )

    if model_name == "WavLM-PEFT":
        if trainable != EXPECTED_M3_TRAINABLE_PARAMS:
            raise RuntimeError(
                f"WavLM-PEFT: unexpected trainable parameter count: "
                f"{trainable:,}. Expected {EXPECTED_M3_TRAINABLE_PARAMS:,}."
            )
        if frozen != EXPECTED_M3_FROZEN_PARAMS:
            raise RuntimeError(
                f"WavLM-PEFT: unexpected frozen parameter count: "
                f"{frozen:,}. Expected {EXPECTED_M3_FROZEN_PARAMS:,}."
            )

        reduction = 100.0 * (1.0 - trainable / total)
        if not np.isclose(reduction, EXPECTED_M3_TRAINABLE_REDUCTION, atol=0.01):
            raise RuntimeError(
                f"WavLM-PEFT: unexpected trainable-parameter reduction: "
                f"{reduction:.2f}%. Expected approximately "
                f"{EXPECTED_M3_TRAINABLE_REDUCTION:.2f}%."
            )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }


def freeze_first_n_layers(model: nn.Module, n_layers: int) -> None:
    layers = model.wavlm.encoder.layers

    if len(layers) != EXPECTED_TRANSFORMER_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRANSFORMER_LAYERS} WavLM Transformer layers, "
            f"found {len(layers)}."
        )

    if not 0 <= n_layers <= len(layers):
        raise ValueError(f"Invalid freeze depth: {n_layers}")

    for layer in layers[:n_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def verify_model3_freezing(model: nn.Module) -> None:
    layers = model.wavlm.encoder.layers

    if len(layers) != EXPECTED_TRANSFORMER_LAYERS:
        raise RuntimeError("Model 3 does not contain exactly 24 Transformer layers.")

    lower_frozen = all(
        not parameter.requires_grad
        for layer in layers[:12]
        for parameter in layer.parameters()
    )
    upper_trainable = all(
        parameter.requires_grad
        for layer in layers[12:]
        for parameter in layer.parameters()
    )

    if not lower_frozen or not upper_trainable:
        raise RuntimeError(
            "Model 3 freezing configuration does not match the manuscript: "
            "layers 1-12 must be frozen and layers 13-24 must be trainable."
        )


def load_checkpoint_exact(
    model_name: str,
    checkpoint_path: Path,
    config_name: str,
    num_classes: int,
    freeze_n: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Reconstruct and load one trained checkpoint with strict validation."""

    print("\n" + "=" * 76)
    print(f"Loading: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 76)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {model_name}: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(f"{model_name}: checkpoint must be a dictionary.")

    # Training saves the raw ``model.state_dict()`` directly.  Do not wrap it
    # in or expect a nested ``model_state`` dictionary: doing so would make
    # the benchmark incompatible with the actual training artifacts.
    checkpoint_state = checkpoint
    if not all(isinstance(key, str) for key in checkpoint_state.keys()):
        raise TypeError(f"{model_name}: checkpoint keys must be strings.")

    config = WavLMConfig.from_pretrained(
        config_name,
        num_labels=num_classes,
    )
    model = WavLMForSequenceClassification(config)

    model_state = model.state_dict()
    checkpoint_keys = set(checkpoint_state.keys())
    model_keys = set(model_state.keys())

    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"{model_name}: checkpoint/model key mismatch.\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )

    shape_mismatches = []
    for key in sorted(model_keys):
        checkpoint_shape = tuple(checkpoint_state[key].shape)
        model_shape = tuple(model_state[key].shape)
        if checkpoint_shape != model_shape:
            shape_mismatches.append(
                (key, checkpoint_shape, model_shape)
            )

    if shape_mismatches:
        raise RuntimeError(
            f"{model_name}: tensor-shape mismatch detected:\n"
            + "\n".join(map(str, shape_mismatches[:20]))
        )

    # The key/shape checks above make strict loading a deliberate safety gate.
    model.load_state_dict(checkpoint_state, strict=True)

    freeze_first_n_layers(model, freeze_n)

    if model_name == "WavLM-PEFT":
        verify_model3_freezing(model)

    parameter_info = validate_parameter_counts(model_name, model)

    print(f"Checkpoint state-dict keys: {len(checkpoint_keys):,}")
    print("Missing keys after exact validation: []")
    print("Unexpected keys after exact validation: []")
    print("Tensor-shape mismatches: []")
    print("✓ Exact checkpoint state-dict match")
    print(f"Total parameters: {parameter_info['total_parameters']:,}")
    print(f"Trainable parameters: {parameter_info['trainable_parameters']:,}")
    print(f"Frozen parameters: {parameter_info['frozen_parameters']:,}")

    return model, {
        "checkpoint_format": "raw_state_dict",
        **parameter_info,
    }


# ---------------------------------------------------------------------------
# Standardized input
# ---------------------------------------------------------------------------

def create_benchmark_input(config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    sample_rate = int(config["input"]["sampling_rate_hz"])
    duration = float(config["input"]["duration_seconds"])
    batch_size = int(config["input"]["batch_size"])
    waveform_type = str(config["input"]["waveform_type"])

    if batch_size != 1:
        raise ValueError("This benchmark is defined for batch size 1.")
    if sample_rate != 16_000:
        raise ValueError("This benchmark is defined for 16 kHz input.")
    if not np.isclose(duration, 4.0):
        raise ValueError("This benchmark is defined for 4.0-second input.")
    if waveform_type != "zero_tensor":
        raise ValueError(
            "The standardized benchmark currently requires waveform_type='zero_tensor'."
        )

    num_samples = int(round(sample_rate * duration))
    if num_samples != 64_000:
        raise RuntimeError("Expected 64,000 samples for 4 seconds at 16 kHz.")

    return torch.zeros(
        (batch_size, num_samples),
        dtype=torch.float32,
        device=device,
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def synchronize_cuda() -> None:
    torch.cuda.synchronize()


def forward_once(model: nn.Module, model_input: torch.Tensor) -> None:
    _ = model(model_input)


def run_timed_inference(
    model: nn.Module,
    model_input: torch.Tensor,
    warmup_iterations: int,
    timed_iterations: int,
) -> Tuple[List[float], float]:
    """Run the common warm-up and timed protocol."""

    print(f"Running {warmup_iterations} warm-up iterations...")

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            forward_once(model, model_input)

    synchronize_cuda()
    torch.cuda.reset_peak_memory_stats()
    synchronize_cuda()

    print(f"Running {timed_iterations} timed iterations...")

    latencies_ms: List[float] = []

    with torch.inference_mode():
        for _ in range(timed_iterations):
            synchronize_cuda()
            start = time.perf_counter()
            forward_once(model, model_input)
            synchronize_cuda()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(float(elapsed_ms))

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return latencies_ms, float(peak_memory_gb)


def summarize_latencies(
    latencies_ms: List[float],
    input_duration_seconds: float,
) -> Dict[str, float]:
    values = np.asarray(latencies_ms, dtype=np.float64)

    if len(values) == 0:
        raise ValueError("No timing observations were collected.")

    mean_ms = float(np.mean(values))
    median_ms = float(np.median(values))
    p95_ms = float(np.percentile(values, 95))
    throughput = 1000.0 / mean_ms
    rtf = (mean_ms / 1000.0) / input_duration_seconds

    return {
        "mean_latency_ms": mean_ms,
        "median_latency_ms": median_ms,
        "p95_latency_ms": p95_ms,
        "throughput_samples_per_s": float(throughput),
        "rtf": float(rtf),
    }


# ---------------------------------------------------------------------------
# Model 3 forward-path verification
# ---------------------------------------------------------------------------

def verify_model3_forward_path(
    model: nn.Module,
    model_input: torch.Tensor,
    expected_iterations: int,
) -> List[int]:
    """Verify all 24 Transformer layers execute exactly once per forward."""

    layers = model.wavlm.encoder.layers

    if len(layers) != EXPECTED_TRANSFORMER_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRANSFORMER_LAYERS} Transformer layers, "
            f"found {len(layers)}."
        )

    counts = [0] * len(layers)
    handles = []

    for index, layer in enumerate(layers):
        def hook(_module, _inputs, _output, index=index):
            counts[index] += 1

        handles.append(layer.register_forward_hook(hook))

    try:
        with torch.inference_mode():
            for _ in range(expected_iterations):
                forward_once(model, model_input)
        synchronize_cuda()
    finally:
        for handle in handles:
            handle.remove()

    if any(count != expected_iterations for count in counts):
        raise RuntimeError(
            "Model 3 layer execution verification failed. "
            f"Expected {expected_iterations} executions per layer; counts={counts}"
        )

    return counts


def save_layer_verification(
    counts: List[int],
    expected_iterations: int,
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_index", "execution_count", "expected_count", "verified"])
        for index, count in enumerate(counts, start=1):
            writer.writerow([
                index,
                count,
                expected_iterations,
                count == expected_iterations,
            ])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_timings(model_name: str, latencies_ms: List[float]) -> Path:
    filename = {
        "WavLM-FT": "model1_wavlm_ft_timings.csv",
        "WavLM-FT+Aug": "model2_wavlm_ft_aug_timings.csv",
        "WavLM-PEFT": "model3_wavlm_peft_timings.csv",
    }[model_name]

    path = RAW_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "latency_ms"])
        for index, value in enumerate(latencies_ms, start=1):
            writer.writerow([index, f"{value:.8f}"])

    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    require_cuda()
    gpu_name = verify_gpu(CONFIG)

    architecture = CONFIG["architecture"]
    num_classes = int(architecture["num_classes"])
    config_name = str(architecture["pretrained_config"])

    if list(architecture["class_order"]) != EXPECTED_CLASS_ORDER:
        raise RuntimeError("Class order in benchmark_config.yaml does not match the manuscript configuration.")

    if int(architecture["transformer_layers"]) != EXPECTED_TRANSFORMER_LAYERS:
        raise RuntimeError("Expected 24 WavLM-Large Transformer layers.")

    if int(architecture["hidden_size"]) != 1024 or int(architecture["projector_size"]) != 256:
        raise RuntimeError("Architecture dimensions do not match the BNSER model definition.")

    device = torch.device("cuda")
    sample_rate = int(CONFIG["input"]["sampling_rate_hz"])
    duration = float(CONFIG["input"]["duration_seconds"])
    warmup = int(CONFIG["inference"]["warmup_iterations"])
    timed = int(CONFIG["inference"]["timed_iterations"])

    if not CONFIG["inference"]["eval_mode"]:
        raise RuntimeError("The standardized protocol requires eval_mode=true.")
    if not CONFIG["inference"]["inference_mode"]:
        raise RuntimeError("The standardized protocol requires inference_mode=true.")
    if not CONFIG["inference"]["cuda_synchronize"]:
        raise RuntimeError("The standardized protocol requires CUDA synchronization.")
    if warmup != 20 or timed != 300:
        raise RuntimeError("The standardized protocol requires 20 warm-up and 300 timed iterations.")

    model_results = []
    benchmark_records = []
    layer_counts = None

    print("\n" + "=" * 76)
    print("BNSER-WavLM STANDARDIZED INFERENCE-EFFICIENCY BENCHMARK")
    print("=" * 76)
    print(f"GPU: {gpu_name}")
    print(f"Input: batch=1, {sample_rate} Hz, {duration:.1f} s, 64,000 samples")
    print(f"Warm-up: {warmup} | Timed: {timed}")
    print("Augmentation during inference: disabled")
    print("=" * 76)

    for model_name in MODEL_NAMES:
        model_cfg = CONFIG["models"][model_name]
        checkpoint_path = resolve_checkpoint_path(str(model_cfg["checkpoint"]))
        freeze_n = int(model_cfg["freeze_first_n_layers"])

        if model_cfg["augmentation_during_inference"] is not False:
            raise RuntimeError(f"{model_name}: augmentation during inference must be disabled.")

        model, checkpoint_info = load_checkpoint_exact(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            config_name=config_name,
            num_classes=num_classes,
            freeze_n=freeze_n,
        )

        model = model.to(device)
        model.eval()
        model_input = create_benchmark_input(CONFIG, device)

        # Sanity-check one forward pass before timing.
        with torch.inference_mode():
            output = model(model_input)
        synchronize_cuda()

        if tuple(output.shape) != (1, num_classes):
            raise RuntimeError(
                f"{model_name}: unexpected output shape {tuple(output.shape)}; "
                f"expected (1, {num_classes})."
            )
        if not torch.isfinite(output).all().item():
            raise RuntimeError(f"{model_name}: non-finite output detected during sanity check.")

        # Hooks are deliberately NOT active during timed inference. This keeps
        # the latency comparison fair across all three configurations.
        latencies_ms, peak_memory_gb = run_timed_inference(
            model=model,
            model_input=model_input,
            warmup_iterations=warmup,
            timed_iterations=timed,
        )

        summary = summarize_latencies(latencies_ms, duration)
        timing_path = save_timings(model_name, latencies_ms)

        parameter_info = validate_parameter_counts(model_name, model)

        result_row = {
            "model": model_name,
            **summary,
            "peak_gpu_memory_allocated_gb": peak_memory_gb,
            **parameter_info,
            "gpu": gpu_name,
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "sample_rate_hz": sample_rate,
            "input_duration_seconds": duration,
            "input_samples": int(sample_rate * duration),
            "batch_size": 1,
            "warmup_iterations": warmup,
            "timed_iterations": timed,
            "eval_mode": True,
            "inference_mode": True,
            "cuda_synchronized": True,
            "augmentation_during_inference": False,
        }
        model_results.append(result_row)
        benchmark_records.append({
            "model": model_name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": checkpoint_info["checkpoint_format"],
            "timing_file": str(timing_path.relative_to(ROOT)),
        })

        print("\n" + "-" * 76)
        print(f"RESULT: {model_name}")
        print("-" * 76)
        print(f"Mean latency       : {summary['mean_latency_ms']:.4f} ms")
        print(f"Median latency     : {summary['median_latency_ms']:.4f} ms")
        print(f"P95 latency        : {summary['p95_latency_ms']:.4f} ms")
        print(f"Throughput         : {summary['throughput_samples_per_s']:.4f} samples/s")
        print(f"RTF                : {summary['rtf']:.6f}")
        print(f"Peak GPU memory    : {peak_memory_gb:.4f} GB")
        print(f"Total parameters   : {parameter_info['total_parameters']:,}")
        print(f"Trainable params   : {parameter_info['trainable_parameters']:,}")
        print(f"Frozen parameters  : {parameter_info['frozen_parameters']:,}")

        # Model 3 forward-path verification is deliberately separate from the
        # timed benchmark so that hooks do not alter the measured latency.
        if model_name == "WavLM-PEFT":
            print("\nModel 3 layer execution verification (separate from timing):")
            layer_counts = verify_model3_forward_path(
                model=model,
                model_input=model_input,
                expected_iterations=timed,
            )

            layer_path = RESULTS_DIR / CONFIG["outputs"]["layer_verification_file"]
            save_layer_verification(layer_counts, timed, layer_path)

            print(f"Expected layers : {EXPECTED_TRANSFORMER_LAYERS}")
            print(f"Executed layers : {sum(count > 0 for count in layer_counts)}")
            for index, count in enumerate(layer_counts, start=1):
                print(f"Layer {index:02d}: {count} execution(s)")
            print("✓ All 24 WavLM-Large Transformer layers executed.")

        del model
        del model_input
        gc.collect()
        torch.cuda.empty_cache()
        synchronize_cuda()

    # Summary CSV.  The field order is fixed to the canonical manuscript-facing
    # schema used by the committed evidence artifact.
    summary_path = RESULTS_DIR / CONFIG["outputs"]["summary_file"]
    if any(set(row) != set(CANONICAL_RESULT_COLUMNS) for row in model_results):
        raise RuntimeError(
            "Benchmark result schema drift detected. "
            f"Expected columns: {CANONICAL_RESULT_COLUMNS}"
        )
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(model_results)

    metadata = {
        "benchmark_name": "BNSER-WavLM standardized inference-efficiency benchmark",
        "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        "benchmark_config_sha256": sha256_file(CONFIG_PATH),
        "hardware": gpu_name,
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_version": torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
        "sampling_rate_hz": sample_rate,
        "input_duration_seconds": duration,
        "input_samples": int(sample_rate * duration),
        "batch_size": 1,
        "waveform_type": CONFIG["input"]["waveform_type"],
        "warmup_iterations": warmup,
        "timed_iterations": timed,
        "eval_mode": True,
        "torch_inference_mode": True,
        "cuda_synchronized": True,
        "augmentation_during_inference": False,
        "canonical_result_columns": CANONICAL_RESULT_COLUMNS,
        "benchmark_records": benchmark_records,
        "m3_all_24_layers_verified": layer_counts is not None,
        "m3_expected_execution_count_per_layer": timed,
        "class_order": EXPECTED_CLASS_ORDER,
        "architecture": {
            "backbone": config_name,
            "transformer_layers": EXPECTED_TRANSFORMER_LAYERS,
            "hidden_size": 1024,
            "projector_size": 256,
            "num_classes": num_classes,
        },
    }

    metadata_path = RAW_DIR / CONFIG["outputs"]["metadata_file"]
    write_json(metadata_path, metadata)

    print("\n" + "=" * 76)
    print("BENCHMARK COMPLETED")
    print(f"Summary: {summary_path}")
    print(f"Raw timings: {RAW_DIR}")
    if layer_counts is not None:
        print(f"Layer verification: {RESULTS_DIR / CONFIG['outputs']['layer_verification_file']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
