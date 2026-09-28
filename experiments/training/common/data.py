"""Dataset loading and fixed-length preprocessing for BNSER-WavLM."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import librosa
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor

from .utils import EMOTIONS


SUPPORTED_EXTENSION = ".wav"


def collect_split(
    data_root: str | Path,
    split_name: str,
) -> list[dict[str, str]]:
    """Load one predefined speaker-independent dataset partition."""
    root = Path(data_root) / split_name

    if not root.is_dir():
        raise FileNotFoundError(
            f"Expected split directory not found: {root}"
        )

    records: list[dict[str, str]] = []

    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue

        for emotion in EMOTIONS:
            emotion_dir = speaker_dir / emotion

            if not emotion_dir.is_dir():
                continue

            for audio_path in sorted(emotion_dir.iterdir()):
                if (
                    audio_path.is_file()
                    and audio_path.suffix.lower() == SUPPORTED_EXTENSION
                ):
                    records.append(
                        {
                            "path": str(audio_path),
                            "speaker": speaker_dir.name,
                            "emotion": emotion,
                        }
                    )

    if not records:
        raise RuntimeError(
            f"No WAV files were found under {root}"
        )

    return records


def load_predefined_splits(
    data_root: str | Path,
    split_names: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    """Load the predefined Train/Val/Test speaker-independent splits."""
    return {
        "train": collect_split(
            data_root,
            split_names["train"],
        ),
        "validation": collect_split(
            data_root,
            split_names["validation"],
        ),
        "test": collect_split(
            data_root,
            split_names["test"],
        ),
    }


def create_feature_extractor(
    model_name: str,
    max_length: int,
) -> Wav2Vec2FeatureExtractor:
    """Create the WavLM feature extractor used in the study."""
    return Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        max_length=max_length,
        truncation=True,
        do_normalize=True,
    )


class BNSERDataset(Dataset):
    """BNSER preprocessing with model-specific original crop behavior."""

    def __init__(
        self,
        records: list[dict[str, str]],
        feature_extractor: Wav2Vec2FeatureExtractor,
        sample_rate: int,
        target_length: int,
        train: bool = False,
        augmentation=None,
        train_crop_mode: str = "random",
        eval_crop_mode: str = "center",
    ) -> None:
        self.records = records
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.train = train
        self.augmentation = augmentation
        self.train_crop_mode = train_crop_mode
        self.eval_crop_mode = eval_crop_mode

        valid_modes = {"pad_or_truncate", "random", "center"}

        if train_crop_mode not in valid_modes:
            raise ValueError(
                "train_crop_mode must be one of "
                f"{sorted(valid_modes)}; received {train_crop_mode!r}."
            )

        if eval_crop_mode not in valid_modes:
            raise ValueError(
                "eval_crop_mode must be one of "
                f"{sorted(valid_modes)}; received {eval_crop_mode!r}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def _fix_length(
        self,
        speech,
        crop_mode: str,
    ):
        """Apply the crop/pad rule corresponding to the original run."""

        if len(speech) <= self.target_length:
            return librosa.util.fix_length(
                speech,
                size=self.target_length,
            )

        if crop_mode == "pad_or_truncate":
            # Exact behavior of the original Model-1 notebook:
            # librosa.util.fix_length(..., size=64000)
            # truncates the waveform from the end when it is too long.
            return librosa.util.fix_length(
                speech,
                size=self.target_length,
            )

        if crop_mode == "random":
            start = random.randint(
                0,
                len(speech) - self.target_length,
            )
            return speech[
                start : start + self.target_length
            ]

        if crop_mode == "center":
            start = (
                len(speech) - self.target_length
            ) // 2
            return speech[
                start : start + self.target_length
            ]

        raise RuntimeError(
            f"Unsupported crop mode: {crop_mode}"
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        speech, sampling_rate = librosa.load(
            record["path"],
            sr=self.sample_rate,
            mono=True,
        )

        # Original Model-2/Model-3 order:
        # augmentation is applied before the four-second crop.
        if self.train and self.augmentation is not None:
            speech = self.augmentation(speech)

        crop_mode = (
            self.train_crop_mode
            if self.train
            else self.eval_crop_mode
        )

        speech = self._fix_length(
            speech,
            crop_mode,
        )

        inputs = self.feature_extractor(
            speech,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        label = EMOTIONS.index(record["emotion"])

        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "sample_id": Path(record["path"]).stem,
            "speaker": record["speaker"],
            "emotion": record["emotion"],
            "path": record["path"],
        }


def create_dataloaders(
    splits: dict[str, list[dict[str, str]]],
    feature_extractor: Wav2Vec2FeatureExtractor,
    sample_rate: int,
    target_length: int,
    batch_size: int,
    augmentation=None,
    train_crop_mode: str = "random",
    eval_crop_mode: str = "center",
) -> dict[str, DataLoader]:
    """Create loaders using the exact configured preprocessing protocol."""

    train_dataset = BNSERDataset(
        splits["train"],
        feature_extractor,
        sample_rate,
        target_length,
        train=True,
        augmentation=augmentation,
        train_crop_mode=train_crop_mode,
        eval_crop_mode=eval_crop_mode,
    )

    validation_dataset = BNSERDataset(
        splits["validation"],
        feature_extractor,
        sample_rate,
        target_length,
        train=False,
        augmentation=None,
        train_crop_mode=train_crop_mode,
        eval_crop_mode=eval_crop_mode,
    )

    test_dataset = BNSERDataset(
        splits["test"],
        feature_extractor,
        sample_rate,
        target_length,
        train=False,
        augmentation=None,
        train_crop_mode=train_crop_mode,
        eval_crop_mode=eval_crop_mode,
    )

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        ),
        "validation": DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
    }


def write_split_manifests(
    splits: dict[str, list[dict[str, str]]],
    data_root: str | Path,
    output_dir: str | Path,
) -> None:
    """Save relative-path manifests without exposing private paths."""
    import pandas as pd

    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, records in splits.items():
        rows = []

        for record in records:
            relative_path = (
                Path(record["path"])
                .resolve()
                .relative_to(data_root)
            )

            rows.append(
                {
                    "path": str(relative_path),
                    "speaker": record["speaker"],
                    "emotion": record["emotion"],
                }
            )

        pd.DataFrame(rows).to_csv(
            output_dir / f"{split_name}_manifest.csv",
            index=False,
        )