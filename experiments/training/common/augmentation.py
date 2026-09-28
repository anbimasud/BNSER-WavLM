"""Controlled training-time augmentation used by Models 2 and 3."""

from __future__ import annotations

import random

import librosa
import numpy as np


class ControlledAugmentation:
    """Apply the manuscript-defined probabilistic augmentation scheme."""

    def __init__(
        self,
        probability: float = 0.35,
        noise_probability: float = 0.40,
        pitch_probability: float = 0.30,
        time_stretch_probability: float = 0.30,
        noise_level_min: float = 0.005,
        noise_level_max: float = 0.020,
        pitch_steps_min: float = -0.5,
        pitch_steps_max: float = 0.5,
        stretch_rate_min: float = 0.95,
        stretch_rate_max: float = 1.05,
        sample_rate: int = 16000,
    ) -> None:
        self.probability = probability
        self.noise_probability = noise_probability
        self.pitch_probability = pitch_probability
        self.time_stretch_probability = time_stretch_probability
        self.noise_level_min = noise_level_min
        self.noise_level_max = noise_level_max
        self.pitch_steps_min = pitch_steps_min
        self.pitch_steps_max = pitch_steps_max
        self.stretch_rate_min = stretch_rate_min
        self.stretch_rate_max = stretch_rate_max
        self.sample_rate = sample_rate

        total_probability = (
            noise_probability + pitch_probability + time_stretch_probability
        )
        if not np.isclose(total_probability, 1.0):
            raise ValueError(
                "Augmentation type probabilities must sum to 1.0; "
                f"received {total_probability:.4f}."
            )

    def __call__(self, speech: np.ndarray) -> np.ndarray:
        if random.random() >= self.probability:
            return speech

        choice = random.random()

        if choice < self.noise_probability:
            # The original Model-3 implementation scales Gaussian noise by
            # the peak amplitude of the current waveform.
            current_level = np.max(np.abs(speech))
            noise_level = random.uniform(
                self.noise_level_min,
                self.noise_level_max,
            ) * current_level
            noise = np.random.randn(len(speech)) * noise_level
            speech = speech + noise

        elif choice < self.noise_probability + self.pitch_probability:
            n_steps = random.uniform(
                self.pitch_steps_min,
                self.pitch_steps_max,
            )
            speech = librosa.effects.pitch_shift(
                speech,
                sr=self.sample_rate,
                n_steps=n_steps,
            )

        else:
            rate = random.uniform(
                self.stretch_rate_min,
                self.stretch_rate_max,
            )
            speech = librosa.effects.time_stretch(
                speech,
                rate=rate,
            )

        return np.asarray(speech, dtype=np.float32)
