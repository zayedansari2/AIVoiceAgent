"""
Voice Activity Detection (VAD) using Silero VAD.

Silero VAD is a lightweight (~1 MB) PyTorch model that classifies each
30 ms audio chunk as *speech* or *non-speech*.  This module wraps the
model and provides a simple stateful interface that accumulates chunks
and yields complete speech segments.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, List, Optional

import numpy as np
import torch

from voice_agent.config import Config


class SileroVAD:
    """Voice Activity Detector backed by Silero VAD.

    The detector is *stateful*: call :meth:`process_chunk` for every
    incoming audio chunk and it will emit complete speech segments via
    :meth:`get_speech_segment` (or iterate with :meth:`iter_segments`).

    Parameters
    ----------
    config:
        Shared configuration object.
    """

    # Silero VAD repository on torch.hub
    _REPO = "snakers4/silero-vad"
    _MODEL_NAME = "silero_vad"

    def __init__(self, config: Config) -> None:
        self._config = config
        self._chunk_samples = int(config.sample_rate * config.vad_chunk_ms / 1000)
        self._min_speech_chunks = max(
            1,
            config.vad_min_speech_ms // config.vad_chunk_ms,
        )
        self._min_silence_chunks = max(
            1,
            config.vad_min_silence_ms // config.vad_chunk_ms,
        )

        # Internal state
        self._speech_buffer: List[np.ndarray] = []
        self._silence_counter: int = 0
        self._speech_counter: int = 0
        self._in_speech: bool = False

        # Queue of completed speech segments (numpy arrays)
        self._segments: Deque[np.ndarray] = deque()

        self._model: Optional[torch.nn.Module] = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Download / load the Silero VAD model from torch.hub."""
        if self._config.debug:
            print("[SileroVAD] Loading Silero VAD model …")
        # Trust is required for torch.hub models.
        self._model, _ = torch.hub.load(
            self._REPO,
            self._MODEL_NAME,
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        self._model.eval()
        if self._config.debug:
            print("[SileroVAD] Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> None:
        """Feed a single audio chunk to the VAD.

        Parameters
        ----------
        chunk:
            1-D float32 numpy array of length ``config.vad_chunk_ms`` ms.
        """
        # Ensure correct length (pad or truncate silently).
        if len(chunk) < self._chunk_samples:
            chunk = np.pad(chunk, (0, self._chunk_samples - len(chunk)))
        elif len(chunk) > self._chunk_samples:
            chunk = chunk[: self._chunk_samples]

        tensor = torch.from_numpy(chunk).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            prob: float = self._model(tensor, self._config.sample_rate).item()

        is_speech = prob >= self._config.vad_threshold

        if is_speech:
            self._speech_buffer.append(chunk)
            self._speech_counter += 1
            self._silence_counter = 0
            if not self._in_speech and self._speech_counter >= self._min_speech_chunks:
                self._in_speech = True
        else:
            if self._in_speech:
                self._silence_counter += 1
                # Keep a short look-ahead buffer so we don't clip the end of words.
                self._speech_buffer.append(chunk)
                if self._silence_counter >= self._min_silence_chunks:
                    self._flush_segment()
            else:
                # Not in speech – discard leading silence.
                self._speech_counter = 0
                self._speech_buffer.clear()

    def has_segment(self) -> bool:
        """Return True if at least one complete segment is ready."""
        return bool(self._segments)

    def get_segment(self) -> Optional[np.ndarray]:
        """Return the oldest completed speech segment, or None."""
        try:
            return self._segments.popleft()
        except IndexError:
            return None

    def iter_segments(self) -> Iterator[np.ndarray]:
        """Yield all currently available speech segments."""
        while self._segments:
            yield self._segments.popleft()

    def reset(self) -> None:
        """Clear all internal state."""
        self._speech_buffer.clear()
        self._segments.clear()
        self._silence_counter = 0
        self._speech_counter = 0
        self._in_speech = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_segment(self) -> None:
        """Move the accumulated speech buffer to the completed-segments queue."""
        if self._speech_buffer:
            segment = np.concatenate(self._speech_buffer)
            self._segments.append(segment)
        self._speech_buffer = []
        self._silence_counter = 0
        self._speech_counter = 0
        self._in_speech = False
