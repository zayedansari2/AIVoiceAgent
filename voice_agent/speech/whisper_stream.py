"""
Streaming speech recognition using *faster-whisper*.

faster-whisper is a reimplementation of OpenAI Whisper in CTranslate2,
offering significantly faster CPU inference and lower memory usage.

The :class:`WhisperStream` class accepts a speech segment (numpy array)
and returns the transcribed text together with the detected language.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from voice_agent.config import Config

# faster-whisper is imported lazily so the module can be imported even
# when the library is not installed (useful for unit-testing utilities).
try:
    from faster_whisper import WhisperModel as _WhisperModel  # type: ignore
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTER_WHISPER_AVAILABLE = False


class WhisperStream:
    """CPU-optimised speech recogniser using *faster-whisper*.

    The model is loaded once at construction time and reused for every
    transcription call, avoiding the overhead of re-loading.

    Parameters
    ----------
    config:
        Shared configuration object.

    Raises
    ------
    ImportError
        If ``faster-whisper`` is not installed.
    """

    def __init__(self, config: Config) -> None:
        if not _FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is not installed.  "
                "Run: pip install faster-whisper"
            )
        self._config = config
        self._model: Optional[_WhisperModel] = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Instantiate the Whisper model (downloads on first run)."""
        if self._config.debug:
            print(
                f"[WhisperStream] Loading model '{self._config.whisper_model_size}' "
                f"({self._config.whisper_compute_type}) on {self._config.whisper_device} …"
            )
        self._model = _WhisperModel(
            self._config.whisper_model_size,
            device=self._config.whisper_device,
            compute_type=self._config.whisper_compute_type,
        )
        if self._config.debug:
            print("[WhisperStream] Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Transcribe a speech segment.

        Parameters
        ----------
        audio:
            1-D float32 numpy array sampled at ``config.sample_rate``.
        language:
            Optional ISO-639-1 hint (e.g. ``"es"``).  When *None*,
            Whisper detects the language automatically.

        Returns
        -------
        (text, detected_language)
            ``text`` is the concatenated transcription.
            ``detected_language`` is the two-letter ISO-639-1 code
            (e.g. ``"es"``, ``"ar"``).
        """
        assert self._model is not None, "Model not loaded"

        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self._config.whisper_beam_size,
            vad_filter=False,  # We run our own VAD upstream.
        )

        text_parts = [seg.text.strip() for seg in segments]
        text = " ".join(text_parts).strip()
        detected_lang: str = info.language  # type: ignore[attr-defined]

        if self._config.debug:
            print(
                f"[WhisperStream] Detected language: {detected_lang} "
                f"(prob={info.language_probability:.2f})"
            )
            print(f"[WhisperStream] Transcript: {text!r}")

        return text, detected_lang
