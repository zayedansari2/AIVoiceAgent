"""
Text-to-Speech output.

Two engines are supported:

1. **Piper TTS** (recommended) – a fast, local neural TTS engine that
   runs entirely on CPU.  Models are downloaded from the Piper releases
   page.  Install with ``pip install piper-tts``.

2. **pyttsx3** (fallback) – uses the OS text-to-speech engine
   (macOS: say, Linux: espeak / festival, Windows: SAPI).
   No additional model downloads needed.  Install with
   ``pip install pyttsx3``.

The :class:`Speaker` class automatically falls back to *pyttsx3* when
*piper-tts* is not installed.
"""

from __future__ import annotations

import io
import os
import wave
from typing import Optional

import numpy as np
import sounddevice as sd

from voice_agent.config import Config

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from piper import PiperVoice  # type: ignore
    _PIPER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIPER_AVAILABLE = False

try:
    import pyttsx3  # type: ignore
    _PYTTSX3_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYTTSX3_AVAILABLE = False


class Speaker:
    """Convert text to speech and play it through the default audio output.

    Parameters
    ----------
    config:
        Shared configuration object.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._engine: Optional[str] = None
        self._piper_voice: Optional[object] = None
        self._pyttsx3_engine: Optional[object] = None
        self._init_engine()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_engine(self) -> None:
        """Choose and initialise the best available TTS engine."""
        preferred = self._config.tts_engine.lower()

        if preferred == "piper" and _PIPER_AVAILABLE:
            self._init_piper()
            if self._piper_voice is not None:
                self._engine = "piper"
                return

        if _PYTTSX3_AVAILABLE:
            self._init_pyttsx3()
            self._engine = "pyttsx3"
            return

        print(
            "[Speaker] WARNING: No TTS engine available.  "
            "Install piper-tts or pyttsx3.\n"
            "          Speech output will be disabled."
        )

    def _init_piper(self) -> None:
        """Load a Piper voice model."""
        voice_name = self._config.piper_voice
        voices_dir = self._config.piper_voices_dir

        # Look for <voice>.onnx in the voices directory.
        onnx_path = os.path.join(voices_dir, f"{voice_name}.onnx")
        if not os.path.isfile(onnx_path):
            print(
                f"[Speaker] Piper voice model not found: {onnx_path}\n"
                f"          Download it from https://github.com/rhasspy/piper/releases\n"
                f"          and place the .onnx + .onnx.json files in {voices_dir}\n"
                "          Falling back to pyttsx3 …"
            )
            return

        if self._config.debug:
            print(f"[Speaker] Loading Piper voice: {onnx_path}")
        self._piper_voice = PiperVoice.load(onnx_path)  # type: ignore[attr-defined]

    def _init_pyttsx3(self) -> None:
        """Initialise the pyttsx3 engine."""
        if self._config.debug:
            print("[Speaker] Initialising pyttsx3 engine …")
        self._pyttsx3_engine = pyttsx3.init()  # type: ignore[attr-defined]
        self._pyttsx3_engine.setProperty(  # type: ignore[union-attr]
            "rate", self._config.pyttsx3_rate
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Convert *text* to speech and play it.

        Parameters
        ----------
        text:
            English text to synthesise.
        """
        if not text.strip():
            return

        if self._config.debug:
            print(f"[Speaker] Speaking ({self._engine}): {text!r}")

        if self._engine == "piper":
            self._speak_piper(text)
        elif self._engine == "pyttsx3":
            self._speak_pyttsx3(text)
        else:
            # No engine – print a notice only once.
            pass

    # ------------------------------------------------------------------
    # Engine-specific helpers
    # ------------------------------------------------------------------

    def _speak_piper(self, text: str) -> None:
        """Synthesise and play audio via Piper TTS."""
        assert self._piper_voice is not None

        # Synthesise to an in-memory WAV buffer.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._piper_voice.synthesize(text, wf)  # type: ignore[union-attr]

        # Read PCM samples and play with sounddevice.
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        audio = (
            np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        )
        sd.play(audio, samplerate=sample_rate, blocking=True)

    def _speak_pyttsx3(self, text: str) -> None:
        """Synthesise and play audio via pyttsx3."""
        assert self._pyttsx3_engine is not None
        self._pyttsx3_engine.say(text)  # type: ignore[union-attr]
        self._pyttsx3_engine.runAndWait()  # type: ignore[union-attr]
