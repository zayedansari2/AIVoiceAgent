"""
VoiceAgent – main pipeline controller.

Orchestrates the full audio → VAD → speech recognition →
language detection → translation → (optional LLM) → TTS pipeline.

Usage (CLI)::

    python -m voice_agent.main

Or from Python::

    from voice_agent.main import VoiceAgent
    from voice_agent.config import Config

    agent = VoiceAgent(Config())
    agent.run()           # blocks until KeyboardInterrupt
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

from voice_agent.audio.recorder import AudioRecorder
from voice_agent.audio.vad import SileroVAD
from voice_agent.config import Config
from voice_agent.reasoning.llm_processor import LLMProcessor
from voice_agent.speech.whisper_stream import WhisperStream
from voice_agent.translation.translator import Translator
from voice_agent.tts.speaker import Speaker
from voice_agent.utils.language_detection import get_language_name, is_supported


class VoiceAgent:
    """Real-time multilingual voice agent.

    The agent listens to the microphone, detects speech, transcribes it,
    translates it to English and optionally post-processes it with a
    local LLM before speaking the result aloud.

    Parameters
    ----------
    config:
        Shared configuration object.  Defaults to a new :class:`Config`
        with all default values.

    Example
    -------
    >>> agent = VoiceAgent()
    >>> agent.run()   # press Ctrl-C to stop
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()
        self._running = False

        print("[VoiceAgent] Initialising components …")
        self._recorder = AudioRecorder(self._config)
        self._vad = SileroVAD(self._config)
        self._whisper = WhisperStream(self._config)
        self._translator = Translator(self._config)
        self._llm = LLMProcessor(self._config)
        self._speaker = Speaker(self._config)
        print("[VoiceAgent] All components ready.  Speak now …\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the real-time voice agent loop.

        Blocks until a ``KeyboardInterrupt`` or :meth:`stop` is called.
        """
        self._running = True

        # Allow graceful shutdown via Ctrl-C.
        def _signal_handler(sig: int, frame: object) -> None:  # noqa: ARG001
            print("\n[VoiceAgent] Shutting down …")
            self.stop()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        with self._recorder:
            try:
                self._loop()
            except Exception as exc:  # pragma: no cover
                print(f"[VoiceAgent] Fatal error: {exc}")
                raise
            finally:
                self._running = False

    def stop(self) -> None:
        """Request the agent to stop processing."""
        self._running = False

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main processing loop: read audio chunks → VAD → transcribe."""
        while self._running:
            try:
                chunk = self._recorder.read(timeout=0.1)
            except Exception:  # queue.Empty or timeout
                continue

            self._vad.process_chunk(chunk)

            for segment in self._vad.iter_segments():
                self._process_segment(segment)

    def _process_segment(self, audio: np.ndarray) -> None:
        """Run the full pipeline on a single speech segment.

        Parameters
        ----------
        audio:
            1-D float32 PCM array at ``config.sample_rate``.
        """
        # 1. Speech recognition + language detection.
        text, lang_code = self._whisper.transcribe(audio)

        if not text:
            return

        lang_name = get_language_name(lang_code)
        print(f"[{lang_name}] {text}")

        # 2. Translation → English.
        if lang_code == "en":
            english_text = text
        elif is_supported(lang_code):
            english_text = self._translator.translate(text, lang_code)
            print(f"[English] {english_text}")
        else:
            print(
                f"[VoiceAgent] Language '{lang_code}' is not in the "
                "supported list – skipping translation."
            )
            english_text = text

        # 3. Optional LLM post-processing.
        final_text = self._llm.process(english_text)
        if self._config.enable_llm and final_text != english_text:
            print(f"[LLM refined] {final_text}")

        # 4. Text-to-speech.
        self._speaker.speak(final_text)
