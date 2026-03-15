"""
Configuration module for the AI Voice Agent.

All tuneable parameters are centralised here so that the rest of the
codebase only needs to import a single :class:`Config` object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Central configuration for every component of the voice agent.

    All values are chosen to keep memory usage under ~3 GB on a CPU-only
    MacBook Air while maintaining acceptable latency.
    """

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------

    #: Sampling rate expected by Whisper and Silero VAD (must be 16 000 Hz).
    sample_rate: int = 16_000

    #: Number of audio channels to record (mono).
    channels: int = 1

    #: Duration of a single audio chunk fed to VAD, in milliseconds.
    #: Silero VAD supports 30 ms or 60 ms at 16 kHz.
    vad_chunk_ms: int = 30

    #: Maximum seconds to accumulate before forcing a transcription even
    #: when speech is still detected (prevents unbounded buffer growth).
    max_speech_duration_s: float = 30.0

    # ------------------------------------------------------------------
    # Voice Activity Detection (Silero VAD)
    # ------------------------------------------------------------------

    #: Probability threshold above which a chunk is classified as speech.
    vad_threshold: float = 0.5

    #: Minimum contiguous speech duration (ms) before a segment is accepted.
    vad_min_speech_ms: int = 250

    #: Silence duration (ms) after which a speech segment is considered ended.
    vad_min_silence_ms: int = 500

    # ------------------------------------------------------------------
    # Speech recognition (faster-whisper)
    # ------------------------------------------------------------------

    #: Whisper model size. "base" balances speed vs. accuracy on CPU.
    #: Options: tiny | base | small | medium
    whisper_model_size: str = "base"

    #: Quantisation type. "int8" reduces memory and speeds up CPU inference.
    whisper_compute_type: str = "int8"

    #: Device for Whisper inference.
    whisper_device: str = "cpu"

    #: Beam size for Whisper decoding (lower = faster, slightly less accurate).
    whisper_beam_size: int = 1

    # ------------------------------------------------------------------
    # Translation (NLLB-200)
    # ------------------------------------------------------------------

    #: HuggingFace model ID for the translation pipeline.
    translation_model: str = "facebook/nllb-200-distilled-600M"

    #: NLLB BCP-47 code for the target language.
    translation_target_lang: str = "eng_Latn"

    #: Maximum token length for the translation model's output.
    translation_max_length: int = 512

    #: Supported source language codes as returned by Whisper.
    supported_source_languages: List[str] = field(
        default_factory=lambda: ["es", "hi", "ur", "ar"]
    )

    # ------------------------------------------------------------------
    # Optional LLM reasoning layer (llama-cpp-python)
    # ------------------------------------------------------------------

    #: Set to True to enable the LLM post-processing step.
    enable_llm: bool = False

    #: Absolute path to a GGUF model file (e.g. Llama-3.2-1B-Q4_K_M.gguf).
    llm_model_path: Optional[str] = None

    #: Number of tokens to generate in the LLM response.
    llm_max_tokens: int = 256

    #: Context size for the LLM (smaller = less RAM).
    llm_n_ctx: int = 2048

    # ------------------------------------------------------------------
    # Text-to-Speech
    # ------------------------------------------------------------------

    #: TTS engine to use.  "piper" (recommended) or "pyttsx3" (fallback).
    tts_engine: str = "piper"

    #: Piper voice name (must match a downloaded .onnx + .onnx.json pair).
    #: See https://github.com/rhasspy/piper/releases for available voices.
    piper_voice: str = "en_US-lessac-medium"

    #: Directory where Piper voice files are stored.
    piper_voices_dir: str = os.path.join(os.path.expanduser("~"), ".local", "share", "piper-voices")

    #: Speech rate for pyttsx3 fallback (words per minute).
    pyttsx3_rate: int = 160

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------

    #: If True, print debug information to stdout.
    debug: bool = False
