"""
Entry point for the AI Voice Agent.

Run directly::

    python main.py [--debug] [--no-llm] [--no-tts]
              [--whisper-model {tiny,base,small}]
              [--llm-model PATH]

Or as a module::

    python -m voice_agent.main
"""

from __future__ import annotations

import argparse

from voice_agent.config import Config
from voice_agent.main import VoiceAgent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time multilingual AI voice agent"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print verbose debug information.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium"],
        help="faster-whisper model size (default: base).",
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable the optional LLM reasoning layer.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        metavar="PATH",
        help="Path to a GGUF model file for LLM reasoning.",
    )
    parser.add_argument(
        "--tts-engine",
        default="piper",
        choices=["piper", "pyttsx3"],
        help="TTS engine to use (default: piper).",
    )
    parser.add_argument(
        "--piper-voice",
        default="en_US-lessac-medium",
        help="Piper voice name (default: en_US-lessac-medium).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = Config(
        debug=args.debug,
        whisper_model_size=args.whisper_model,
        enable_llm=args.enable_llm,
        llm_model_path=args.llm_model,
        tts_engine=args.tts_engine,
        piper_voice=args.piper_voice,
    )

    agent = VoiceAgent(config)
    agent.run()


if __name__ == "__main__":
    main()
