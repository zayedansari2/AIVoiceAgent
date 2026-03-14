"""
Neural machine translation using *facebook/nllb-200-distilled-600M*.

NLLB-200 (No Language Left Behind) supports 200 languages.  The 600 M
distilled variant uses roughly 1.2 GB of RAM and runs acceptably fast
on a modern CPU.

Supported source languages for this project:
    * Spanish  (spa_Latn)
    * Hindi    (hin_Deva)
    * Urdu     (urd_Arab)
    * Arabic   (arb_Arab)
"""

from __future__ import annotations

from typing import Optional

from voice_agent.config import Config
from voice_agent.utils.language_detection import (
    ENGLISH_NLLB,
    needs_translation,
    whisper_to_nllb,
)

try:
    from transformers import pipeline as hf_pipeline  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False


class Translator:
    """Translate text to English using NLLB-200.

    The HuggingFace translation pipeline is loaded once at construction
    time.  The model is cached in ``~/.cache/huggingface/`` after the
    first download.

    Parameters
    ----------
    config:
        Shared configuration object.

    Raises
    ------
    ImportError
        If ``transformers`` or ``sentencepiece`` is not installed.
    """

    def __init__(self, config: Config) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not installed.  "
                "Run: pip install transformers sentencepiece"
            )
        self._config = config
        self._pipeline = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Instantiate the HuggingFace translation pipeline."""
        if self._config.debug:
            print(
                f"[Translator] Loading '{self._config.translation_model}' …"
            )
        self._pipeline = hf_pipeline(
            "translation",
            model=self._config.translation_model,
            device=-1,  # CPU
            torch_dtype="auto",
        )
        if self._config.debug:
            print("[Translator] Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str, source_lang: str) -> str:
        """Translate *text* from *source_lang* to English.

        Parameters
        ----------
        text:
            Input text in the source language.
        source_lang:
            Whisper ISO-639-1 language code (e.g. ``"es"``).

        Returns
        -------
        str
            Translated English text.  Returns *text* unchanged when the
            source language is already English or translation is not
            possible.
        """
        if not needs_translation(source_lang):
            return text

        nllb_src = whisper_to_nllb(source_lang)
        if nllb_src is None:
            if self._config.debug:
                print(
                    f"[Translator] No NLLB code for Whisper lang '{source_lang}'. "
                    "Returning original text."
                )
            return text

        if self._config.debug:
            print(f"[Translator] Translating '{source_lang}' → English: {text!r}")

        result = self._pipeline(
            text,
            src_lang=nllb_src,
            tgt_lang=ENGLISH_NLLB,
            max_length=self._config.translation_max_length,
        )
        translated: str = result[0]["translation_text"]  # type: ignore[index]

        if self._config.debug:
            print(f"[Translator] Result: {translated!r}")

        return translated

    def is_supported_language(self, whisper_lang: str) -> bool:
        """Return True if *whisper_lang* is in the supported-languages list."""
        return whisper_lang.lower() in self._config.supported_source_languages
