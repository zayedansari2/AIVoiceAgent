"""
Language code utilities.

Maps the short ISO-639-1 language codes returned by Whisper to the
BCP-47-like codes required by the NLLB-200 translation model, and
provides human-readable names for display.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Whisper lang code → (NLLB BCP-47 code, human-readable name)
# ---------------------------------------------------------------------------

_LANGUAGE_MAP: Dict[str, Tuple[str, str]] = {
    # Spanish
    "es": ("spa_Latn", "Spanish"),
    # Hindi
    "hi": ("hin_Deva", "Hindi"),
    # Urdu  (Whisper may detect Urdu as "ur")
    "ur": ("urd_Arab", "Urdu"),
    # Arabic
    "ar": ("arb_Arab", "Arabic"),
    # English (pass-through – no translation needed)
    "en": ("eng_Latn", "English"),
}

# NLLB code for English (the translation target).
ENGLISH_NLLB = "eng_Latn"


def whisper_to_nllb(whisper_lang: str) -> Optional[str]:
    """Convert a Whisper language code to an NLLB BCP-47 code.

    Parameters
    ----------
    whisper_lang:
        Two-letter ISO-639-1 code as returned by ``faster-whisper``
        (e.g. ``"es"``, ``"ar"``).

    Returns
    -------
    str or None
        Corresponding NLLB language code, or *None* if the language is
        not in the mapping.
    """
    entry = _LANGUAGE_MAP.get(whisper_lang.lower())
    return entry[0] if entry else None


def get_language_name(whisper_lang: str) -> str:
    """Return a human-readable name for a Whisper language code.

    Falls back to the raw code if the language is unknown.
    """
    entry = _LANGUAGE_MAP.get(whisper_lang.lower())
    return entry[1] if entry else whisper_lang.upper()


def is_supported(whisper_lang: str) -> bool:
    """Return True if the language is supported for translation."""
    return whisper_lang.lower() in _LANGUAGE_MAP


def needs_translation(whisper_lang: str) -> bool:
    """Return True if the detected language is not English."""
    return whisper_lang.lower() != "en"
