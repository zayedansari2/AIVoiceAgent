"""
Optional LLM reasoning layer.

When enabled, the translated English text is passed through a small
local language model (loaded via *llama-cpp-python*) which can refine
phrasing, answer follow-up questions, or add context.

The component is intentionally lightweight:

* Model loading is guarded by ``config.enable_llm``.
* A GGUF-format model (e.g. Llama-3.2-1B-Q4_K_M.gguf) is required.
* The model path must be set in ``config.llm_model_path``.

If llama-cpp-python is not installed or no model path is provided, the
processor acts as a pass-through so the rest of the pipeline continues
to work.
"""

from __future__ import annotations

from typing import Optional

from voice_agent.config import Config

try:
    from llama_cpp import Llama  # type: ignore
    _LLAMA_CPP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LLAMA_CPP_AVAILABLE = False

# Prompt template for the reasoning step.
_SYSTEM_PROMPT = (
    "You are a helpful translation assistant.  "
    "The user provides text that was translated from a foreign language into English.  "
    "Your job is to improve the naturalness of the English translation and "
    "correct any obvious translation errors.  "
    "Reply ONLY with the improved English text, nothing else."
)


class LLMProcessor:
    """Optional LLM post-processing for translated text.

    If ``config.enable_llm`` is False (the default), :meth:`process`
    returns the input text unchanged.

    Parameters
    ----------
    config:
        Shared configuration object.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Optional[object] = None

        if config.enable_llm:
            self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the GGUF model via llama-cpp-python."""
        if not _LLAMA_CPP_AVAILABLE:
            print(
                "[LLMProcessor] WARNING: llama-cpp-python is not installed. "
                "LLM reasoning will be skipped.  "
                "Run: pip install llama-cpp-python"
            )
            return

        if not self._config.llm_model_path:
            print(
                "[LLMProcessor] WARNING: config.llm_model_path is not set. "
                "LLM reasoning will be skipped."
            )
            return

        if self._config.debug:
            print(f"[LLMProcessor] Loading model: {self._config.llm_model_path} …")

        self._model = Llama(  # type: ignore[operator]
            model_path=self._config.llm_model_path,
            n_ctx=self._config.llm_n_ctx,
            n_threads=4,
            verbose=False,
        )

        if self._config.debug:
            print("[LLMProcessor] Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> str:
        """Run LLM post-processing on *text*.

        Parameters
        ----------
        text:
            Translated English text.

        Returns
        -------
        str
            Refined text from the LLM, or *text* unchanged if LLM is
            disabled / unavailable.
        """
        if not self._config.enable_llm or self._model is None:
            return text

        if self._config.debug:
            print(f"[LLMProcessor] Processing: {text!r}")

        prompt = (
            f"<|system|>\n{_SYSTEM_PROMPT}\n"
            f"<|user|>\n{text}\n"
            "<|assistant|>\n"
        )

        response = self._model(  # type: ignore[operator]
            prompt,
            max_tokens=self._config.llm_max_tokens,
            stop=["<|user|>", "<|system|>"],
            echo=False,
        )

        refined: str = response["choices"][0]["text"].strip()  # type: ignore[index]

        if self._config.debug:
            print(f"[LLMProcessor] Result: {refined!r}")

        return refined if refined else text
