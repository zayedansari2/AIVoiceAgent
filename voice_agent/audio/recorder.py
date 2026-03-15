"""
Microphone audio recorder.

Continuously captures audio from the default input device and pushes
fixed-duration chunks into a :class:`queue.Queue` for downstream
consumers (VAD, transcription, etc.).
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from voice_agent.config import Config


class AudioRecorder:
    """Continuous microphone capture using *sounddevice*.

    Audio is captured as 32-bit floats, resampled internally to
    ``config.sample_rate``, and placed into :attr:`audio_queue` as
    1-D :class:`numpy.ndarray` chunks whose length equals
    ``config.vad_chunk_ms`` milliseconds of audio.

    Example usage::

        recorder = AudioRecorder(config)
        recorder.start()
        try:
            while True:
                chunk = recorder.read()   # blocks until data is available
                process(chunk)
        finally:
            recorder.stop()
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._chunk_samples: int = int(
            config.sample_rate * config.vad_chunk_ms / 1000
        )
        # Unbounded queue – VAD/transcription must keep up.
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the input stream and start filling :attr:`audio_queue`."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stream = sd.InputStream(
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype="float32",
                blocksize=self._chunk_samples,
                callback=self._callback,
            )
            self._stream.start()

    def stop(self) -> None:
        """Stop capturing audio and close the input stream."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def read(self, timeout: Optional[float] = None) -> np.ndarray:
        """Block until a chunk is available and return it.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.  Raises :class:`queue.Empty` if the
            timeout expires without data.
        """
        return self.audio_queue.get(timeout=timeout)

    def __enter__(self) -> "AudioRecorder":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal sounddevice callback
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time: object,  # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice on every new audio block."""
        if status and self._config.debug:
            print(f"[AudioRecorder] sounddevice status: {status}")

        # indata shape: (frames, channels) – flatten to mono 1-D array.
        chunk = indata[:, 0].copy()
        self.audio_queue.put_nowait(chunk)
