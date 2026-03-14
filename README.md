# AIVoiceAgent

A **real-time multilingual AI voice agent** that runs fully locally – no
external APIs required.  Speak in Spanish, Hindi/Urdu, or Arabic and the
system prints and speaks back the English translation in near real time.

---

## Architecture

```
Microphone
    │
    ▼
AudioRecorder  (sounddevice, 16 kHz mono, 30 ms chunks)
    │
    ▼
SileroVAD      (torch.hub — detects speech segments)
    │
    ▼
WhisperStream  (faster-whisper — transcription + language detection)
    │
    ▼
Translator     (facebook/nllb-200-distilled-600M — → English)
    │
    ▼  [optional]
LLMProcessor   (llama-cpp-python — refines translation quality)
    │
    ▼
Speaker        (Piper TTS / pyttsx3 fallback — English speech output)
```

---

## Supported Languages

| Language    | Whisper code | NLLB-200 code |
|-------------|:------------:|:-------------:|
| Spanish     | `es`         | `spa_Latn`    |
| Hindi       | `hi`         | `hin_Deva`    |
| Urdu        | `ur`         | `urd_Arab`    |
| Arabic      | `ar`         | `arb_Arab`    |

---

## Project Structure

```
AIVoiceAgent/
├── main.py                        # CLI entry point
├── requirements.txt
├── README.md
└── voice_agent/
    ├── config.py                  # Centralised configuration
    ├── main.py                    # VoiceAgent controller
    ├── audio/
    │   ├── recorder.py            # Microphone capture
    │   └── vad.py                 # Silero VAD wrapper
    ├── speech/
    │   └── whisper_stream.py      # faster-whisper transcription
    ├── translation/
    │   └── translator.py          # NLLB-200 translation
    ├── reasoning/
    │   └── llm_processor.py       # Optional LLM post-processing
    ├── tts/
    │   └── speaker.py             # Piper TTS / pyttsx3 output
    └── utils/
        └── language_detection.py  # Language code utilities
```

---

## Hardware Requirements

| Resource | Requirement |
|----------|-------------|
| CPU      | Any modern x86-64 or Apple Silicon (no GPU needed) |
| RAM      | ≥ 4 GB available (models use ~2–2.5 GB total) |
| Storage  | ~2 GB for model downloads |
| OS       | macOS, Linux, or Windows |

Tested on a MacBook Air (M-series / Intel) without a discrete GPU.

---

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS note:** If `sounddevice` fails to find a microphone, install
> PortAudio first:
> ```bash
> brew install portaudio
> ```

### 3. (Optional) Install Piper TTS voice model

Piper TTS needs a voice model file (`.onnx` + `.onnx.json`).  Download
the `en_US-lessac-medium` voice (≈ 60 MB) and place it in
`~/.local/share/piper-voices/`:

```bash
mkdir -p ~/.local/share/piper-voices
cd ~/.local/share/piper-voices

# Download voice model
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx.json
```

If the voice model is not found, the agent automatically falls back to
`pyttsx3` (uses the OS built-in speech synthesiser).

### 4. (Optional) Enable LLM reasoning

To enable the optional LLM post-processing step:

1. Uncomment `llama-cpp-python` in `requirements.txt` and reinstall:
   ```bash
   pip install llama-cpp-python
   ```
2. Download a small GGUF model, for example
   [Llama-3.2-1B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF).
3. Pass the path when running:
   ```bash
   python main.py --enable-llm --llm-model /path/to/model.gguf
   ```

---

## Usage

### Basic (default settings)

```bash
python main.py
```

The agent starts listening immediately.  Speak in Spanish, Hindi, Urdu,
or Arabic and the translation is printed and spoken aloud.

### CLI options

```
python main.py [OPTIONS]

Options:
  --debug                    Print verbose debug information
  --whisper-model {tiny,base,small,medium}
                             Model size for speech recognition (default: base)
  --enable-llm               Enable the optional LLM reasoning layer
  --llm-model PATH           Path to a GGUF model for LLM reasoning
  --tts-engine {piper,pyttsx3}
                             TTS engine (default: piper)
  --piper-voice VOICE        Piper voice name (default: en_US-lessac-medium)
```

### From Python

```python
from voice_agent.config import Config
from voice_agent.main import VoiceAgent

config = Config(
    debug=True,
    whisper_model_size="base",   # tiny | base | small | medium
    enable_llm=False,
)

agent = VoiceAgent(config)
agent.run()   # blocks; press Ctrl-C to stop
```

---

## Configuration

All parameters are in `voice_agent/config.py`.  Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `whisper_model_size` | `"base"` | Whisper model size (`tiny` is fastest, `small` is more accurate) |
| `whisper_compute_type` | `"int8"` | Quantisation (`int8` saves RAM on CPU) |
| `vad_threshold` | `0.5` | Speech probability threshold for Silero VAD |
| `translation_model` | `"facebook/nllb-200-distilled-600M"` | HuggingFace model for translation |
| `enable_llm` | `False` | Enable LLM post-processing |
| `tts_engine` | `"piper"` | TTS engine (`"piper"` or `"pyttsx3"`) |

---

## Model Downloads (first run)

On the first run the following models are downloaded automatically:

| Model | Size | Destination |
|-------|------|-------------|
| Silero VAD | ~5 MB | `~/.cache/torch/hub/` |
| faster-whisper base | ~148 MB | `~/.cache/huggingface/hub/` |
| NLLB-200-distilled-600M | ~1.2 GB | `~/.cache/huggingface/hub/` |
| Piper voice (optional) | ~60 MB | `~/.local/share/piper-voices/` |

Total (excluding optional LLM): ~1.4 GB.

---

## Troubleshooting

**"No microphone found"**  
→ Make sure your microphone is connected and the app has permission to
access it (macOS: System Settings → Privacy → Microphone).

**Translation is slow**  
→ Try a smaller Whisper model: `--whisper-model tiny`.  The NLLB model
takes ~1–2 s per sentence on modern CPU hardware.

**Piper TTS not working**  
→ Check that the `.onnx` and `.onnx.json` voice files are in
`~/.local/share/piper-voices/`.  The agent falls back to `pyttsx3`
automatically.

**Out of memory**  
→ Use `--whisper-model tiny` and ensure no other large applications are
running.  The full pipeline uses ~2–2.5 GB RAM.
