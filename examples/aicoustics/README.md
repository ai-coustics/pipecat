# ai|coustics Enhanced Chatbot Example

## 🛠 Installation

Adapt the top-level pyproject.toml to use the [aicoustics-wheel](https://github.com/ai-coustics/aic-sdk-py/releases/tag/v0.5.0a1) which suits your OS.

1. **Set up a virtual environment:**

    ```bash
    uv venv --python 3.10
    ````

2. **Install dependencies:**

    ```bash
    uv pip install -e ".[aicoustics,silero,daily,deepgram,openai,websocket,webrtc]"
    uv pip install -r examples/aicoustics/requirements.txt
    ```

3. **Create a `.env` file** with the following keys:

   ```
   AICOUSTICS_LICENSE_KEY=...
   OPENAI_API_KEY=...
   DEEPGRAM_API_KEY=...
   ```

   This file provides the necessary credentials for the enhancement and transcription services.

---

## ▶️ Run Chatbot Example

This example runs a real-time voice chatbot where the input audio is enhanced using [ai|coustics](https://www.ai-coustics.com/) speech enhancement.

### Benefits:

* Improved speech-to-text accuracy.
* Faster, more natural turn-taking.
* Better reliability with short, low-energy utterances (e.g., “yes”, “no”).

### Run:

```bash
source .venv/bin/activate
python examples/aicoustics/interruptible_aicoustics.py
```

---

## 📋 Notes

* Make sure your microphone and audio input permissions are correctly configured.
* This example uses ai|coustics’ real-time enhancement pipeline combined with OpenAI and Deepgram for language and transcription features.

