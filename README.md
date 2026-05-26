# Sign Translator — ASL Model

Real-time American Sign Language (ASL) fingerspelling recognition with AI-powered sentence cleanup and text-to-speech.

## Features

- Live webcam recognition via WebRTC (works on Streamlit Cloud)
- Dual-hand tracking with dominant-hand selection
- Groq (Llama) translation with Google Gemini fallback
- gTTS speech output

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Optional API keys (for “Clean & Speak”): create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-key"
GEMINI_API_KEY = "your-gemini-key"
```

## Deploy on Streamlit Community Cloud

**Important:** Do **not** add a `packages.txt` file (apt packages break on Streamlit Cloud). Hand tracking uses the bundled `hand_landmarker.task` in this repo. If a previous deploy failed, **delete the app** on Streamlit Cloud and create it again.

1. Push this repo to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → select **mithunkm70263/Sign-Translator-ASL-model**.
4. Set **Main file path** to `app.py`.
5. Under **Advanced settings** → **Secrets**, paste:

```toml
GROQ_API_KEY = "your-groq-key"
GEMINI_API_KEY = "your-gemini-key"
```

6. Click **Deploy**. Wait for the build to finish (first deploy may take several minutes).

The app works without API keys for sign recognition; keys are only needed for AI sentence polishing.

## Repository files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit entry point |
| `asl_model.pkl`, `labels.pkl` | Trained classifier |
| `requirements.txt` | Python dependencies |
| `packages.txt` | Linux system libs (OpenCV) |
| `runtime.txt` | Python 3.11 |
