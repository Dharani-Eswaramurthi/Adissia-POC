# Adissia-POC: Enhanced Event-Feedback Collector

This project is an advanced conversational event feedback collector using OpenAI's APIs for speech-to-text, translation, and sentiment analysis. It records user feedback via microphone, transcribes and analyzes it, and outputs structured data in JSON format. The system is optimized for Windows, using WAV audio and simpleaudio for playback.

## Features
- Voice-activated recording with silence detection
- Transcription using OpenAI Whisper
- Language detection and translation
- Conversational follow-up questions to fill missing feedback fields
- Sentiment analysis for each response
- Text-to-speech (TTS) output in WAV format (Windows-friendly)
- Structured output saved as JSON

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Setup
1. Clone this repository.
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key in a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage
Run the main script:
```cmd
python pipeline-b.py
```

Follow the voice prompts. Speak naturally; say "done" when finished. The final structured feedback will be saved as `event_feedback_final.json`.

## Notes
- Audio is saved and played back as WAV for maximum Windows compatibility.
- Uses `simpleaudio` for playback (no codec issues on Windows).
- All temporary audio files are deleted after playback.

## License
MIT License
