# openai_realtime_full.py

import os
import json
import threading
import base64
import numpy as np
import websocket
import sounddevice as sd
import simpleaudio as sa
from dotenv import load_dotenv

# ─── Configuration ──────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment or .env file")

WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
HEADERS = [
    f"Authorization: Bearer {API_KEY}",
    "OpenAI-Beta: realtime=v1"
]

# Audio stream settings
SAMPLE_RATE = 24000    # 24 kHz
CHANNELS    = 1        # mono
BLOCKSIZE   = 4096     # number of frames per callback

# ─── Audio Capture Thread ──────────────────────────────────────────────────────

def audio_producer(ws):
    """
    Continuously capture mic audio and send it as base64-encoded buffers.
    Rely on server-side VAD; do NOT send any commit messages.
    """
    def callback(indata, frames, time_info, status):
        # indata is a numpy array of shape (frames, CHANNELS), dtype=float32
        # Convert to 16-bit PCM
        pcm16 = (indata * 32767).astype(np.int16)
        b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")
        msg = {"type": "input_audio_buffer.append", "audio": b64}
        ws.send(json.dumps(msg))

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCKSIZE,
        dtype="float32",
        callback=callback
    ):
        # Keep the stream alive indefinitely
        threading.Event().wait()

# ─── WebSocket Event Handlers ──────────────────────────────────────────────────

def on_open(ws):
    print("▶️ Connected to OpenAI Realtime API")
    # Enable server-side VAD (automated speech boundary detection)
    ws.send(json.dumps({
        "type": "session.update",
        "session": {"turn_detection": {"type": "server_vad"}}
    }))
    # Start capturing and streaming audio
    threading.Thread(target=audio_producer, args=(ws,), daemon=True).start()

def on_message(ws, message):
    event = json.loads(message)
    etype = event.get("type", "")
    if etype == "input_audio_buffer.speech_started":
        print("\n[User] 🎙️ started speaking")
    elif etype == "input_audio_buffer.speech_stopped":
        print("\n[User] 🛑 stopped speaking")
    elif etype == "transcript.delta":
        # Incremental ASR transcript
        print(f"[Transcript] {event['delta']}", end="", flush=True)
    elif etype == "response.audio.delta":
        # Play back the model's audio
        chunk = base64.b64decode(event["delta"])
        pcm = np.frombuffer(chunk, dtype=np.int16)
        play = sa.play_buffer(pcm, CHANNELS, 2, SAMPLE_RATE)
        play.wait_done()

def on_error(ws, error):
    print("❌ Error:", error)

def on_close(ws, code, reason):
    print(f"🔌 Connection closed: {reason or code}")

# ─── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ws_app = websocket.WebSocketApp(
        WS_URL,
        header=HEADERS,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    # Run the WebSocket client forever (reconnects on its own)
    ws_app.run_forever()
