from fastapi import FastAPI, File, UploadFile, List
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import os
import tempfile
import json
import logging
from threading import Thread
import time
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

# Load OpenAI API Key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("FATAL: OPENAI_API_KEY not set")
    exit(1)

client = OpenAI(api_key=API_KEY)

# Global Variables
SAMPLERATE = 16_000
SILENCE_SEC = 4.0
MAX_LEN_SEC = 600
RMS_THR = 40.0

# Set up logging
logging.basicConfig(filename="debug_log.txt", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

# Feedback Fields
FIELDS = ["Overall handover satisfaction", "Material quality (tiles/paint/fixtures)", "Finishing (paint/fittings/floor)",
          "Progress communication", "Timeline adherence / delay reason", "Supervisor attentiveness", "Reworks escalated",
          "Kept informed of changes", "Heard on customisations", "Layout matches promise", "Utilities per plan",
          "Post-handover issues", "Support responsiveness", "Cost / payment transparency", "Concerns taken seriously",
          "One thing to change", "Would recommend"]
STOP_WORDS = {"finished", "done"}

# Helper function to record audio
def record_until_silence(fs: int = SAMPLERATE, silence: float = SILENCE_SEC, max_len: int = MAX_LEN_SEC, thr: float = RMS_THR) -> str:
    print("Listening… (speak now)")
    last_voice = time.time()
    frames: List[np.ndarray] = []

    def cb(indata, *_):
        nonlocal last_voice
        frames.append(indata.copy())
        if np.sqrt(np.mean(indata**2)) * 1000 > thr:
            last_voice = time.time()

    with sd.InputStream(samplerate=fs, channels=1, callback=cb):
        start = time.time()
        while True:
            sd.sleep(100)
            if time.time() - last_voice > silence or time.time() - start > max_len:
                break
    audio = np.concatenate(frames) if frames else np.empty((0, 1))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, fs)
    print("Saved to", tmp.name)
    return tmp.name

# Endpoint for speech-to-text
@app.post("/stt/")
async def stt(file: UploadFile = File(...)):
    wav = tempfile.NamedTemporaryFile(delete=False)
    with open(wav.name, 'wb') as buffer:
        buffer.write(await file.read())
    
    start_time = time.time()  # Start time for speech-to-text
    stt_result = client.audio.transcriptions.create(model="whisper-1", file=open(wav.name, "rb")).text.strip()
    stt_latency = time.time() - start_time  # Time taken for speech-to-text
    print(f"STT Processing Latency: {stt_latency:.4f} seconds")
    
    return {"transcription": stt_result}

# Endpoint for next question
@app.post("/next_question/")
async def next_question(convo: str, missing_fields: list, asked_fields: list):
    payload = {"missing_keys": missing_fields, "asked_keys": asked_fields, "conversation": convo}
    
    start_time = time.time()  # Start time for question generation
    r = client.chat.completions.create(model="gpt-4o-mini", messages=[{
        "role": "system",
        "content": "You are collecting project feedback conversationally. Ask efficient questions to gather feedback."
    }, {"role": "user", "content": json.dumps(payload)}], max_tokens=60)
    question_latency = time.time() - start_time  # Time taken for question generation
    print(f"Next Question Latency: {question_latency:.4f} seconds")
    
    return {"next_question": r.choices[0].message.content.strip()}

# Endpoint for text-to-speech
@app.post("/tts/")
async def tts(text: str):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        print(f"Starting TTS processing for: {text}")
        start_time = time.time()  # Start time for TTS
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Calm, friendly",
            response_format="wav"
        ) as resp:
            resp.stream_to_file(tmp.name)
        tts_latency = time.time() - start_time  # Time taken for TTS
        print(f"TTS Processing Latency: {tts_latency:.4f} seconds")
        
        tmp.close()  # flush + close handle so Windows can reopen it
        
        # Return audio file as streaming response
        return StreamingResponse(open(tmp.name, "rb"), media_type="audio/wav")

    finally:
        os.remove(tmp.name)  # Cleanup the temporary file after usage

# Main routine
@app.get("/")
def read_root():
    return {"message": "Welcome to the Adhissia Feedback Collector API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
