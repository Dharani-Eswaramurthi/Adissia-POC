from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import time
from dotenv import load_dotenv
from pathlib import Path
import json
import re
import datetime
import simpleaudio as sa  # Reliable cross‑platform WAV playback

"""
Enhanced Event‑Feedback Collector (v2)
=====================================
Fixes the Windows playback error by:
• Requesting TTS output as **WAV** (not MP3) which Windows natively supports.
• Using **simpleaudio** for playback instead of `playsound`, avoiding MCI codec issues.
Install extra dep:  `pip install simpleaudio`
"""

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ╭────────────────────────────── VAD Recorder ─────────────────────────────╮

def record_until_silence(fs: int = 22_050,
                          silence_secs: float = 3.0,
                          max_secs: int = 45,
                          threshold: float = 50.0):
    """Record until ≈ *silence_secs* of quiet or *max_secs* elapsed; return .wav path."""
    print("🎤  Listening… (speak now)")
    last_voice = time.time()
    frames = []

    def callback(indata, *_):
        nonlocal last_voice
        rms = np.sqrt(np.mean(indata ** 2)) * 1000
        frames.append(indata.copy())
        if rms > threshold:
            last_voice = time.time()

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        start = time.time()
        while True:
            sd.sleep(200)
            if time.time() - last_voice > silence_secs or time.time() - start > max_secs:
                break

    audio = np.concatenate(frames)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, fs)
    print(f"💾  Saved to {tmp.name}")
    return tmp.name

# ╭──────────────────────────── OpenAI Helpers ─────────────────────────────╮

def transcribe_audio(path: str) -> str:
    with open(path, "rb") as f:
        tr = client.audio.transcriptions.create(model="whisper-1", file=f)
    print("📝  Transcription:", tr.text.strip())
    return tr.text.strip()


def detect_language(text: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a language detector."},
            {"role": "user", "content": f"Identify the language of this text and return only the name.\n\n{text}"}
        ]
    )
    lang = r.choices[0].message.content.strip()
    print("🌐  Language:", lang)
    return lang


def translate(text: str, tgt: str) -> str:
    prompt = f"Translate this into {tgt} in a simple, friendly way:\n\n{text}"
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": prompt}
        ]
    )
    out = r.choices[0].message.content.strip()
    print("🔤  Translated:", out)
    return out


def talk(text: str, lang: str, tone: str = "positive", fname: str | None = None) -> str:
    if fname is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"speech_{ts}.wav"

    voice_map = {
        "positive": f"Use {lang} accent. Bright, upbeat, gentle smile.",
        "neutral": f"Use {lang} accent. Calm, even pacing.",
        "negative": "Soft, slower, empathetic."}
    instructions = voice_map.get(tone, voice_map["neutral"])

    path = Path(fname)
    print(f"🔈  Generating WAV → {path}")
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        format="wav",  # crucial – Windows‑friendly
        instructions=instructions,
    ) as resp:
        resp.stream_to_file(path)
    return str(path)


def playback_audio(wav_path: str):
    if not os.path.exists(wav_path):
        print("❌  File missing", wav_path)
        return
    print(f"▶️  {wav_path}")
    try:
        wave_obj = sa.WaveObject.from_wave_file(wav_path)
        play = wave_obj.play()
        play.wait_done()
    except Exception as e:
        print("❌  Playback error:", e)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

# sentiment helper

def sentiment(text: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Classify the sentiment as Positive, Neutral, or Negative."},
            {"role": "user", "content": text}
        ]
    )
    s = r.choices[0].message.content.lower()
    return "positive" if "pos" in s else "negative" if "neg" in s else "neutral"

# ╭──────────────────────────── Conversation Logic ─────────────────────────╮

def ask_follow_up(missing, convo):
    base = (
        "You are collecting event feedback conversationally. "
        "If many fields are blank, ask ONE open question. If only 1–2 blank, ask a short specific question. "
        "Keep language simple – no numbers or rating scales."
    )
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": base},
            {"role": "assistant", "content": f"Missing: {missing}"},
            {"role": "assistant", "content": f"Conversation so far:\n{convo}"}
        ]
    )
    q = r.choices[0].message.content.strip()
    print("❓  Follow‑up:", q)
    return q


def extract_form(text: str, fields):
    prompt = (
        "Extract answers for these fields from the conversation. If not found leave empty. Output JSON only.\n\n" +
        f"Fields: {fields}\n\nConversation:\n{text}"
    )
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}]
    )
    m = re.search(r"\{[\s\S]*\}", r.choices[0].message.content)
    return json.loads(m.group(0)) if m else {}

# ╭──────────────────────────── Main Entry ─────────────────────────────╮

if __name__ == "__main__":
    fields = [
        "Event Name", "Date Attended", "Overall Satisfaction", "Speaker Effectiveness",
        "Relevance of Content", "Usefulness of Takeaways", "Organization and Logistics",
        "Likelihood to Recommend", "Suggestions for Improvement"
    ]

    first_wav = record_until_silence()
    transcript = transcribe_audio(first_wav)
    lang = detect_language(transcript)

    intro = translate("Hi! I’ll ask a few quick questions about the event – just speak naturally. Say 'done' when you’re finished.", lang)
    playback_audio(talk(intro, lang))

    convo = transcript + "\n"
    history = []
    stop_words = ["done", "enough", "thank you", "முடிச்சாச்சு"]

    while True:
        data = extract_form(convo, fields)
        missing = [f for f in fields if not data.get(f)]
        if not missing:
            break

        nxt = ask_follow_up(missing, convo)
        playback_audio(talk(translate(nxt, lang), lang, sentiment(convo)))

        ans_wav = record_until_silence()
        ans_txt = transcribe_audio(ans_wav)
        if any(w in ans_txt.lower() for w in stop_words):
            break
        convo += ans_txt + "\n"
        history.append({"text": ans_txt, "emotion": sentiment(ans_txt)})

    print("\n✅  Final structured data:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    with open("event_feedback_final.json", "w", encoding="utf-8") as f:
        json.dump({"form_data": data, "responses": history}, f, indent=2, ensure_ascii=False)

    outro = translate("Thanks so much for sharing your thoughts!", lang)
    playback_audio(talk(outro, lang, "positive"))
