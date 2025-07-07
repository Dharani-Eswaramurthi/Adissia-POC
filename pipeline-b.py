# -*- coding: utf-8 -*-
"""
Pipeline‑B (v2.5) — WAV‑safe Windows Build
=========================================
Fully‑working voice feedback collector (VAD → Whisper → GPT‑4o → TTS) with
reliable **WAV** playback.

Fix in this version
-------------------
OpenAI’s TTS defaults to an internal compressed format if you don’t explicitly
request *wav*. We now pass `format="wav"` to the `audio.speech` endpoint, so
the file starts with a proper **RIFF** header and `simpleaudio` plays without
error.

Everything else (logging, UTF‑8 console safety, startup tests) is unchanged.
"""
from __future__ import annotations

# ── Stdlib ───────────────────────────────────────────────────────────────────
import sys, os, time, json, re, tempfile, datetime, traceback, logging, builtins
from pathlib import Path
from typing import List, Dict, Any

# ── Third‑party ──────────────────────────────────────────────────────────────
import numpy as np
import sounddevice as sd
import soundfile as sf
import simpleaudio as sa
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ╭──────────────────────── Console & Logger Setup ───────────────────────────╮
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

orig_print = builtins.print  # preserve original

def safe_print(*args, **kwargs):
    try:
        orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        orig_print(text.encode("ascii", "replace").decode("ascii"), **kwargs)

print = safe_print  # type: ignore

logging.basicConfig(
    filename="debug_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def log_unhandled(exc_type, exc_value, exc_tb):
    logging.error("UNHANDLED EXCEPTION", exc_info=(exc_type, exc_value, exc_tb))
    sys.__stderr__.write("\nUNHANDLED EXCEPTION (see debug_log.txt)\n")
    traceback.print_exception(exc_type, exc_value, exc_tb)

sys.excepthook = log_unhandled

# ╭──────────────────────── OpenAI Initialisation ────────────────────────────╮
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)
try:
    client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[{"role": "user", "content": "ping"}], max_tokens=1)
    print("OpenAI key OK ✔")
except OpenAIError as e:
    logging.error("OpenAI connectivity test failed", exc_info=True)
    print("FATAL: OpenAI test failed —", e)
    sys.exit(1)

# ╭──────────────────────── Audio Stack Self‑check ────────────────────────────╮

def assert_audio_stack():
    devs = sd.query_devices()
    if not any(d["max_input_channels"] > 0 for d in devs):
        raise RuntimeError("No microphone detected by sounddevice.")
    if not any(d["max_output_channels"] > 0 for d in devs):
        raise RuntimeError("No speaker detected by sounddevice.")

try:
    assert_audio_stack()
    print("Audio devices OK ✔")
except Exception as e:
    logging.error("Audio stack test failed", exc_info=True)
    print("FATAL:", e)
    sys.exit(1)

# ╭──────────────────────── Helper Functions ─────────────────────────────────╮

def record_until_silence(fs: int = 22_050, silence: float = 3.0, max_len: int = 45, thr: float = 50.0) -> str:
    print("Listening… (speak now)")
    last_voice = time.time()
    frames: List[np.ndarray] = []

    def cb(indata, *_):
        nonlocal last_voice
        rms = np.sqrt(np.mean(indata ** 2)) * 1000
        frames.append(indata.copy())
        if rms > thr:
            last_voice = time.time()

    with sd.InputStream(samplerate=fs, channels=1, callback=cb):
        start = time.time()
        while True:
            sd.sleep(200)
            if time.time() - last_voice > silence or time.time() - start > max_len:
                break

    audio = np.concatenate(frames) if frames else np.empty((0, 1))
    fp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(fp.name, audio, fs)
    print("Saved to", fp.name)
    return fp.name

# ── OpenAI helpers -----------------------------------------------------------

def transcribe_audio(path: str) -> str:
    with open(path, "rb") as f:
        tr = client.audio.transcriptions.create(model="whisper-1", file=f)
    print("Transcription:", tr.text.strip())
    return tr.text.strip()


def detect_language(text: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a language detector."},
            {"role": "user", "content": f"Identify the language of this text and return only the name.\n\n{text}"},
        ],
    )
    lang = r.choices[0].message.content.strip()
    print("Language:", lang)
    return lang


def translate(text: str, tgt: str) -> str:
    prompt = f"Translate this into {tgt} in a simple, friendly way:\n\n{text}"
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": prompt},
        ],
    )
    out = r.choices[0].message.content.strip()
    print("Translated:", out)
    return out


def talk(text: str, lang: str, tone: str = "neutral", fname: str | None = None) -> str:
    if fname is None:
        fname = f"speech_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"

    voice_map = {
        "positive": f"Use {lang} accent. Bright, upbeat, gentle smile.",
        "neutral": f"Use {lang} accent. Calm, even pacing.",
        "negative": "Soft, slower, empathetic.",
    }
    instructions = voice_map.get(tone, voice_map["neutral"])

    path = Path(fname)
    print("Generating WAV →", path)
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions=instructions,
        response_format="wav",  # Ensure RIFF header for simpleaudio playback
    ) as resp:
        resp.stream_to_file(path)
    return str(path)


def playback_audio(path: str):
    if not os.path.exists(path):
        print("File missing:", path)
        return
    try:
        sa.WaveObject.from_wave_file(path).play().wait_done()
    except Exception as e:
        logging.error("Playback error", exc_info=True)
        print("Playback error:", e)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

# ── GPT helpers --------------------------------------------------------------

def sentiment(text: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Classify the sentiment as Positive, Neutral, or Negative."},
            {"role": "user", "content": text},
        ],
    )
    lower = r.choices[0].message.content.lower()
    return "positive" if "pos" in lower else "negative" if "neg" in lower else "neutral"


def ask_follow_up(missing: List[str], convo: str) -> str:
    base = (
        "You are collecting event feedback conversationally. "
        "If many fields are blank, ask ONE open question. If only 1‑2 blank, ask a short specific question. "
        "Keep language simple – no numbers or rating scales."
    )
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": base},
            {"role": "assistant", "content": f"Missing: {missing}"},
            {"role": "assistant", "content": f"Conversation so far:\n{convo}"},
        ],
    )
    q = r.choices[0].message.content.strip()
    print("Follow‑up:", q)
    return q


def extract_form(text: str, fields: List[str]) -> Dict[str, Any]:
    prompt = (
        "Extract answers for these fields from the conversation. If not found leave empty. Output JSON only.\n\n"
        + f"Fields: {fields}\n\nConversation:\n{text}"
    )
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
    )
    m = re.search(r"\{[\s\S]*\}", r.choices[0].message.content)
    return json.loads(m.group(0)) if m else {}

# ╭──────────────────────── Main Flow ─────────────────────────────────────────╮

def main():
    try:
        fields = [
            "Event Name",
            "Date Attended",
            "Overall Satisfaction",
            "Speaker Effectiveness",
            "Relevance of Content",
            "Usefulness of Takeaways",
            "Organization and Logistics",
            "Likelihood to Recommend",
            "Suggestions for Improvement",
        ]

        first_wav = record_until_silence()
        transcript = transcribe_audio(first_wav)
        lang = detect_language(transcript)

        intro = translate("Hi! I will ask a few quick questions about the event – just speak naturally. Say 'done' when you are finished.", lang)
        playback_audio(talk(intro, lang))

        convo = transcript + "\n"
        history: List[Dict[str, str]] = []
        stop_words = ["done", "enough", "thank you", "முடிச்சாச்சு"]

        while True:
            data = extract_form(convo, fields)
            missing = [f for f in fields if not data.get(f)]
            if not missing:
                break

            follow_q = ask_follow_up(missing, convo)
            playback_audio(talk(translate(follow_q, lang), lang, sentiment(convo)))

            answer_wav = record_until_silence()
            answer_txt = transcribe_audio(answer_wav)
            if any(w in answer_txt.lower() for w in stop_words):
                break
            convo += answer_txt + "\n"
            history.append({"text": answer_txt, "emotion": sentiment(answer_txt)})

        print("\nFinal structured data:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        with open("event_feedback_final.json", "w", encoding="utf-8") as f:
            json.dump({"form_data": data, "responses": history}, f, indent=2, ensure_ascii=False)

        outro = translate("Thanks so much for sharing your thoughts!", lang)
        playback_audio(talk(outro, lang, "positive"))
    except Exception:
        logging.error("Exception inside main()", exc_info=True)
        raise  # Let global excepthook handle

# ╭──────────────────────── Entry Point ───────────────────────────────────────╮
if __name__ == "__main__":
    main()
