# -*- coding: utf-8 -*-
"""
Adhissia Feedback Collector — Latency‑Optimised (v1.4)
=====================================================
**New conversation‑flow intelligence**

* Detects when the user *might* be finished and asks for confirmation instead
  of new questions.
* Remembers last bot prompt → if user says "repeat that" (or similar), it
  re‑prompts the same question rather than generating a new one.
* Maintains an `asked` set to guarantee we never ask about an already‑covered field.

Latency improvements from previous versions remain intact.
"""
from __future__ import annotations

import sys, os, time, json, re, tempfile, datetime, traceback, logging, builtins
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd
import soundfile as sf
import simpleaudio as sa
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

############################ Console / logger #################################
ORIG_PRINT = builtins.print
builtins.print = lambda *a, **k: ORIG_PRINT("[AFC]", *a, **k)
logging.basicConfig(filename="debug_log.txt", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

############################# OpenAI client ###################################
load_dotenv(); API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY: print("FATAL: OPENAI_API_KEY not set"); sys.exit(1)
client = OpenAI(api_key=API_KEY)
try:
    client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[{"role":"user","content":"ping"}], max_tokens=1)
except OpenAIError as e:
    print("FATAL: OpenAI connectivity check failed →", e); sys.exit(1)

################################ Audio utils ##################################
SAMPLERATE=16_000; SILENCE_SEC=4.0; MAX_LEN_SEC=600; RMS_THR=40.0

def record_until_silence(fs:int=SAMPLERATE, silence:float=SILENCE_SEC, max_len:int=MAX_LEN_SEC, thr:float=RMS_THR)->str:
    print("Listening… (speak now)"); last_voice=time.time(); frames:List[np.ndarray]=[]
    def cb(indata,*_):
        nonlocal last_voice; frames.append(indata.copy())
        if np.sqrt(np.mean(indata**2))*1000>thr: last_voice=time.time()
    with sd.InputStream(samplerate=fs, channels=1, callback=cb):
        start=time.time()
        while True:
            sd.sleep(100)
            if time.time()-last_voice>silence or time.time()-start>max_len: break
    audio=np.concatenate(frames) if frames else np.empty((0,1))
    tmp=tempfile.NamedTemporaryFile(suffix=".wav", delete=False); sf.write(tmp.name, audio, fs)
    print("Saved to", tmp.name); return tmp.name

################################ TTS helper ###################################

def speak(text: str, lang: str = "en") -> None:
    """Generate TTS audio, play it, then safely delete the temp file (Windows‑safe)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        # 1) Generate speech to tmp file
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

        # 2) Play via simpleaudio
        print("Playing TTS audio")
        wave_obj = sa.WaveObject.from_wave_file(tmp.name)
        play_obj = wave_obj.play()
        play_obj.wait_done()

    finally:
        # 3) Ensure file handle is released before deletion (Windows requirement)
        try:
            os.remove(tmp.name)
        except PermissionError:
            # In rare cases, give the OS a tiny delay then retry
            time.sleep(0.1)
            try:
                os.remove(tmp.name)
            except Exception as e:
                logging.warning("Temp file cleanup failed: %s", e)

############################### AI prompts ####################################
SYS_LANG ={"role":"system","content":"Return ONLY language name."}
SYS_SENT ={"role":"system","content":"Positive, Neutral, Negative."}
SYS_DONE ={"role":"system","content":"Does this text indicate the user wants to conclude? Respond Yes or No."}
SYS_EXTRACT={"role":"system","content":"""
                You are a helpful information extractor from a feedback given by user in the form of audio and is transcribed. You job is to extract information from a text and update in the respective field.

                Fields: Keys   
                Text: Convo

                STEP-BY-STEP INSTRUCTIONS:
                1. Firstly, understand and analyse the feedback text from the user.
                2. Extract what the user is trying to say from the feedback.
                3. Now understand the given form fields that needs to be filled or updated.
                4. Put the information in that field, if it is empty put it, else update the field alongwith the current extracted information.
                5. If no specific feedback were extracted, leave it empty.
                6. Provide the final out as only JSON.
"""}
SYS_FOLLOW ={"role":"system","content": f"""
        You are collecting project feedback conversationally. Ask efficient simple combined questions to gather feedback in short amount of time
        This system is intended to ask followup questions to gather a feedback, not is a single go but by going forth and back through questioaning and answering.

        Form Fields: missing + asked
        Convo: convo
        Last Question: last_q
        Confirmation for User Request: user_req_repeat
        
        STEP-BY-STEP INSTRUNCTIONS:
        1. Understand the filled fields first.
        2. Now understand the missing fields then.
        3. Ask a question to gather information for the missing fields, and avoid asking questions related to filled fields.
        4. If you think the filled fields are vague and specified in general ask about specifics if necessary.
        5. Ask a question, if they can be related in a single question to be more effective.
        6. Use friendly yet professional tone, appreciate user whenever necessary, but not all the time.
        7. Also help user, if he/she wants to repeat the last question, instead of moving forward with another question, modify the question in a way to make the user understand it better in simple words.
        8. If the Confirmation is True, ask a confirmation question for the last question for further actions.
        6. Keep the question short and crisp.
    """}

################################ AI wrappers ##################################

def detect_language(text:str)->str:
    print(f"Starting language detection for: {text}")
    start_time = time.time()  # Start time for language detection
    r = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[SYS_LANG, {"role":"user", "content": text}], max_tokens=3)
    language_latency = time.time() - start_time  # Time taken for language detection
    print(f"Language Detection Latency: {language_latency:.4f} seconds")
    return r.choices[0].message.content.strip()

def sentiment(text:str)->str:
    print(f"Starting sentiment analysis for: {text}")
    start_time = time.time()  # Start time for sentiment analysis
    r = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[SYS_SENT, {"role":"user", "content": text}], max_tokens=2)
    sentiment_latency = time.time() - start_time  # Time taken for sentiment analysis
    print(f"Sentiment Analysis Latency: {sentiment_latency:.4f} seconds")
    return r.choices[0].message.content.strip().lower()

def user_wants_done(text:str)->bool:
    print(f"Starting check for user conclusion: {text}")
    start_time = time.time()  # Start time for conclusion check
    r = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[SYS_DONE, {"role":"user", "content": text}], max_tokens=2)
    done_latency = time.time() - start_time  # Time taken for conclusion check
    print(f"User Conclusion Check Latency: {done_latency:.4f} seconds")
    return r.choices[0].message.content.strip().lower().startswith("y")

def extract_form(conv:str, keys:List[str])->Dict[str,str]:
    print(f"Starting form extraction for conversation: {conv[:50]}...")  # Only previewing the start for brevity
    start_time = time.time()  # Start time for form extraction
    r = client.chat.completions.create(model="gpt-4o-mini", messages=[SYS_EXTRACT, {"role":"user", "content": json.dumps(keys) + "\n" + conv}], max_tokens=256)
    form_extraction_latency = time.time() - start_time  # Time taken for form extraction
    print(f"Form Extraction Latency: {form_extraction_latency:.4f} seconds")
    m = re.search(r"\{[\s\S]*?\}", r.choices[0].message.content)
    return json.loads(m.group(0)) if m else {}

def next_question(missing:List[str], asked:List[str], last_q:str, user_req_repeat:bool, convo: List[str])->str:
    payload = {"missing_keys": missing, "asked_keys": asked, "last_question": last_q, "repeat": user_req_repeat, "conversation": convo}
    print(f"Starting next question generation for: {json.dumps(payload)[:50]}...")  # Only previewing the start for brevity
    start_time = time.time()  # Start time for question generation
    r = client.chat.completions.create(model="gpt-4o-mini", messages=[SYS_FOLLOW, {"role":"user", "content": json.dumps(payload)}], max_tokens=60)
    question_latency = time.time() - start_time  # Time taken for question generation
    print(f"Next Question Latency: {question_latency:.4f} seconds")
    return r.choices[0].message.content.strip()

################################ Main routine #################################
FIELDS = ["Overall handover satisfaction", "Material quality (tiles/paint/fixtures)", "Finishing (paint/fittings/floor)", 
          "Progress communication", "Timeline adherence / delay reason", "Supervisor attentiveness", "Reworks escalated", 
          "Kept informed of changes", "Heard on customisations", "Layout matches promise", "Utilities per plan", 
          "Post-handover issues", "Support responsiveness", "Cost / payment transparency", "Concerns taken seriously", 
          "One thing to change", "Would recommend"]
STOP_WORDS = {"finished", "done"}

def main():
    wav = record_until_silence()
    start_time = time.time()  # Start time for speech-to-text
    first_txt = client.audio.transcriptions.create(model="whisper-1", file=open(wav, "rb")).text.strip()
    stt_latency = time.time() - start_time  # Time taken for speech-to-text
    print(f"STT Processing Latency: {stt_latency:.4f} seconds")
    
    lang = detect_language(first_txt)
    speak("Hi! Let's capture your feedback on the recent Adhissia project. Say 'finished' any time to stop.", lang)

    convo = first_txt + "\n"
    history = []
    asked = []
    last_q = ""
    
    while True:
        data = extract_form(convo, FIELDS)
        missing = [k for k in FIELDS if not data.get(k)]
        if not missing:
            confirm = "Looks like we've covered everything. Would you like to add anything else before we end?"
            print("BOT:", confirm)
            speak(confirm, lang)
            wav = record_until_silence()
            ans = client.audio.transcriptions.create(model="whisper-1", file=open(wav, "rb")).text.strip()
            if user_wants_done(ans):
                break
            convo += ans + "\n"
            continue

        repeat_req = False
        if history and re.search(r"repeat|again|pardon", history[-1]["text"], re.I):
            repeat_req = True

        print(f"Whole convo: {convo},\n History: {history}")
        q = next_question(missing, asked, last_q, repeat_req, convo)
        if q.strip() == "---DONE---":
            break
        print("BOT:", q)
        speak(q, lang)
        last_q = q
        asked.extend([k for k in missing if k not in asked])

        wav = record_until_silence()
        ans_txt = client.audio.transcriptions.create(model="whisper-1", file=open(wav, "rb")).text.strip()
        if any(w in ans_txt.lower() for w in STOP_WORDS) or user_wants_done(ans_txt):
            confirm = "Just to confirm, would you like to finish the feedback session now?"
            print("BOT:", confirm)
            speak(confirm, lang)
            wav = record_until_silence()
            conf_ans = client.audio.transcriptions.create(model="whisper-1", file=open(wav, "rb")).text.strip()
            if user_wants_done(conf_ans):
                break
            convo += conf_ans + "\n"
            continue
        convo += ans_txt + "\n"
        history.append({"text": ans_txt, "emotion": sentiment(ans_txt)})

    print("★ Collected Feedback JSON\n", json.dumps(data, indent=2, ensure_ascii=False))
    with open("adhissia_feedback.json", "w", encoding="utf-8") as fp:
        json.dump({"form_data": data, "responses": history}, fp, indent=2, ensure_ascii=False)
    speak("Thanks for your time and feedback!", lang)

if __name__ == "__main__":
    ThreadPoolExecutor(max_workers=4)
    main()
    