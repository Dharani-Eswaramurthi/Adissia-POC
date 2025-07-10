import streamlit as st
import tempfile
import os
import json
from pathlib import Path

# Import the main logic from pipeline-b.py
import sys
sys.path.append(str(Path(__file__).parent))
import pipeline_b as afc

st.set_page_config(page_title="Adhissia Feedback Collector", layout="centered")
st.title("🏗️ Adhissia Project Feedback Collector")

if "convo" not in st.session_state:
    st.session_state.convo = ""
    st.session_state.history = []
    st.session_state.asked = []
    st.session_state.last_q = ""
    st.session_state.data = {}
    st.session_state.missing = afc.FIELDS.copy()
    st.session_state.lang = "en"
    st.session_state.finished = False

def reset():
    for k in ["convo","history","asked","last_q","data","missing","lang","finished"]:
        if k in st.session_state: del st.session_state[k]

st.sidebar.button("🔄 Restart Session", on_click=reset)

st.markdown("### Step 1: Record your feedback")
audio_bytes = st.audio_recorder("Record your feedback (click to start/stop)", key="audio1")
if audio_bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        wav_path = tmp.name
    st.success("Audio recorded. Transcribing...")
    first_txt = afc.client.audio.transcriptions.create(model="whisper-1", file=open(wav_path, "rb")).text.strip()
    st.session_state.lang = afc.detect_language(first_txt)
    st.session_state.convo = first_txt + "\n"
    st.session_state.history = []
    st.session_state.asked = []
    st.session_state.last_q = ""
    st.session_state.data = {}
    st.session_state.missing = afc.FIELDS.copy()
    st.session_state.finished = False
    os.remove(wav_path)
    st.write("**Transcribed:**", first_txt)
    st.success("Language detected: " + st.session_state.lang)

if st.session_state.convo:
    st.markdown("### Step 2: Feedback Q&A")
    while not st.session_state.finished:
        st.session_state.data = afc.extract_form(st.session_state.convo, afc.FIELDS)
        st.session_state.missing = [k for k in afc.FIELDS if not st.session_state.data.get(k)]
        if not st.session_state.missing:
            st.info("All fields covered. Would you like to add anything else?")
            break
        repeat_req = False
        if st.session_state.history and any(w in st.session_state.history[-1]["text"].lower() for w in ["repeat", "again", "pardon"]):
            repeat_req = True
        q = afc.next_question(st.session_state.missing, st.session_state.asked, st.session_state.last_q, repeat_req)
        if q.strip() == "---DONE---":
            st.session_state.finished = True
            break
        st.session_state.last_q = q
        st.session_state.asked.extend([k for k in st.session_state.missing if k not in st.session_state.asked])
        st.info(f"**Bot:** {q}")
        user_audio = st.audio_recorder("Your answer (click to record)", key=f"audio_{len(st.session_state.history)}")
        if user_audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(user_audio)
                tmp.flush()
                wav_path = tmp.name
            ans_txt = afc.client.audio.transcriptions.create(model="whisper-1", file=open(wav_path, "rb")).text.strip()
            os.remove(wav_path)
            st.write("**You:**", ans_txt)
            if any(w in ans_txt.lower() for w in afc.STOP_WORDS) or afc.user_wants_done(ans_txt):
                st.session_state.finished = True
                break
            st.session_state.convo += ans_txt + "\n"
            st.session_state.history.append({"text": ans_txt, "emotion": afc.sentiment(ans_txt)})
            st.experimental_rerun()
        else:
            st.stop()

if st.session_state.finished:
    st.success("Feedback session complete!")
    st.markdown("### Collected Feedback")
    st.json(st.session_state.data)
    st.markdown("### All Responses")
    st.json(st.session_state.history)
    if st.button("Download Feedback JSON"):
        feedback = {"form_data": st.session_state.data, "responses": st.session_state.history}
        st.download_button("Download", data=json.dumps(feedback, indent=2, ensure_ascii=False), file_name="adhissia_feedback.json", mime="application/json")
