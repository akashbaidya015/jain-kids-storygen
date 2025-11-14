import os
import time
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Gemini SDK
from google import genai
from google.genai import types

# Optional Google Cloud TTS
HAS_GCLOUD_TTS = True
try:
    from google.cloud import texttospeech
except Exception:
    HAS_GCLOUD_TTS = False

# ---------- Setup ----------
BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SYSTEM_PROMPT = (BASE_DIR / "prompts" / "system_story_prompt.txt").read_text(encoding="utf-8")

AGE_BANDS = {
    "4–6": {"level": "early reader", "min_words": 120, "max_words": 180},
    "7–9": {"level": "elementary", "min_words": 220, "max_words": 320},
    "10–12": {"level": "middle grade", "min_words": 350, "max_words": 500},
}
VALUES = [
    "Ahimsa (non-violence)",
    "Aparigraha (non-possessiveness)",
    "Satya (truthfulness)",
    "Asteya (non-stealing)",
    "Brahmacharya (self-discipline)",
]

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-1.5-flash"  # fast and good for kids stories

def build_user_prompt(age_band: str, value_choice: str, hint: str, cfg: dict) -> str:
    return f"""
Write a children’s story that teaches the Jain value: "{value_choice}".
Age band: {age_band} ({cfg['level']})
Target length: {cfg['min_words']}–{cfg['max_words']} words.
Use an everyday setting and relatable characters.
End with:
Moral of the story: <1–2 short lines>

Extra hint (optional): {hint.strip() if hint else "None"}
"""

def generate_story(age_band: str, value_choice: str, extra_hint: str, temperature: float = 0.7) -> str:
    cfg = AGE_BANDS[age_band]
    user_prompt = build_user_prompt(age_band, value_choice, extra_hint, cfg)

    resp = client.responses.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=1200,
        ),
        contents=[types.Content(role="user", parts=[types.Part.from_text(user_prompt)])],
    )

    # Prefer output_text, fallback to text
    text = getattr(resp, "output_text", None) or getattr(resp, "text", "")
    return text.strip()

def synthesize_google_tts(text: str) -> Optional[bytes]:
    """Return MP3 bytes using Google Cloud TTS if configured, else None."""
    if not HAS_GCLOUD_TTS:
        return None
    # If no credentials env, return None (browser TTS will handle playback)
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return None

    client_tts = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Neutral child-friendly English voice (you can tweak)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-C",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.02,
        pitch=0.0,
    )
    audio = client_tts.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return audio.audio_content

def browser_tts_component(text: str):
    """Fallback TTS using the browser's Web Speech API via a small HTML/JS component."""
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("</", "<\\/").replace("\n", "\\n")
    html = f"""
    <div>
      <button id="playTTS">Play</button>
      <button id="pauseTTS">Pause</button>
      <button id="stopTTS">Stop</button>
    </div>
    <script>
      const text = `{escaped}`;
      let utterance = null;

      function ensureUtterance() {{
        if (!utterance) {{
          utterance = new SpeechSynthesisUtterance(text);
          utterance.rate = 1.02;
          utterance.pitch = 1.0;
        }}
        return utterance;
      }}

      document.getElementById("playTTS").onclick = () => {{
        const u = ensureUtterance();
        if (speechSynthesis.paused) {{
          speechSynthesis.resume();
        }} else {{
          speechSynthesis.cancel();
          speechSynthesis.speak(u);
        }}
      }};
      document.getElementById("pauseTTS").onclick = () => speechSynthesis.pause();
      document.getElementById("stopTTS").onclick = () => speechSynthesis.cancel();
    </script>
    """
    st.components.v1.html(html, height=60)

# ---------- UI ----------
st.set_page_config(page_title="Jain Kids Story Generator (Gemini)", page_icon="📖", layout="centered")
st.title("📖 AI Story Generator for Kids — Gemini")
st.caption("Pick an age range and a Jain value to create a short, uplifting story with a clear moral.")

with st.form("story_form"):
    c1, c2 = st.columns(2)
    with c1:
        age_band = st.selectbox("Age range", list(AGE_BANDS.keys()), index=0)
    with c2:
        value_choice = st.selectbox("Value", VALUES, index=0)
    extra_hint = st.text_input("Optional: setting or character hint (e.g., school garden, siblings, teacher)")
    temp = st.slider("Creativity", 0.0, 1.0, 0.7, 0.05)
    use_cloud_tts = st.checkbox("Use Google Cloud Text-to-Speech if available", value=False)
    submitted = st.form_submit_button("Generate Story")

if submitted:
    if not GOOGLE_API_KEY:
        st.error("Missing GOOGLE_API_KEY in .env")
        st.stop()

    with st.spinner("Creating your story..."):
        try:
            story = generate_story(age_band, value_choice, extra_hint, temperature=temp)
        except Exception as e:
            st.error(f"Story generation failed: {e}")
            st.stop()

    st.subheader("Your Story")
    st.write(story)

    st.download_button(
        label="Download as .txt",
        data=story.encode("utf-8"),
        file_name=f"jain_story_{age_band.replace('–','-')}.txt",
        mime="text/plain",
    )

    # Audio options
    st.markdown("### Read Aloud")
    audio_done = False
    if use_cloud_tts:
        audio_bytes = synthesize_google_tts(story)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
            st.success("Cloud audio ready")
            audio_done = True
        else:
            st.info("Google Cloud TTS not configured. Falling back to browser TTS below.")

    if not audio_done:
        st.caption("Play using your browser’s voice:")
        browser_tts_component(story)

st.markdown("---")
st.caption("Tip: keep creativity between 0.5 and 0.8 for balanced variety. Add a small hint for context, like “festival day at school.”")
