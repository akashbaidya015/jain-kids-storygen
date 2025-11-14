import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    speechsdk = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "sutras.json"


@dataclass
class Sutra:
    id: str
    title: str
    script: str
    transliteration: str
    translation: str
    locale: str
    voice: str
    tips: List[str]
    tts_language: str = "hi"


@dataclass
class SpeechBackend:
    mode: str  # "azure" or "local"
    speech_key: Optional[str] = None
    speech_region: Optional[str] = None


@st.cache_data(show_spinner=False)
def load_sutras() -> List[Sutra]:
    if not DATA_PATH.exists():
        st.error(f"❌ Could not locate sutra data at `{DATA_PATH}`.")
        st.stop()
    with DATA_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    sutras: List[Sutra] = []
    for item in raw:
        item.setdefault("tts_language", "hi")
        sutras.append(Sutra(**item))
    return sutras


@st.cache_resource
def get_speech_backend() -> SpeechBackend:
    load_dotenv()
    speech_key = os.getenv("AZURE_SPEECH_KEY") or os.getenv("SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION") or os.getenv("SPEECH_REGION")
    if speechsdk and speech_key and speech_region:
        return SpeechBackend("azure", speech_key=speech_key, speech_region=speech_region)
    return SpeechBackend("local")


@st.cache_resource
def load_whisper_model(model_size: str = "small") -> WhisperModel:
    if WhisperModel is None:
        st.error(
            "⚠️ `faster-whisper` is not installed. "
            "Install the optional dependency to enable local speech-to-text."
        )
        st.stop()
    return WhisperModel(model_size, device="cpu", compute_type="int8")


@st.cache_data(show_spinner="🎧 Generating reference audio...")
def get_reference_audio(
    sutra_id: str,
    text: str,
    alt_text: str,
    voice: str,
    tts_language: str,
    backend_mode: str,
    speech_key: Optional[str],
    speech_region: Optional[str],
) -> bytes:
    if backend_mode == "azure":
        if speechsdk is None:
            raise RuntimeError("Azure Speech SDK is not installed.")
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_synthesis_voice_name = voice
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = getattr(result, "cancellation_details", None)
            reason = getattr(details, "reason", "Unknown")
            message = getattr(details, "error_details", "")
            raise RuntimeError(f"Synthesis failed: {reason} {message}")
        stream = speechsdk.AudioDataStream(result)
        buffer = io.BytesIO()
        stream.save_to_wave_stream(buffer)
        buffer.seek(0)
        return buffer.read()

    if gTTS is None or AudioSegment is None:
        raise RuntimeError(
            "Local TTS requires `gTTS` and `pydub`. Install them and ensure FFmpeg is available."
        )

    # gTTS works best with transliteration for Latin characters; fall back to script.
    gtts_text = text.replace("\n", " ").strip()
    fallback_text = (alt_text or text).replace("\n", " ").strip()
    try:
        tts = gTTS(text=gtts_text, lang=tts_language)
    except ValueError:
        # If language not supported, retry with transliteration or Hindi fallback.
        try:
            tts = gTTS(text=fallback_text, lang=tts_language)
        except Exception:
            tts = gTTS(text=fallback_text, lang="hi")
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)

    audio_segment = AudioSegment.from_file(mp3_buffer, format="mp3")
    wav_buffer = io.BytesIO()
    audio_segment.set_frame_rate(16000).set_channels(1).export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer.read()


def ensure_wav(audio_bytes: bytes) -> bytes:
    """Audio from recorder comes as wav; uploads are restricted to wav. Return bytes unchanged."""
    return audio_bytes


def transcribe_audio(
    audio_bytes: bytes,
    locale: str,
    backend: SpeechBackend,
) -> Tuple[str, Optional[str]]:
    if backend.mode == "azure":
        if speechsdk is None:
            return "", "Azure Speech SDK is missing."
        speech_config = speechsdk.SpeechConfig(subscription=backend.speech_key, region=backend.speech_region)
        speech_config.speech_recognition_language = locale
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            temp_path = tmp_file.name
        try:
            audio_config = speechsdk.audio.AudioConfig(filename=temp_path)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            result = recognizer.recognize_once_async().get()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text.strip(), None
            if result.reason == speechsdk.ResultReason.NoMatch:
                return "", "We couldn't detect spoken audio. Try speaking closer to the microphone."
            if result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                return "", f"Recognition canceled: {details.reason}. {details.error_details}"
            return "", "Unexpected recognition response."
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if WhisperModel is None:
        return "", "Local speech-to-text requires the `faster-whisper` package."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name

    try:
        model = load_whisper_model()
        # Whisper expects language code like "hi" or "sa".
        language = locale.split("-")[0] if locale else None
        segments, _ = model.transcribe(temp_path, language=language, beam_size=5)
        transcript_parts = [segment.text.strip() for segment in segments if segment.text]
        transcript = " ".join(transcript_parts).strip()
        return transcript, None
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zāīūṛṝēōṃṁṅñṭḍṇśṣḷ\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[^\s]+", text)


def compute_pronunciation_feedback(
    reference_transliteration: str,
    recognized_text: str,
    preset_tips: List[str],
) -> Tuple[float, List[str], List[str], List[str]]:
    cleaned_reference = normalize_text(reference_transliteration)
    cleaned_recognized = normalize_text(recognized_text)

    if not cleaned_recognized:
        return 0.0, preset_tips, tokenize(cleaned_reference), []

    import difflib

    matcher = difflib.SequenceMatcher(None, cleaned_reference, cleaned_recognized)
    similarity = round(matcher.ratio() * 100, 1)

    ref_tokens = tokenize(cleaned_reference)
    rec_tokens = tokenize(cleaned_recognized)

    missing = [tok for tok in ref_tokens if tok not in rec_tokens]
    unexpected = [tok for tok in rec_tokens if tok not in ref_tokens]

    dynamic_tips: List[str] = []
    if similarity < 85:
        dynamic_tips.append("Slow down slightly and elongate long vowels such as ā, ī, ū.")
    if missing:
        dynamic_tips.append(
            "Revisit these syllables: " + ", ".join(dict.fromkeys(missing))
        )
    if unexpected:
        dynamic_tips.append(
            "Watch for extra sounds: " + ", ".join(dict.fromkeys(unexpected))
        )
    final_tips = list(dict.fromkeys(dynamic_tips + preset_tips))
    return similarity, final_tips, missing, unexpected


def render_reference_section(sutra: Sutra, backend: SpeechBackend):
    st.subheader("🎧 Listen to the Pronunciation")
    if st.button("Play Reference Audio", type="primary"):
        spinner_text = (
            "Requesting Azure Speech synthesizer..."
            if backend.mode == "azure"
            else "Generating sample audio with gTTS..."
        )
        with st.spinner(spinner_text):
            try:
                audio_bytes = get_reference_audio(
                    sutra.id,
                    sutra.script,
                    sutra.transliteration,
                    sutra.voice,
                    sutra.tts_language,
                    backend.mode,
                    backend.speech_key,
                    backend.speech_region,
                )
                st.session_state["reference_audio"] = audio_bytes
            except Exception as exc:
                st.error(f"Failed to synthesize audio: {exc}")
    if st.session_state.get("reference_audio"):
        st.audio(st.session_state["reference_audio"], format="audio/wav")
    st.caption(f"Voice: {sutra.voice} · Locale: {sutra.locale}")


def render_recording_section():
    st.subheader("🎙️ Practice Pronunciation")
    st.caption("Record directly or upload a `.wav` file not exceeding ~1 minute.")
    col1, col2 = st.columns(2)
    with col1:
        recorded_audio = audio_recorder(text="Tap to record", icon_name="microphone")
        if recorded_audio:
            st.session_state["practice_audio"] = ensure_wav(recorded_audio)
            st.success("Captured recording.")
    with col2:
        uploaded_audio = st.file_uploader("Upload audio (`.wav`)", type=["wav"])
        if uploaded_audio is not None:
            file_bytes = uploaded_audio.read()
            st.session_state["practice_audio"] = ensure_wav(file_bytes)
            st.success(f"Loaded `{uploaded_audio.name}`.")

    if audio_bytes := st.session_state.get("practice_audio"):
        st.audio(audio_bytes, format="audio/wav")


def render_analysis_section(sutra: Sutra, backend: SpeechBackend):
    st.subheader("🧠 Pronunciation Feedback")
    audio_bytes = st.session_state.get("practice_audio")
    if not audio_bytes:
        st.info("Record or upload your chanting, then click **Analyze Pronunciation**.")
        return

    if st.button("Analyze Pronunciation", type="primary"):
        spinner_text = (
            "Analyzing pronunciation with Azure Speech..."
            if backend.mode == "azure"
            else "Running local Whisper transcription..."
        )
        with st.spinner(spinner_text):
            transcript, error = transcribe_audio(
                audio_bytes,
                sutra.locale,
                backend,
            )
            if error:
                st.error(error)
                return
            if not transcript:
                st.warning("We couldn't understand the audio. Try again a bit louder or reduce background noise.")
                return

            score, tips, missing, unexpected = compute_pronunciation_feedback(
                sutra.transliteration,
                transcript,
                sutra.tips,
            )
            st.session_state["analysis_result"] = {
                "transcript": transcript,
                "score": score,
                "tips": tips,
                "missing": missing,
                "unexpected": unexpected,
            }

    if result := st.session_state.get("analysis_result"):
        st.metric("Pronunciation Score", f"{result['score']} / 100")
        st.write("**Transcription:**")
        st.info(result["transcript"])

        st.write("**Improvement Tips:**")
        for tip in result["tips"][:4]:
            st.markdown(f"- {tip}")

        if result["missing"]:
            st.write("**Words to focus on:** " + ", ".join(dict.fromkeys(result["missing"])))
        if result["unexpected"]:
            st.write("**Consider trimming extra sounds:** " + ", ".join(dict.fromkeys(result["unexpected"])))


def main():
    st.set_page_config(page_title="Sutra Pronunciation Tutor", page_icon="🔊", layout="wide")
    backend = get_speech_backend()

    st.title("🔊 Sutra Pronunciation Tutor")
    st.caption(
        "Guided Jain sutra chanting with pronunciation feedback. "
        "Uses Azure Speech when credentials are provided, otherwise falls back to free open-source tools."
    )

    sutras = load_sutras()
    sutra_lookup = {sutra.title: sutra for sutra in sutras}

    with st.sidebar:
        st.header("Practice Playlist")
        choice = st.selectbox("Choose a sutra to practice", list(sutra_lookup.keys()))
        st.divider()
        st.markdown("**How it works:**")
        reference_source = "Azure Speech" if backend.mode == "azure" else "gTTS (Google Text-to-Speech)"
        recognizer_source = "Azure Speech-to-Text" if backend.mode == "azure" else "Whisper (open-source)"
        st.markdown(
            f"1. Listen to the reference chant powered by **{reference_source}**.\n"
            "2. Record or upload your attempt.\n"
            f"3. Let **{recognizer_source}** transcribe it.\n"
            "4. Review your score and personalized tips."
        )
        st.divider()
        if backend.mode == "azure":
            st.success("✅ Azure Speech mode enabled (best quality voices & recognition).")
        else:
            st.info(
                "🔓 Azure credentials not detected. Using gTTS for reference audio and the open-source Whisper model for transcription. "
                "Install FFmpeg for best audio quality."
            )
        st.markdown("💡 Tip: Use a quiet room and speak clearly into the microphone.")

    sutra = sutra_lookup[choice]

    if st.session_state.get("active_sutra") != sutra.id:
        st.session_state["active_sutra"] = sutra.id
        st.session_state.pop("reference_audio", None)
        st.session_state.pop("practice_audio", None)
        st.session_state.pop("analysis_result", None)

    st.markdown(f"## {sutra.title}")
    st.markdown("### 📜 Original Script")
    st.code(sutra.script, language="text")

    st.markdown("### 🔤 Transliteration")
    st.write(sutra.transliteration.replace("\n", " · "))

    st.markdown("### 🕯️ Meaning")
    st.write(sutra.translation)

    render_reference_section(sutra, backend)
    st.write("---")

    render_recording_section()
    st.write("---")

    render_analysis_section(sutra, backend)

    st.write("---")
    st.caption(
        "Powered by Streamlit · Azure Speech or gTTS + Whisper · Jain Sutra Pronunciation Tutor prototype."
    )


if __name__ == "__main__":
    main()

