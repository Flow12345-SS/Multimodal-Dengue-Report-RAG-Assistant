"""
Multilingual and Voice Assistant Utilities for Dengue Report RAG Assistant
Supports English, Hindi, Kannada, Telugu, Tamil, Marathi.
"""
import io
import re
import urllib.parse
from typing import Optional, Tuple

SUPPORTED_LANGUAGES = {
    "English": {"code": "en", "bcp47": "en-IN", "name": "English", "native": "English"},
    "Hindi": {"code": "hi", "bcp47": "hi-IN", "name": "Hindi", "native": "हिंदी"},
    "Kannada": {"code": "kn", "bcp47": "kn-IN", "name": "Kannada", "native": "ಕನ್ನಡ"},
    "Telugu": {"code": "te", "bcp47": "te-IN", "name": "Telugu", "native": "తెలుగు"},
    "Tamil": {"code": "ta", "bcp47": "ta-IN", "name": "Tamil", "native": "தமிழ்"},
    "Marathi": {"code": "mr", "bcp47": "mr-IN", "name": "Marathi", "native": "मराठी"},
}

# Cache for translations to prevent repeated API calls
_TRANSLATION_CACHE = {}


def detect_language(text: str) -> str:
    """
    Detects language of input text among English, Hindi, Kannada, Telugu, Tamil, Marathi.
    Uses script inspection first, with fallback to langdetect.
    """
    if not text or not text.strip():
        return "English"

    clean_text = text.strip()

    # Script-based detection
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', clean_text))
    has_kannada = bool(re.search(r'[\u0C80-\u0CFF]', clean_text))
    has_telugu = bool(re.search(r'[\u0C00-\u0C7F]', clean_text))
    has_tamil = bool(re.search(r'[\u0B80-\u0BFF]', clean_text))

    if has_kannada:
        return "Kannada"
    if has_telugu:
        return "Telugu"
    if has_tamil:
        return "Tamil"
    if has_devanagari:
        # Distinguish Marathi from Hindi using common Marathi stopwords
        marathi_markers = ['आहे', 'आहेत', 'झाले', 'काय', 'कसे', 'नाही', 'यांचे', 'त्यांचे', 'प्लेटलेटची', 'रुग्ण']
        if any(marker in clean_text for marker in marathi_markers):
            return "Marathi"
        return "Hindi"

    # Fallback to langdetect if Latin/English or ambiguous
    try:
        from langdetect import detect
        code = detect(clean_text)
        mapping = {
            "en": "English",
            "hi": "Hindi",
            "kn": "Kannada",
            "te": "Telugu",
            "ta": "Tamil",
            "mr": "Marathi",
        }
        return mapping.get(code, "English")
    except Exception:
        return "English"


def clean_markdown_for_speech(text: str) -> str:
    """Cleans markdown symbols, bullets, asterisks for natural voice synthesis."""
    if not text:
        return ""
    cleaned = re.sub(r'[*_#`~\[\]\(\)]', '', text)
    cleaned = re.sub(r'•|\-|\+|>', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def translate_text(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """
    Translates text to the target language (English, Hindi, Kannada, Telugu, Tamil, Marathi).
    """
    if not text or not text.strip():
        return text

    target_info = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES["English"])
    target_code = target_info["code"]

    # If target is English and source is detected as English, return original
    if target_lang == "English":
        detected = detect_language(text)
        if detected == "English":
            return text

    cache_key = (text.strip(), target_code, source_lang)
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    target_mymemory = f"{target_code}-IN" if target_code != "en" else "en-US"
    source_mymemory = "en-US" if source_lang == "en" or source_lang == "auto" else f"{SUPPORTED_LANGUAGES.get(source_lang, {}).get('code', 'en')}-IN"

    translated = None

    # Try MyMemoryTranslator
    try:
        from deep_translator import MyMemoryTranslator
        translator = MyMemoryTranslator(source=source_mymemory, target=target_mymemory)
        # Split text into chunks if too long
        if len(text) > 400:
            parts = [p.strip() for p in text.split('\n') if p.strip()]
            trans_parts = []
            for p in parts:
                trans_parts.append(translator.translate(p))
            translated = "\n".join(trans_parts)
        else:
            translated = translator.translate(text)
    except Exception:
        pass

    # Fallback to GoogleTranslator
    if not translated or "MYMEMORY WARNING" in str(translated):
        try:
            from deep_translator import GoogleTranslator
            gt = GoogleTranslator(source=source_lang if source_lang != 'auto' else 'auto', target=target_code)
            translated = gt.translate(text)
        except Exception:
            pass

    if not translated:
        translated = text

    _TRANSLATION_CACHE[cache_key] = translated
    return translated


def text_to_speech_audio(text: str, lang: str = "English") -> Optional[bytes]:
    """
    Generates MP3 audio bytes using gTTS for the specified language.
    """
    if not text or not text.strip():
        return None

    lang_info = SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES["English"])
    lang_code = lang_info["code"]

    speech_text = clean_markdown_for_speech(text)
    if not speech_text:
        return None

    # Limit speech length for responsiveness
    if len(speech_text) > 1000:
        speech_text = speech_text[:1000] + "..."

    try:
        from gtts import gTTS
        tts = gTTS(text=speech_text, lang=lang_code)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"gTTS error: {e}")
        return None


def transcribe_audio_bytes(audio_bytes: bytes, lang: str = "English") -> Tuple[bool, str]:
    """
    Transcribes audio bytes (e.g. from st.audio_input) using SpeechRecognition.
    """
    if not audio_bytes:
        return False, "No audio provided"

    lang_info = SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES["English"])
    bcp47 = lang_info["bcp47"]

    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()

        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language=bcp47)
        return True, text
    except Exception as e:
        return False, str(e)
