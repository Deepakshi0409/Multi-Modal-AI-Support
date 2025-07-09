import speech_recognition as sr
from pydub import AudioSegment
from app.text_handler import classify_text
import os

def transcribe_and_classify(audio_path):
    recognizer = sr.Recognizer()
    
    # Convert MP3 to WAV if needed
    if audio_path.endswith(".mp3"):
        wav_path = audio_path.replace(".mp3", ".wav")
        sound = AudioSegment.from_mp3(audio_path)
        sound.export(wav_path, format="wav")
        audio_path = wav_path

    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

            if not text.strip():
                return {"error": "No recognizable speech found in audio."}

            result = classify_text(text)
            return {
                "transcribed_text": text,
                "classification": result
            }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(audio_path) and audio_path.endswith(".wav"):
            os.remove(audio_path)  # Clean up temporary wav files
