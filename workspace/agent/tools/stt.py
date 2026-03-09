import os
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SAMPLE_RATE = 16000
CHANNELS = 1

def listen():
    audio = sd.rec(
        int(60 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )

    input()
    sd.stop()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, audio)
        wav_path = f.name

    with open(wav_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-mini-transcribe",
            language="en",
            prompt="Always output the transcription in English."
        )

    os.remove(wav_path)

    return transcript.text