import os
import tempfile
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SAMPLE_RATE = 16000

def listen():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    print("Listening... Press Enter to stop.")

    # 🔥 arecord 시작 (백그라운드 실행)
    record_process = subprocess.Popen([
        "arecord",
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        wav_path
    ])

    input()  # Enter 누르면 종료
    record_process.terminate()
    record_process.wait()

    # 🔥 OpenAI STT 호출
    with open(wav_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-mini-transcribe",
            language="en"
        )

    os.remove(wav_path)

    return transcript.text