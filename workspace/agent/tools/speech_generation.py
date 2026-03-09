import io
import os
import re
import time
import json
import base64

from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from unitree_api.msg import Request

from agent.tools.logger import inference_log

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def preprocess_for_tts(text: str) -> str:
    text = re.sub(r",\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

class Go2Speaker(Node):

    AUDIO_TOPIC = "/api/audiohub/request"

    def __init__(self):

        super().__init__("go2_speaker")

        self.publisher = self.create_publisher(
            Request,
            self.AUDIO_TOPIC,
            10
        )

        self.get_logger().info("Go2 Speaker Ready")

    def speak(self, text: str):

        if not text.strip():
            return

        tts_text = preprocess_for_tts(text)
        self.get_logger().info(f'🔊 TTS: "{tts_text}"')
        
        start_time = time.perf_counter()

        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="shimmer",
            input=tts_text,
            speed=1.3
        )

        mp3_bytes = response.read()

        end_time = time.perf_counter()
        inference_log("TTS", end_time - start_time)

        wav_bytes = self._convert_mp3_to_wav(mp3_bytes)

        with open("agent/tools/temp.wav", "wb") as f:
            f.write(wav_bytes)

        self._send_audio(wav_bytes)

    def _convert_mp3_to_wav(self, mp3_bytes):

        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))

        wav_io = io.BytesIO()

        audio.export(
            wav_io,
            format="wav"
        )

        return wav_io.getvalue()

    def _send_audio(self, wav_bytes):
        duration = len(AudioSegment.from_wav(io.BytesIO(wav_bytes))) / 1000.0
        b64 = base64.b64encode(wav_bytes).decode("utf-8")
        chunk_size = 16000

        chunks = [
            b64[i:i+chunk_size]
            for i in range(0, len(b64), chunk_size)
        ]

        total = len(chunks)

        self.get_logger().info(
            f"Sending audio ({total} chunks)"
        )

        self._send_cmd(4001, "")
        time.sleep(0.1)

        for i, chunk in enumerate(chunks, 1):

            block = {
                "current_block_index": i,
                "total_block_number": total,
                "block_content": chunk
            }

            self._send_cmd(4003, json.dumps(block))

            time.sleep(0.1)

        time.sleep(duration + 1.0)
        self._send_cmd(4002, "")
        self.get_logger().info("Playback Done")

    def _send_cmd(self, api_id, parameter):

        msg = Request()

        msg.header.identity.id = int(time.time() * 1000)
        msg.header.identity.api_id = api_id

        msg.header.lease.id = 0

        msg.header.policy.priority = 0
        msg.header.policy.noreply = False

        msg.parameter = parameter
        msg.binary = []

        self.publisher.publish(msg)

        rclpy.spin_once(self, timeout_sec=0.01)