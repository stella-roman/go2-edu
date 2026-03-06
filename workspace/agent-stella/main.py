import re
import random
import threading

from agent.tools.stt import listen
from datetime import datetime
from agent.tools.logger import print_and_save_log
from agent.services.agent_runner import ask_agent, ask_vision

import rclpy
from rclpy.executors import MultiThreadedExecutor

from agent.tools.tts import Go2Speaker   # TTS node


WAKE_RESPONSES = [
    "Yes, sir?",
    "At your service, sir.",
    "How can I help you, sir?",
    "System’s online, sir."
]

SHUTDOWN_RESPONSES = [
    "Shutting down. Goodbye, sir.",
    "System shutting down. Have a nice day, sir.",
    "Powering off. Until next time, sir."
]

FRAME_INTERVAL_SECONDS = 2
VIDEO_PATH = "input/sample.mp4"


def is_vision_question(text: str) -> bool:
    vision_keywords = [
        "see", "what do you see", "what can you see", "show me",
        "describe the video", "describe the scene", "what is on the screen",
        "video", "frame", "scene"
    ]
    return any(keyword in text.lower() for keyword in vision_keywords)


def main():

    # =========================
    # ROS2 INIT
    # =========================
    rclpy.init()

    speaker = Go2Speaker()

    executor = MultiThreadedExecutor()
    executor.add_node(speaker)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # =========================
    # START
    # =========================
    print("\n=== Starting chat with Stella ===")
    print("Type your question below. (Type 'shut down' or 'power off' to quit)")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("agent_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"========== STELLA ACTIVATED ========== {now}\n")

    while True:

        user_input = input("\nYou: ").strip()
        normalized = user_input.lower().strip()

        # =========================
        # Shutdown
        # =========================
        if "power off" in normalized or "shutdown" in normalized or "shut down" in normalized:

            print("=====================================")

            response = random.choice(SHUTDOWN_RESPONSES)

            print_and_save_log(question=normalized, answer=response)

            speaker.speak(response)

            break

        # =========================
        # Wake word
        # =========================
        if "stella" in normalized:

            response = random.choice(WAKE_RESPONSES)

            print_and_save_log(question=normalized, answer=response)

            speaker.speak(response)

            continue

        cleaned_input = re.sub(
            r"\bstella\b", "", user_input, flags=re.IGNORECASE
        ).strip()

        if not cleaned_input:

            response = "I'm listening, sir."

            print_and_save_log(question=normalized, answer=response)

            speaker.speak(response)

            continue

        # =========================
        # Agent Call
        # =========================
        try:

            if is_vision_question(cleaned_input):

                response = "Analyzing video... please wait."

                print_and_save_log(question=normalized, answer=response)

                speaker.speak(response)

                answer = ask_vision(
                    question=cleaned_input,
                    video_path=VIDEO_PATH,
                    frame_interval=FRAME_INTERVAL_SECONDS
                )

                print("STELLA:", answer)

                speaker.speak(answer)

            else:

                response = ask_agent(cleaned_input)

                print_and_save_log(question=normalized, answer=response)

                speaker.speak(response)

        except Exception as e:
            print("Error:", e)

    # =========================
    # 종료
    # =========================
    speaker.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()