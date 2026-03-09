import re
import random
import threading
from agent.tools.stt import listen
from datetime import datetime
from agent.tools.logger import save_log
from agent.services.agent_runner import ask_agent
from agent.services.vision_runner import ask_vision
import rclpy
from rclpy.executors import MultiThreadedExecutor
from workspace.agent.tools.speech_generation import Go2Speaker

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

def is_vision_question(text: str) -> bool:
    vision_keywords = [
        "see", "what do you see", "what can you see", "show me",
        "describe the video", "describe the scene", "what is on the screen",
        "video", "frame", "scene"
    ]
    return any(keyword in text.lower() for keyword in vision_keywords)

def main():
    rclpy.init()

    speaker = Go2Speaker()

    executor = MultiThreadedExecutor()
    executor.add_node(speaker)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("\n=== Starting chat with Stella ===")
    print("Type your question below. (Type 'shut down' or 'power off' to quit)")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("agent_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"========== STELLA ACTIVATED ========== {now}\n")

    while True:
        user_input = input("\nYou: ").strip()
        normalized = user_input.lower().strip()

        # Handle shutdown commands (terminate the agent)
        if "power off" in normalized or "shutdown" in normalized or "shut down" in normalized:
            print("=====================================")
            response = random.choice(SHUTDOWN_RESPONSES)
            save_log(question=normalized, answer=response)
            speaker.speak(response)
            break

        # Handle wake-only inputs (respond without executing a command)
        if "stella" in normalized:
            response = random.choice(WAKE_RESPONSES)
            save_log(question=normalized, answer=response)
            speaker.speak(response)
            continue

        # Remove the wake word ("stella") from the prompt before sending it
        cleaned_input = re.sub(
            r"\bstella\b", "", user_input, flags=re.IGNORECASE
        ).strip()

        if not cleaned_input:
            response = "I'm listening, sir."
            save_log(question=normalized, answer=response)
            speaker.speak(response)
            continue

        # Agent Call
        try:
            if is_vision_question(cleaned_input):
                ask_vision(question=cleaned_input)
                # print("STELLA:", answer)
                # print_and_speak(answer)
            else:
                response = ask_agent(cleaned_input)
                save_log(question=normalized, answer=response)
                speaker.speak(response)

        except Exception as e:
            print("Error:", e)

    speaker.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()