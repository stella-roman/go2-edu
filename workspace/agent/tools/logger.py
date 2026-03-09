from datetime import datetime
from typing import Optional

# without using llm/vlm models
def save_log(
    question: str,
    answer: str,
    filename: Optional[str] = "agent_logs.txt",
):
    # --- log to agent_logs.txt ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{now}]\n")
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n")

# when using llm/vlm models
def inference_log(
    tag: str,
    inference_time: float,
    total_tokens: Optional[int] = 0,
    filename: str = "agent_logs.txt"
):
    with open(filename, "a", encoding="utf-8") as f:
        if total_tokens == 0:
            f.write(f"([{tag}] Inference took {inference_time:.3f} seconds.)\n\n") # when using tts models (they don't provide token count)
        else:
            f.write(f"([{tag}] Inference took {inference_time:.3f} seconds, using {total_tokens} tokens.)\n")
        