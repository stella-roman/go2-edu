import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Retrieve the current weather information for a specific city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"],
        }
    }
}

def create_weather_agent():
    return client.chat.completions