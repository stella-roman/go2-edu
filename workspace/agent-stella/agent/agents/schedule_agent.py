import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

schedule_tool = {
    "type": "function",
    "function": {
        "name": "manage_schedule",
        "description": "Add an event or retrieve today's schedule.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "get_today"],
                },
                "event": {
                    "type": "string",
                    "description": "Description of the event."
                },
                "date": {
                    "type": "string",
                    "description": "Natural language date (e.g., 'today', 'tomorrow', 'next Monday')."
                },
                "time": {
                    "type": "string",
                    "description": "Natural language time (e.g., '7 PM', 'this evening', 'after lunch')."
                }
            },
            "required": ["action"]
        }
    }
}

def create_schedule_agent():
    return client.chat.completions