import os
import json
import time
import requests
from agent.agents.weather_agent import client, weather_tool
from agent.agents.schedule_agent import schedule_tool
from agent.tools.weather_api import get_weather
from agent.tools.schedule_api import manage_schedule
from agent.tools.video_processor import extract_frames
from agent.tools.vision_api import analyze_frames
from agent.tools.logger import print_and_save_log, inference_log
from workspace.agent.tools.speech_generation import Go2Speaker

SYSTEM_PROMPT = """
You are an AI assistant that helps with everyday tasks.
Always respond in natural English.

You have access to the following tools:
- Weather tool: Provides current weather information.
- Schedule tool: Adds or retrieves schedules.

Tool usage rules (VERY IMPORTANT):
1. If the user asks about the weather, temperature, forecast, rain, snow,
   or general weather conditions, you MUST call the Weather tool.
2. If the user asks about the weather but does NOT specify a city,
   you MUST still call the Weather tool with city set to null.
   Do NOT guess or assume a city yourself.
3. Do NOT answer weather-related questions directly without calling the Weather tool.

Schedule rules:
4. Call the Schedule tool ONLY when the user wants to add a schedule
   or check schedules (e.g., today’s schedule).

General rules:
5. If the user's request is not about weather or scheduling,
   answer normally without calling any tool.
6. Do NOT call any tool unless the user intent clearly matches a tool.

Time expressions:
If the user refers to a date or time using natural expressions such as
"today", "tomorrow", "this evening", "at 7 PM", etc.,
do NOT ask for clarification unless the expression is truly ambiguous.
Pass the expression exactly as the user said it to the schedule tool.
The backend will normalize it.

"""

def get_tool_prompt(tool_name):
    TOOL_SYSTEM_PROMPTS = {
    "get_weather": """
        You are a concise weather assistant.

        From the provided weather data, respond using ONLY:
        - Current temperature
        - Feels-like temperature
        - Weather condition (e.g., sunny, cloudy, rainy)

        If the weather condition indicates rain (rain, drizzle, thunderstorm, shower, etc.),
        tell the user to bring an umbrella.

        Do NOT mention:
        - Humidity
        - Wind speed
        - Pressure
        - Any other extra weather metrics

        Keep the response natural, short, and suitable for voice output.
    """,

    "manage_schedule": """
        You are a personal scheduling assistant.
        Confirm actions clearly.
        Be structured and precise.
    """
    }

    return TOOL_SYSTEM_PROMPTS.get(
        tool_name,
        SYSTEM_PROMPT
    )

def execute_tool(tool_name, args):
    if tool_name == "get_weather":
        city = args.get("city") or get_city()
        print(f"(current location: {city})")
        return get_weather(city)

    elif tool_name == "manage_schedule":
        return manage_schedule(**args)

    else:
        raise ValueError(f"Unknown tool: {tool_name}")

def get_city():
    res = requests.get("https://ipinfo.io/json", timeout=3)
    res.raise_for_status()
    data = res.json()
    return data.get("city")

def ask_agent(question: str):
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        tools=[weather_tool, schedule_tool],
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        tool_result = execute_tool(tool_name, args)
        tool_system_prompt = get_tool_prompt(tool_name)
        start_time = time.perf_counter()

        final_response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": tool_system_prompt},
                {"role": "user", "content": question},
                msg,
                {
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                    "tool_call_id": tool_call.id
                },
            ]
        )

        end_time = time.perf_counter()
        inference_time = end_time - start_time

        answer = final_response.choices[0].message.content
        total_tokens = final_response.usage.total_tokens

        print_and_save_log(question=question, answer=answer)
        inference_log("LLM", inference_time, total_tokens)
        Go2Speaker(answer)

        return answer

    return msg.content