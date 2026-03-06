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
from agent.tools.tts import Go2Speaker   # TTS node

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

def ask_vision(question: str, video_path: str, frame_interval: int = 2):
    """
    영상을 분석하여 시간대별 장면을 묘사합니다.
    
    Args:
        question (str): 사용자 질문
        video_path (str): 분석할 영상 파일 경로
        frame_interval (int): 프레임 추출 간격 (초)
        
    Returns:
        str: 시간대별 장면 묘사
    """
    
    try:
        print(f"\n{'='*60}")
        print(f"[Vision Agent] 영상 분석 시작")
        print(f"- 영상 경로: {video_path}")
        print(f"- 프레임 간격: {frame_interval}초")
        print(f"- 질문: {question}")
        print(f"{'='*60}\n")
        
        # 1. 영상에서 프레임 추출
        frames_info = extract_frames(video_path, frame_interval)
        
        if not frames_info:
            return "영상에서 프레임을 추출할 수 없습니다."
        
        # 2. Vision API로 프레임 분석
        description = analyze_frames(frames_info, question)
        
        # 3. 결과 저장 (프레임과 설명 모두 temp 폴더에 보관)
        os.makedirs("temp", exist_ok=True)
        explanation_path = os.path.join("temp", "explanation.txt")
        with open(explanation_path, "w", encoding="utf-8") as f:
            f.write(description or "")
        print(f"\n[Vision Agent] 설명을 저장했습니다: {explanation_path}\n")
        
        return description
        
    except FileNotFoundError as e:
        return f"오류: {str(e)}\n\n영상 파일을 'input/' 폴더에 넣어주세요."
    
    except ValueError as e:
        return f"오류: {str(e)}\n\n영상 파일이 손상되었거나 지원하지 않는 형식입니다."
    
    except Exception as e:
        return f"영상 분석 중 오류가 발생했습니다: {str(e)}"


def ask_vision_interactive(video_path: str, frame_interval: int = 2):
    """
    영상을 분석하고 각 프레임마다 대화형으로 질문을 받습니다.
    
    Args:
        video_path (str): 분석할 영상 파일 경로
        frame_interval (int): 프레임 추출 간격 (초)
    """
    
    try:
        print(f"\n{'='*60}")
        print(f"[Interactive Vision Agent] 영상 분석 시작")
        print(f"- 영상 경로: {video_path}")
        print(f"- 프레임 간격: {frame_interval}초")
        print(f"{'='*60}\n")
        
        # 1. 영상에서 프레임 추출
        frames_info = extract_frames(video_path, frame_interval)
        
        if not frames_info:
            print("영상에서 프레임을 추출할 수 없습니다.")
            return
        
        # 2. 전체 기본 설명 생성
        print("[기본 설명 생성 중...]")
        basic_descriptions = analyze_frames(frames_info, "지금 뭐가 보여?")
        
        # 3. 결과 저장
        os.makedirs("temp", exist_ok=True)
        explanation_path = os.path.join("temp", "explanation.txt")
        with open(explanation_path, "w", encoding="utf-8") as f:
            f.write(basic_descriptions or "")
        print(f"[설명 저장 완료: {explanation_path}]\n")
        
        # 4. 기본 설명을 프레임별로 분리
        descriptions_by_frame = {}
        current_timestamp = None
        current_desc = []
        
        for line in basic_descriptions.split('\n'):
            # [0-2초] 형식의 타임스탬프 찾기
            if line.strip().startswith('[') and '초]' in line:
                if current_timestamp:
                    descriptions_by_frame[current_timestamp] = '\n'.join(current_desc).strip()
                current_timestamp = line.strip()
                current_desc = []
            elif current_timestamp:
                current_desc.append(line)
        
        # 마지막 프레임 저장
        if current_timestamp:
            descriptions_by_frame[current_timestamp] = '\n'.join(current_desc).strip()
        
        # 5. 각 프레임마다 대화형 루프
        print(f"\n{'='*60}")
        print("대화형 모드 시작!")
        print("- 각 프레임에서 질문할 수 있습니다")
        print("- 질문이 없으면 Enter를 눌러 다음 프레임으로")
        print("- 'quit' 입력시 종료")
        print(f"{'='*60}\n")
        
        for frame in frames_info:
            timestamp = f"[{frame['timestamp']}]"
            
            # 기본 설명 출력
            print(f"\n{timestamp}")
            if timestamp in descriptions_by_frame:
                print(descriptions_by_frame[timestamp])
            else:
                print("(기본 설명 없음)")
            
            # 질문 루프
            while True:
                question = input("\nQuestion: ").strip()
                
                # 빈 입력 → 다음 프레임
                if not question:
                    break
                
                # 종료 명령
                if question.lower() in ['quit', 'exit', '종료', '끝']:
                    print("\n대화형 모드를 종료합니다.")
                    return
                
                # 질문에 답변
                from tools.vision_api import analyze_single_frame
                answer = analyze_single_frame(frame, question)
                print(f"Answer: {answer}")
        
        print(f"\n{'='*60}")
        print("모든 프레임 분석 완료!")
        print(f"{'='*60}\n")
        
    except FileNotFoundError as e:
        print(f"오류: {str(e)}\n영상 파일을 'input/' 폴더에 넣어주세요.")
    
    except ValueError as e:
        print(f"오류: {str(e)}\n영상 파일이 손상되었거나 지원하지 않는 형식입니다.")
    
    except Exception as e:
        print(f"영상 분석 중 오류가 발생했습니다: {str(e)}")