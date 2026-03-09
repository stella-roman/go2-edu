import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

vision_tool = {
    "type": "function",
    "function": {
        "name": "analyze_video",
        "description": "영상의 장면을 시간대별로 묘사합니다. 시각장애인을 위한 상세한 장면 설명을 제공합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "분석할 영상 파일의 경로"
                },
                "frame_interval": {
                    "type": "number",
                    "description": "프레임 추출 간격 (초 단위)"
                }
            },
            "required": ["video_path"],
        }
    }
}

def create_vision_agent():
    """Vision 에이전트 클라이언트를 반환합니다."""
    return client.chat.completions