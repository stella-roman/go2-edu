import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path):
    """
    이미지 파일을 base64로 인코딩합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        
    Returns:
        str: base64 인코딩된 이미지 문자열
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_frames(frames_info, question):
    """
    추출된 프레임들을 OpenAI Vision API로 분석합니다.
    
    Args:
        frames_info (List[dict]): extract_frames()의 반환값
        question (str): 사용자 질문
        
    Returns:
        str: 시간별 장면 묘사
    """
    
    if not frames_info:
        return "분석할 프레임이 없습니다."
    
    print(f"[Vision API] {len(frames_info)}개의 프레임 분석 시작...")
    
    # 프롬프트 구성
    system_prompt = """Describe video scenes in Korean in a friendly way.

For each time segment shown, write descriptions in this format:

[0-2초]
지금 우리는 [실내/실외]에 있고, 주변은 [환경 설명].
왼쪽에는 [물체들], 오른쪽에는 [물체들]가 있어.
정면에는 [정면 요소]가 보여.

[2-4초]
...

Style notes:
- Start each segment with its timestamp in brackets
- Skip location description if unchanged from previous
- Mention people or moving objects first
- List multiple objects: "너를 기준으로 [물체1], [물체2]"
- Note camera movement: "앞으로 이동할게"
- When sides are clear: "좌우 모두 특별한 물체는 보이지 않아"

Be conversational and natural."""

    # 사용자 메시지 구성
    user_content = [
        {
            "type": "text",
            "text": f"Describe each scene naturally in Korean.\n\n{question}\n\n"
        }
    ]
    
    # 각 프레임을 이미지로 추가
    for frame in frames_info:
        # 이미지 인코딩
        base64_image = encode_image(frame["image_path"])
        
        # 타임스탬프 텍스트 추가
        user_content.append({
            "type": "text",
            "text": f"\n[{frame['timestamp']}]"
        })
        
        # 이미지 추가
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low"  # 저해상도로 변경 (안전 필터 우회)
            }
        })
    
    # Vision API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=3000,  # 구어체 안내만 생성
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        print(f"[Vision API] 분석 완료")
        
        return result
        
    except Exception as e:
        error_msg = f"Vision API 호출 중 오류가 발생했습니다: {str(e)}"
        print(f"[Error] {error_msg}")
        return error_msg

def analyze_single_frame(frame_info, question):
    """
    단일 프레임에 대한 질문에 답변합니다.
    
    Args:
        frame_info (dict): 프레임 정보 {"timestamp": "0-2초", "image_path": "...", ...}
        question (str): 사용자 질문
        
    Returns:
        str: 질문에 대한 답변
    """
    
    if not frame_info:
        return "분석할 프레임이 없습니다."
    
    # 프롬프트 구성
    system_prompt = """You are answering questions about a video scene in Korean.
Be direct, natural, and conversational.
Focus on answering the specific question asked."""

    # 이미지 인코딩
    base64_image = encode_image(frame_info["image_path"])
    
    # 사용자 메시지 구성
    user_content = [
        {
            "type": "text",
            "text": f"이 장면에 대한 질문: {question}"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low"
            }
        }
    ]
    
    # Vision API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=500,  # 짧은 답변
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        return result
        
    except Exception as e:
        error_msg = f"Vision API 호출 중 오류가 발생했습니다: {str(e)}"
        print(f"[Error] {error_msg}")
        return error_msg