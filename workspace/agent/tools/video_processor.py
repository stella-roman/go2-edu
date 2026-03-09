import cv2
import os

def extract_frames(video_path, interval_seconds=2):
    """
    영상에서 N초마다 프레임을 추출합니다.
    
    Args:
        video_path (str): 영상 파일 경로
        interval_seconds (int): 프레임 추출 간격 (초)
        
    Returns:
        List[dict]: [
            {"timestamp": "0-2초", "image_path": "temp/frame_0.jpg", "frame_index": 0},
            {"timestamp": "2-4초", "image_path": "temp/frame_1.jpg", "frame_index": 1},
            ...
        ]
        
    Raises:
        FileNotFoundError: 영상 파일이 없는 경우
        ValueError: 영상 파일이 손상된 경우
    """
    
    # 영상 파일 존재 확인
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    
    # 영상 로드
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식입니다: {video_path}")
    
    # 영상 정보 추출
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"[Video Info] FPS: {fps:.2f}, 총 프레임: {total_frames}, 길이: {duration:.2f}초")
    
    # temp 폴더 확인 및 생성
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 프레임 추출
    frames_info = []
    frame_index = 0
    current_second = 0
    
    while current_second < duration:
        # 해당 시간의 프레임 번호 계산
        frame_number = int(current_second * fps)
        
        # 프레임 위치 설정
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # 프레임 읽기
        success, frame = video.read()
        
        if success:
            # 이미지 파일로 저장
            image_path = os.path.join(temp_dir, f"frame_{frame_index}.jpg")
            cv2.imwrite(image_path, frame)
            
            # 타임스탬프 계산
            start_time = current_second
            end_time = min(current_second + interval_seconds, duration)
            timestamp = f"{start_time:.0f}-{end_time:.0f}초"
            
            frames_info.append({
                "timestamp": timestamp,
                "image_path": image_path,
                "frame_index": frame_index
            })
            
            print(f"[Frame Extracted] {timestamp} → {image_path}")
            frame_index += 1
        
        # 다음 추출 시간으로 이동
        current_second += interval_seconds
    
    video.release()
    
    print(f"[Complete] 총 {len(frames_info)}개의 프레임 추출 완료")
    
    return frames_info

def cleanup_temp_frames():
    """
    temp 폴더의 모든 프레임 이미지를 삭제합니다.
    """
    temp_dir = "temp"
    
    if not os.path.exists(temp_dir):
        return
    
    # temp 폴더 내 모든 jpg 파일 삭제
    for filename in os.listdir(temp_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            file_path = os.path.join(temp_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[Warning] 파일 삭제 실패: {file_path} - {e}")