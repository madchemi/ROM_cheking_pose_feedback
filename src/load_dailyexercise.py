import json
import os

def get_exercise_details(filename="exercise.json"):
    """
    exercise.json 파일에서 각 운동의 임계값(target_range)과 피드백 메시지(feedback_messages)를 추출하여 반환합니다.
    
    반환 형식:
        {
            "butterfly": {
                "target_range": { ... },
                "feedback_messages": { ... }
            },
            "sit_to_stand": { ... },
            "walking": { ... },
            ...
        }
    
    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우.
        Exception: JSON 로드 중 발생하는 기타 예외.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} 파일이 존재하지 않습니다.")
    
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 최상위 구조가 딕셔너리라고 가정하고, 각 운동의 target_range와 feedback_messages 추출
    details = {}
    for exercise_name, info in data.items():
        target_range = info.get("target_range", {})
        feedback_messages = info.get("feedback_messages", {})
        details[exercise_name] = {
            "target_range": target_range,
            "feedback_messages": feedback_messages
        }
    return details

# 사용 예시:
if __name__ == "__main__":
    try:
        exercise_details = get_exercise_details("exercise.json")
        print("각 운동의 임계값과 피드백 메시지:")
        for ex, info in exercise_details.items():
            print(f"- {ex}:")
            print(f"    Target Range: {info['target_range']}")
            print(f"    Feedback Messages: {info['feedback_messages']}")
    except Exception as e:
        print(f"오류 발생: {e}")
