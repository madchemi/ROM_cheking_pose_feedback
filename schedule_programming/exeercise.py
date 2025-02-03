import json

# 사용자 기본 ROM 데이터 (참고용)
USER_ROM_DATA = {
    "hip_abduction": 50,
    "hip_external_rotation": 48,
    "knee_flexion": 130,
    "hip_flexion": 120,
    "ankle_dorsiflexion": 20
}

FRAME_INTERVAL = 30  # 피드백을 제공할 프레임 간격

class Exercise:
    """
    운동 정보를 저장하는 클래스.
    
    Attributes:
      - name: 운동 이름 (예: 'butterfly')
      - target_range: 각 관절별로 권장되는 ROM 범위 (예: {"hip_abduction": (30, 45), ...})
      - feedback_messages: 관절별 피드백 메시지 (예: {"too_tight": "{joint} is too tight. ...", ...})
      - user_rom_data: 사용자의 기본 ROM 데이터 (참고용)
    """
    def __init__(self, name, target_range, feedback_messages, user_rom_data):
        self.name = name
        self.target_range = target_range
        self.feedback_messages = feedback_messages
        self.user_rom_data = user_rom_data

    def to_dict(self):
        return {
            "exercise": self.name,
            "target_range": self.target_range,
            "feedback_messages": self.feedback_messages,
            "user_rom_data": self.user_rom_data
        }

    def __str__(self):
        return (f"{self.name}:\n"
                f"  Target ROM Range: {self.target_range}\n"
                f"  Feedback Messages: {self.feedback_messages}\n"
                f"  User ROM Data: {self.user_rom_data}")

# 기본 운동 목록 생성 (각 운동에 대해 권장 ROM 범위 및 피드백 메시지를 설정)
STRETCHES = {
    "butterfly": Exercise(
        "butterfly",
        {"hip_abduction": (30, 45), "hip_external_rotation": (30, 45)},
        {
            "too_tight": "{joint} is too tight. Try to open your legs wider.",
            "overstretched": "{joint} is overstretched. Avoid pushing too hard."
        },
        USER_ROM_DATA
    ),
    "sit_to_stand": Exercise(
        "sit_to_stand",
        {"hip_flexion": (90, 120), "knee_flexion": (90, 130)},
        {
            "too_tight": "{joint} is too tight. Try to bend more.",
            "overstretched": "{joint} is overstretched. Be careful not to force it."
        },
        USER_ROM_DATA
    ),
    "walking": Exercise(
        "walking",
        {"hip_flexion": (20, 45), "knee_flexion": (10, 70), "ankle_dorsiflexion": (10, 20)},
        {
            "too_tight": "{joint} is too tight. Try to relax your movement.",
            "overstretched": "{joint} is overstretched. Maintain a comfortable gait."
        },
        USER_ROM_DATA
    )
}

# 예시: STRETCHES에 저장된 운동들 출력
for key, exercise in STRETCHES.items():
    print(f"--- {key} ---")
    print(exercise)
    print()

# 전체 운동 목록을 JSON 파일로 저장하는 함수 예시
def save_exercises_to_json(exercises, filename):
    # exercises: Exercise 객체들을 저장한 dict
    data = {name: ex.to_dict() for name, ex in exercises.items()}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"운동 목록이 '{filename}' 파일에 저장되었습니다.")

# JSON 파일로 저장 (예: "stretches.json")
save_exercises_to_json(STRETCHES, "exercise.json")
