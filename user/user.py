# user.py
import json

class User:
    """
    User 클래스는 사용자 정보를 저장합니다.
    
    Attributes:
      - user_id: 사용자 아이디
      - password: 사용자 비밀번호
      - rom_data: 각 관절의 ROM 값을 저장하는 딕셔너리 (예: {"hip_abduction": {"init_angle": 90, "extreme_angle": 120, "rom": 30}, ...})
    """
    def __init__(self, user_id, password, rom_data):
        self.user_id = user_id
        self.password = password  # 실제 서비스에서는 비밀번호 저장 시 암호화 등 보안 처리 필요
        self.rom_data = rom_data

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "password": self.password,
            "rom_data": self.rom_data
        }

    def save_to_json(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        print(f"User 정보가 '{filename}' 파일에 저장되었습니다.")

    def __str__(self):
        return f"User ID: {self.user_id}, ROM Data: {self.rom_data}"
