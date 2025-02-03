import os
import sys

# 현재 파일의 디렉토리와 상위 디렉토리를 구합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
# 상위 디렉토리를 sys.path에 추가하여, 'user' 폴더를 찾을 수 있도록 합니다.
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 이제 'user' 폴더 아래의 user.py 모듈에서 User 클래스를 import할 수 있습니다.
from angle_chek import Calibrator
from user.user import User

def main():
    # ROM 데이터를 측정합니다.
    calibrator = Calibrator(calibration_frames=30, rom_duration=5)
    rom_data = calibrator.run_calibration()
    calibrator.release()
    if rom_data is None:
        print("캘리브레이션 실패")
        return

    # 측정된 ROM 데이터를 출력
    print("측정된 ROM 데이터:")
    for joint, data in rom_data.items():
        print(f"{joint}: {data}")

    # User 객체 생성 (user_id, password는 예시입니다)
    user = User(user_id="user123", password="pass123", rom_data=rom_data)
    print("생성된 사용자 정보:")
    print(user)

    # 저장할 디렉토리가 없으면 생성 (예: "./database/")
    output_dir = "./database"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "user.json")

    # User 정보를 JSON 파일로 저장
    user.save_to_json(output_filename)

if __name__ == "__main__":
    main()
