import json
import os

def get_daily_exercise_info(target_day, filename="daily_exercises.json"):
    """
    daily_exercises.json 파일에서 특정 일차(target_day)의 데이터를 찾아서,
    (1) 사용자 정보와 ROM 정보가 포함된 딕셔너리, 
    (2) 운동 스케줄 정보가 포함된 딕셔너리로 반환합니다.
    
    Parameters:
        target_day (int): 추출할 일차 (예: 1, 2, ...).
        filename (str): daily_exercises.json 파일의 경로.
        
    Returns:
        tuple: (user_info, exercises_info)
            - user_info: {"user": 사용자 정보, "user_rom_data": ROM 데이터}
            - exercises_info: 운동 스케줄 정보 (예: {"butterfly": {"sets": 3, "reps_per_set": 10}, ...})
    
    Raises:
        FileNotFoundError: 지정된 파일이 존재하지 않을 경우.
        ValueError: target_day에 해당하는 데이터가 없는 경우.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} 파일이 존재하지 않습니다.")
    
    with open(filename, "r", encoding="utf-8") as f:
        daily_data = json.load(f)
    
    # daily_data는 DailyExercise 객체들이 to_dict()로 저장된 리스트로 가정합니다.
    for entry in daily_data:
        if entry.get("day") == target_day:
            user_info = {
                "user": entry.get("user", {}),
                "user_rom_data": entry.get("user_rom_data", {})
            }
            exercises_info = entry.get("exercises", {})
            return user_info, exercises_info
    
    raise ValueError(f"Day {target_day} 데이터가 존재하지 않습니다.")
