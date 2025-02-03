import sys 
import json
import os 
from exeercise import Exercise
from dailyexercise import DailyExercise, DailyExerciseManager


def load_user(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
def load_exercise_keys(filename="exercise.json"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} 파일이 존재하지 않습니다.")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    # JSON 파일의 최상위 구조가 딕셔너리이고, 키가 운동 이름임.
    return list(data.keys())


user_file = "database/user.json"
if not os.path.exists(user_file):
    print("user.json 파일이 존재하지 않습니다. 존재하지 않는 사용자입니다.")
    exit(1)
else:
    user_data = load_user(user_file)

# DailyExerciseManager 객체 생성
daily_manager = DailyExerciseManager()

# 1일차부터 16일차까지 DailyExercise 객체 생성하여 추가
for day in range(1, 17):
    daily = DailyExercise(user_data, day)
    daily_manager.add_daily_exercise(daily)

# DailyExercise 스케줄 출력
print(daily_manager)

# 최종적으로 JSON 파일로 저장
daily_manager.save()

