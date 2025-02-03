import json
import os

# 사용자 정보를 로드하는 함수
def load_user(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

# exercise.json 파일에서 운동 목록(키 목록)을 로드하는 함수 (DailyExercise 클래스 내에서 static 메서드로 사용)
class DailyExercise:
    """
    DailyExercise 클래스는 한 사용자의 하루 운동 스케줄을 저장합니다.
    
    Attributes:
      - user: 사용자 정보 (딕셔너리, 예: user.json에서 로드된 데이터)
      - day: 해당 일차 (예: 1일차, 2일차, …)
      - exercises: 운동 스케줄 정보. 각 운동은 운동 이름을 key로 하고, 값은 딕셔너리로 
          { "sets": 세트수, "reps_per_set": 한 세트당 반복횟수 } 형태로 구성됩니다.
          기본값은 3세트, 1세트당 10회로 설정됨.
      - user_rom_data: user.json 파일에서 자동으로 추출한 ROM 데이터
    """
    
    @staticmethod
    def load_exercise_keys(filename="exercise.json"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} 파일이 존재하지 않습니다.")
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 최상위 구조가 딕셔너리이고, 그 키들이 운동 이름임.
        return list(data.keys())

    def __init__(self, user, day, exercises=None, exercises_file="exercise.json"):
        self.user = user
        self.day = day
        if exercises is None:
            try:
                exercise_keys = DailyExercise.load_exercise_keys(exercises_file)
            except FileNotFoundError as e:
                print(e)
                exercise_keys = []  # 파일이 없으면 빈 목록으로 설정
            self.exercises = {name: {"sets": 3, "reps_per_set": 10} for name in exercise_keys}
        else:
            self.exercises = exercises
        # user.json에서 로드한 사용자 정보에서 "rom_data" 추출
        self.user_rom_data = self.user.get("rom_data", {})

    def to_dict(self):
        return {
            "user": self.user,
            "day": self.day,
            "exercises": self.exercises,
            "user_rom_data": self.user_rom_data
        }

    def __str__(self):
        exercise_str = "\n".join([f"  - {ex}: {details['sets']}세트, 1세트당 {details['reps_per_set']}회" 
                                   for ex, details in self.exercises.items()])
        return f"Day {self.day} for user {self.user.get('name', 'Unknown')}:\n{exercise_str}"

class DailyExerciseManager:
    def __init__(self, filename="daily_exercises.json", exercises_file="exercise.json"):
        self.filename = filename
        self.exercises_file = exercises_file
        self.daily_exercises = []
        if os.path.exists(self.filename):
            with open(self.filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 로드된 데이터가 리스트가 아니라면 빈 리스트로 초기화
            if isinstance(data, list):
                self.daily_exercises = data
            else:
                self.daily_exercises = []
        # 기존에 저장된 각 DailyExercise 데이터의 'exercises' 필드가 비어있다면 업데이트합니다.
        self.update_missing_exercises()

    def update_missing_exercises(self):
        """
        각 DailyExercise 항목의 'exercises' 필드가 빈 딕셔너리인 경우,
        exercises_file에서 기본 운동 목록(키 목록)을 불러와 기본값으로 업데이트합니다.
        """
        try:
            exercise_keys = DailyExercise.load_exercise_keys(self.exercises_file)
        except FileNotFoundError as e:
            print(e)
            exercise_keys = []
        for daily in self.daily_exercises:
            if not daily.get("exercises"):
                daily["exercises"] = {name: {"sets": 3, "reps_per_set": 10} for name in exercise_keys}

    def add_daily_exercise(self, daily_exercise):
        self.daily_exercises.append(daily_exercise.to_dict())
        print(f"Day {daily_exercise.day} 스케줄이 추가되었습니다.")

    def save(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.daily_exercises, f, ensure_ascii=False, indent=4)
        print(f"전체 DailyExercise 스케줄이 '{self.filename}'에 저장되었습니다.")

    def __str__(self):
        output = "Daily Exercise 스케줄:\n"
        for day_data in self.daily_exercises:
            output += f"Day {day_data.get('day')} - User: {day_data.get('user', {}).get('name', 'Unknown')}\n"
            for ex, details in day_data.get("exercises", {}).items():
                output += f"  - {ex}: {details['sets']}세트, {details['reps_per_set']}회\n"
        return output