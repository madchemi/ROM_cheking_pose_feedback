import cv2
import numpy as np
import time
import json
import os
import threading
import queue
import mediapipe as mp
import pyttsx3

# ---------------------------------------------
# 1. TTS를 위한 글로벌 자원
# ---------------------------------------------
tts_queue = queue.Queue()      # 피드백 문자열 큐
engine = pyttsx3.init()
engine.setProperty('rate', 500)  # 말하기 속도 (기본 약 200 정도)

def tts_worker():
    """큐에서 문자열을 꺼내 TTS를 실행하는 쓰레드 함수."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# ---------------------------------------------
# 2. Exercise.json 에서 운동 정보 불러오기
# ---------------------------------------------
def get_exercise_details(filename="exercise.json"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} 파일이 존재하지 않습니다.")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ---------------------------------------------
# 3. 사용자 정보를 로드하는 함수
# ---------------------------------------------
def load_user(filename="database/user.json"):
    if not os.path.exists(filename):
        raise FileNotFoundError("user.json 파일이 존재하지 않습니다. 존재하지 않는 사용자입니다.")
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------
# 4. 3D 좌표를 사용한 각도 계산 함수
# ---------------------------------------------
def calculate_angle_3d(p1, p2, p3):
    """
    p1, p2, p3: (x, y, z) 좌표.
    반환값은 float 타입의 각도 (도)입니다.
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return float(angle)

# ---------------------------------------------
# 5. Mediapipe Pose 및 유틸리티 설정
# ---------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------
# 6. 운동별 키 포인트 및 관심 관절 매핑
# ---------------------------------------------
exercise_joint_map = {
    "sit_to_stand": {
        "joint": "hip_flexion",
        "landmarks": [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                      mp_pose.PoseLandmark.LEFT_HIP.value,
                      mp_pose.PoseLandmark.LEFT_KNEE.value]
    },
    "butterfly": {
        "joint": "hip_abduction",
        "landmarks": [mp_pose.PoseLandmark.LEFT_HIP.value,
                      mp_pose.PoseLandmark.LEFT_KNEE.value,
                      mp_pose.PoseLandmark.LEFT_ANKLE.value]
    },
    "walking": {
        "joint": "knee_flexion",
        "landmarks": [mp_pose.PoseLandmark.LEFT_HIP.value,
                      mp_pose.PoseLandmark.LEFT_KNEE.value,
                      mp_pose.PoseLandmark.LEFT_ANKLE.value]
    }
}

# ---------------------------------------------
# 7. 초기 자세를 캘리브레이션하고 ROM 데이터를 비교하여 동작 카운트하는 CalibratedCounter 클래스
# ---------------------------------------------
class CalibratedCounter:
    def __init__(self, init_angle, target_angle, tolerance=5, threshold_ratio=0.15):
        """
        init_angle: 사용자의 초기 자세(예: 곧게 선 상태)에서 측정한 각도
        target_angle: 사용자의 ROM 데이터(예: 동작 시 최대/최소 각도)
        tolerance: 초기 자세 복귀를 판단할 때 허용 오차 (도)
        threshold_ratio: 초기 자세와 target_angle 차이의 몇 %만큼 움직여야 동작으로 판단할지 결정 (예: 0.15 = 15%)
        """
        self.init_angle = init_angle
        self.target_angle = target_angle
        self.tolerance = tolerance
        # 임계값 계산: 초기 각도와 target_angle 사이의 차이의 일정 비율만큼 변화하면 동작으로 판단
        self.threshold = init_angle - (init_angle - target_angle) * threshold_ratio
        self.state = "at_init"  # 초기 상태
        self.rep_count = 0

    def update(self, current_angle):
        # 만약 current_angle이 NumPy 배열이라면 단일 스칼라 값으로 변환
        if isinstance(current_angle, np.ndarray):
            # 요소가 하나만 있다면 .item() 사용
            if current_angle.size == 1:
                current_angle = current_angle.item()
            else:
                # 여러 요소가 있다면 첫번째 요소 사용 (또는 적절한 값 선택)
                current_angle = float(current_angle[0])
        else:
            current_angle = float(current_angle)

        # 만약 현재 각도가 초기 자세보다 더 커지면(예: 과도하게 펴진 경우) 초기값으로 보정
        if current_angle > self.init_angle:
            current_angle = self.init_angle

        # 초기 상태("at_init")에서 현재 각도가 임계값보다 작아지면 상태 전환
        if self.state == "at_init":
            if current_angle < self.threshold:
                self.state = "down"
        # 동작 상태("down")에서 초기 자세(init_angle) 근처(허용 오차 내)로 복귀하면 1회 동작으로 카운트
        elif self.state == "down":
            if abs(current_angle - self.init_angle) < self.tolerance:
                self.rep_count += 1
                print(f"Rep {self.rep_count} completed. (Init_angle: {self.init_angle:.2f}, Threshold: {self.threshold:.2f})")
                tts_queue.put(f"Rep {self.rep_count} completed.")
                self.state = "at_init"
        return self.rep_count

# ---------------------------------------------
# 8. 메인 함수: 캘리브레이션, 웹캠, Mediapipe Pose 및 동작 감지
# ---------------------------------------------
def main():
    # 사용자 및 운동 정보 로드 (필요 시)
    try:
        user_data = load_user("database/user.json")
    except FileNotFoundError as e:
        print(e)
        return

    try:
        exercise_details = get_exercise_details("exercise.json")
    except FileNotFoundError as e:
        print(e)
        return

    # 예시로 "sit_to_stand" 운동의 "hip_flexion"을 사용한다고 가정
    exercise_order = list(exercise_details.keys())
    current_exercise_index = 0
    current_exercise_name = exercise_order[current_exercise_index]
    current_exercise_info = exercise_details[current_exercise_name]
    joint_info = exercise_joint_map.get(current_exercise_name, {
        "joint": "hip_flexion",
        "landmarks": [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                      mp_pose.PoseLandmark.LEFT_HIP.value,
                      mp_pose.PoseLandmark.LEFT_KNEE.value]
    })
    joint_name = joint_info["joint"]

    # ROM 데이터는 exercise_details 내 저장된 값 또는 별도 측정값을 사용할 수 있음
    # 예시에서는 ROM 데이터가 {"hip_flexion": 120}이라고 가정하여 target_angle = 120
    target_angle = current_exercise_info["target_range"].get(joint_name, 120)

    # 웹캠 및 Mediapipe 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # ---------------------------------------------
    # 8-1. 초기 캘리브레이션: 몇 프레임 동안 초기 자세에서 각도를 평균내어 init_angle 계산
    # ---------------------------------------------
    calibration_frames = 30
    calibration_angles = []
    calibrated = False
    init_angle = None

    print("초기 캘리브레이션 중... (바르게 서주세요)")
    while not calibrated:
        ret, frame = cap.read()
        if not ret:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            lm_indices = joint_info["landmarks"]
            pts = []
            for idx in lm_indices:
                lm = landmarks[idx]
                pts.append((lm.x, lm.y, lm.z))
            angle = calculate_angle_3d(pts[0], pts[1], pts[2])
            calibration_angles.append(angle)
            cv2.putText(frame, "Calibrating...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if len(calibration_angles) >= calibration_frames:
                init_angle = np.mean(calibration_angles)
                calibrated = True

    cv2.destroyWindow("Calibration")
    print(f"캘리브레이션 완료! 초기각도: {init_angle:.2f}")

    # CalibratedCounter를 초기각도와 ROM 데이터(target_angle)를 이용해 생성
    counter = CalibratedCounter(init_angle=init_angle, target_angle=target_angle, tolerance=5, threshold_ratio=0.15)

    reps_per_set = 10  # 세트당 반복 횟수
    set_count = 0
    exercise_start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        current_angle = None
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            lm_indices = joint_info["landmarks"]
            pts = []
            for idx in lm_indices:
                lm = landmarks[idx]
                pts.append((lm.x, lm.y, lm.z))
            current_angle = calculate_angle_3d(pts[0], pts[1], pts[2])
            rep_count = counter.update(current_angle)
        else:
            rep_count = counter.rep_count

        elapsed_time = time.time() - exercise_start_time
        # 텍스트 출력: 운동명은 파란색, reps와 set은 검은색, 시간은 노란색
        cv2.putText(frame, f"Exercise: {current_exercise_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Reps: {counter.rep_count}/{reps_per_set}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Set: {set_count+1}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Time: {int(elapsed_time)} sec", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Rehabilitation Exercise", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 세트 완료 처리: 정해진 rep 수가 달성되면
        if counter.rep_count >= reps_per_set:
            set_count += 1
            duration = time.time() - exercise_start_time
            print(f"{current_exercise_name} 세트 {set_count} 완료, 소요시간: {int(duration)}초")
            tts_queue.put(f"{current_exercise_name} set {set_count} complete. Duration {int(duration)} seconds.")
            # 다음 세트를 위해 카운터 초기화 (상태 및 rep 카운트를 리셋)
            counter.state = "at_init"
            counter.rep_count = 0
            exercise_start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None)
    tts_thread.join()
    pose.close()

if __name__ == "__main__":
    main()
