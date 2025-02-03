import cv2
import numpy as np
import time
import pyttsx3
import threading
import queue
import mediapipe as mp

# ---------------------------------------------
# 1. TTS를 위한 글로벌 자원 - 속도(rate) 설정 추가
# ---------------------------------------------
tts_queue = queue.Queue()      # 음성으로 출력할 문자열을 담는 큐
engine = pyttsx3.init()        # TTS 엔진 초기화
engine.setProperty('rate', 500)  # 말하기 속도 증가 (기본값 약 200 전후)

def tts_worker():
    """
    큐에서 문자열을 꺼내어 TTS를 재생하는 쓰레드 함수.
    None이 들어오면 종료.
    """
    while True:
        feedback_text = tts_queue.get()
        if feedback_text is None:
            break
        engine.say(feedback_text)
        engine.runAndWait()

# ---------------------------------------------
# 2. Stretch 클래스 (스트레칭 자세 정보)
# ---------------------------------------------
class Stretch:
    """스트레칭 자세 정보를 담는 클래스"""
    def __init__(self, name, thresholds, feedback_messages, user_data=None):
        self.name = name
        self.default_thresholds = thresholds
        self.feedback_messages = feedback_messages
        self.user_thresholds = self.adjust_thresholds(user_data) if user_data else thresholds

    def adjust_thresholds(self, user_data):
        """사용자 데이터를 기반으로 임계값을 조정 (여기서는 기본값 그대로 사용)"""
        adjusted_thresholds = {}
        for joint, (default_min, default_max) in self.default_thresholds.items():
            if joint in user_data:
                user_max = user_data[joint]
                min_val = user_max * 0.85  
                max_val = user_max
                adjusted_thresholds[joint] = (min_val, max_val)
            else:
                adjusted_thresholds[joint] = (default_min, default_max)
        return adjusted_thresholds

    def provide_feedback(self, angles):
        """각도가 정상 범위를 벗어나면 피드백 문자열 생성"""
        feedback = []
        for joint, angle in angles.items():
            if angle is not None and joint in self.user_thresholds:
                min_val, max_val = self.user_thresholds[joint]
                if angle < min_val:
                    feedback.append(self.feedback_messages["too_tight"].format(joint=joint.replace('_', ' ').title()))
                elif angle > max_val:
                    feedback.append(self.feedback_messages["overstretched"].format(joint=joint.replace('_', ' ').title()))
        return feedback

# ---------------------------------------------
# 3. 사용자별 ROM 데이터 (평균치 예제)
# ---------------------------------------------
USER_ROM_DATA = {
    "hip_flexion": 90,       # 앉은 상태: 약 90° (평균치)
    "knee_flexion": 135,     # 무릎 굴곡: 약 135°
    # 다른 값은 미사용
}

FRAME_INTERVAL = 4  # 피드백 및 각도 계산 처리 프레임 간격

# 스트레칭 동작 설정 (sit_to_stand: 앉았다 일어나는 동작)
# 여기서는 hip_flexion을 기준으로, 앉은 상태: > 100°, 일어선 상태: < 60°로 설정
STRETCHES = {
    "sit_to_stand": Stretch(
        "sit_to_stand",
        {"hip_flexion": (60, 100), "knee_flexion": (30, 135)},
        {"too_tight": "{joint} is too tight. Try to bend more.",
         "overstretched": "{joint} is overstretched. Be careful not to force it."},
        USER_ROM_DATA
    ),
    "butterfly": Stretch(
        "butterfly",
        {"hip_abduction": (30, 45), "hip_external_rotation": (30, 45)},
        {"too_tight": "{joint} is too tight. Try to open your legs wider.",
         "overstretched": "{joint} is overstretched. Avoid pushing too hard."},
        USER_ROM_DATA
    ),
    "walking": Stretch(
        "walking",
        {"hip_flexion": (20, 45), "knee_flexion": (10, 70), "ankle_dorsiflexion": (10, 20)},
        {"too_tight": "{joint} is too tight. Try to relax your movement.",
         "overstretched": "{joint} is overstretched. Maintain a comfortable gait."},
        USER_ROM_DATA
    )
}


# ---------------------------------------------
# 4. Mediapipe Pose 설정 및 유틸리티 함수
# ---------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 3차원 좌표를 사용한 각도 계산 (p1, p2, p3는 각각 (x, y, z) 좌표)
def calculate_angle_3d(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return None
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_frame_mediapipe(image, pose, stretch, frame_count):
    """FRAME_INTERVAL마다 Mediapipe Pose를 통해 keypoints 추출, 각도 계산, 피드백 생성"""
    if frame_count % FRAME_INTERVAL != 0:
        return None, image

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None, image

    # 추출된 33개 랜드마크 중 필요한 좌표(왼쪽 기준)
    landmarks = results.pose_landmarks.landmark

    # 좌표 추출 (x, y, z) (z는 normalized 값)
    # hip_flexion: 왼쪽 어깨(11), 왼쪽 hip(23), 왼쪽 knee(25)
    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z)
    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z)
    left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z)
    # knee_flexion: 왼쪽 hip(23), 왼쪽 knee(25), 왼쪽 ankle(27)
    left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z)

    angles = {}
    angles["hip_flexion"] = calculate_angle_3d(left_shoulder, left_hip, left_knee)
    angles["knee_flexion"] = calculate_angle_3d(left_hip, left_knee, left_ankle)

    # 디버깅: 각도 출력
    print(f"Frame {frame_count}: hip_flexion = {angles['hip_flexion']:.2f}, knee_flexion = {angles['knee_flexion']:.2f}")

    feedback = stretch.provide_feedback(angles)

    # Mediapipe 랜드마크와 연결선을 그리기
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return {"angles": angles, "feedback": feedback}, image

# ---------------------------------------------
# 5. 메인 함수: 웹캠, Mediapipe Pose, 동작 감지 및 반복 카운트
# ---------------------------------------------
def main():
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # Mediapipe Pose 객체 생성 (실시간 처리에 적합하도록 파라미터 설정)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 여기서는 sit_to_stand 동작을 감지 (Stretch 객체)
    stretch = STRETCHES["sit_to_stand"]

    rep_count = 0
    # 상태 머신: "sitting" (앉은 상태)와 "standing" (일어선 상태)
    movement_state = "sitting"
    # sit_to_stand에서는 hip_flexion 기준으로 판별
    # 임계값 재설정: 앉은 상태: hip_flexion > 100°, 일어선 상태: hip_flexion < 60°
    upper_threshold = 100
    lower_threshold = 60

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        data, annotated_frame = process_frame_mediapipe(frame.copy(), pose, stretch, frame_count)
        if data and "angles" in data and "hip_flexion" in data["angles"]:
            current_angle = data["angles"]["hip_flexion"]
            print(f"Current state: {movement_state}, hip_flexion: {current_angle:.2f}")
            # 상태 머신: 앉은 상태에서 일어섰음을 감지
            if movement_state == "sitting" and current_angle < lower_threshold:
                movement_state = "standing"
                print("Transition: Sitting -> Standing")
                tts_queue.put("Standing detected")
            elif movement_state == "standing" and current_angle > upper_threshold:
                rep_count += 1
                movement_state = "sitting"
                print("Transition: Standing -> Sitting, Repetition count:", rep_count)
                tts_queue.put(f"{rep_count} repetition completed")

        cv2.putText(annotated_frame, f"Reps: {rep_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if data and data["feedback"]:
            y0 = 80
            dy = 30
            for i, fb in enumerate(data["feedback"]):
                cv2.putText(annotated_frame, fb, (50, y0 + i*dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # TTS 피드백 전송
            for fb in data["feedback"]:
                tts_queue.put(fb)

        cv2.imshow("Mediapipe Pose", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None)
    tts_thread.join()
    pose.close()

if __name__ == "__main__":
    main()
