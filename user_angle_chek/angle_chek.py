# calibration_object.py
import cv2
import numpy as np
import time
import mediapipe as mp

class Calibrator:
    def __init__(self, calibration_frames=30, rom_duration=5):
        """
        초기화 시 캘리브레이션 프레임 수와 ROM 측정 시간을 설정합니다.
        
        Args:
            calibration_frames (int): 초기 캘리브레이션을 위해 측정할 프레임 수
            rom_duration (int or float): ROM 측정을 진행할 시간(초)
        """
        self.calibration_frames = calibration_frames
        self.rom_duration = rom_duration
        
        # Mediapipe Pose 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, 
                                      min_tracking_confidence=0.5)
        
        # 측정할 관절과 각 관절에 사용할 landmark 인덱스 설정
        self.joint_landmark_map = {
            "hip_flexion": [self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                            self.mp_pose.PoseLandmark.LEFT_HIP.value,
                            self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            "knee_flexion": [self.mp_pose.PoseLandmark.LEFT_HIP.value,
                             self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                             self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
            "ankle_dorsiflexion": [self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                                   self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
                                   self.mp_pose.PoseLandmark.LEFT_HEEL.value],
            # 예시: hip_abduction은 왼쪽 어깨-엉덩이-무릎 각도로 계산 (실제 측정 방법은 다를 수 있음)
            "hip_abduction": [self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              self.mp_pose.PoseLandmark.LEFT_HIP.value,
                              self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            # 예시: hip_external_rotation은 왼쪽 엉덩이-무릎-왼쪽 발가락(foot index) 각도로 계산
            "hip_external_rotation": [self.mp_pose.PoseLandmark.LEFT_HIP.value,
                                      self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                                      self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        }
        self.joints = list(self.joint_landmark_map.keys())
        self.init_angles = {}
        self.rom_results = {}

        # 웹캠 초기화 (캘리브레이션 수행 동안 사용)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다.")

    @staticmethod
    def calculate_angle_3d(p1, p2, p3):
        """
        p1, p2, p3: (x, y, z) 좌표.
        반환값: float 타입의 각도(도)
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return float(angle)

    def _collect_calibration_data(self):
        """
        초기 캘리브레이션 단계: 각 관절별 초기 각도를 측정합니다.
        측정된 각도는 딕셔너리 형식으로 반환됩니다.
        """
        calibration_data = {joint: [] for joint in self.joints}
        print("초기 캘리브레이션 진행 중... (바르게 서주세요)")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            if results.pose_landmarks:
                for joint in self.joints:
                    indices = self.joint_landmark_map[joint]
                    pts = []
                    for idx in indices:
                        lm = results.pose_landmarks.landmark[idx]
                        pts.append((lm.x, lm.y, lm.z))
                    angle = Calibrator.calculate_angle_3d(pts[0], pts[1], pts[2])
                    calibration_data[joint].append(angle)
                cv2.putText(frame, "Calibrating initial angles...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 모든 관절에 대해 충분한 프레임 수가 수집되면 종료
            if all(len(calibration_data[joint]) >= self.calibration_frames for joint in self.joints):
                break

        cv2.destroyWindow("Calibration")
        return calibration_data

    def _compute_init_angles(self, calibration_data):
        """
        수집된 캘리브레이션 데이터를 바탕으로 각 관절의 초기 각도(평균)를 계산합니다.
        """
        for joint in self.joints:
            self.init_angles[joint] = float(np.mean(calibration_data[joint]))
        print("캘리브레이션 완료!")
        for joint in self.joints:
            print(f"{joint}: 초기각도 = {self.init_angles[joint]:.2f}°")

    def _collect_rom_data(self):
        """
        ROM 측정 단계: 사용자가 동작(예: 스쿼트)을 수행할 때 각 관절의 모든 각도 측정값을 수집합니다.
        """
        rom_measurements = {joint: [] for joint in self.joints}
        print("이제 동작을 수행하세요. (예: 스쿼트) {}초간 측정합니다.".format(self.rom_duration))
        rom_start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            if results.pose_landmarks:
                for joint in self.joints:
                    indices = self.joint_landmark_map[joint]
                    pts = []
                    for idx in indices:
                        lm = results.pose_landmarks.landmark[idx]
                        pts.append((lm.x, lm.y, lm.z))
                    angle = Calibrator.calculate_angle_3d(pts[0], pts[1], pts[2])
                    rom_measurements[joint].append(angle)
                cv2.putText(frame, "Measuring ROM...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow("ROM Measurement", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if (time.time() - rom_start_time) > self.rom_duration:
                break

        cv2.destroyWindow("ROM Measurement")
        return rom_measurements

    def _compute_rom_results(self, rom_measurements):
        """
        각 관절별로 초기 각도와 ROM 극단 각도를 계산합니다.
        극단 각도는 초기 각도와의 차이가 큰 값(최대 또는 최소)을 선택하며, ROM은 그 차이입니다.
        """
        for joint in self.joints:
            measurements = rom_measurements[joint]
            if len(measurements) == 0:
                continue
            min_angle = min(measurements)
            max_angle = max(measurements)
            init_angle = self.init_angles[joint]
            if abs(init_angle - min_angle) >= abs(max_angle - init_angle):
                extreme = min_angle
            else:
                extreme = max_angle
            rom = abs(extreme - init_angle)
            self.rom_results[joint] = {
                "init_angle": init_angle,
                "extreme_angle": extreme,
                "rom": rom
            }
        print("ROM 측정 결과:")
        for joint, data in self.rom_results.items():
            print(f"{joint}: {data}")

    def run_calibration(self):
        """
        캘리브레이션 및 ROM 측정을 진행한 후, 각 관절별 결과를 담은 딕셔너리를 반환합니다.
        """
        calib_data = self._collect_calibration_data()
        self._compute_init_angles(calib_data)
        rom_data = self._collect_rom_data()
        self._compute_rom_results(rom_data)
        return self.rom_results

    def release(self):
        """리소스(웹캠, Mediapipe Pose)를 해제합니다."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

# 객체 사용 예시 (직접 모듈을 실행할 때)
if __name__ == "__main__":
    calibrator = Calibrator(calibration_frames=30, rom_duration=5)
    rom_results = calibrator.run_calibration()
    print("최종 ROM 결과:")
    for joint, data in rom_results.items():
        print(f"{joint}: {data}")
    calibrator.release()
