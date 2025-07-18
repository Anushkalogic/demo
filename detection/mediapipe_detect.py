import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def get_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def run_mediapipe_detection(frame):
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose detection
    pose_results = pose.process(frame_rgb)

    # Hands detection
    hand_results = hands.process(frame_rgb)

    detected = []

    # SMOKE: Hand near mouth (e.g., right hand near nose or mouth)
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        nose = (int(lm[mp_pose.PoseLandmark.NOSE].x * width), int(lm[mp_pose.PoseLandmark.NOSE].y * height))
        right_hand = (int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * width), int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * height))

        dist = get_distance(nose, right_hand)
        if dist < 60:  # 🔍 Threshold can be adjusted
            cv2.putText(frame, "MediaPipe: Smoke Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            detected.append("smoke")

    # SLEEP: Head down (neck lower than shoulder or facing down)
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        nose_y = lm[mp_pose.PoseLandmark.NOSE].y
        shoulder_y = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        if nose_y - shoulder_y > 0.1:
            cv2.putText(frame, "MediaPipe: Sleep Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            detected.append("sleep")

    # MOBILE: Hand near ear (right wrist near right ear/eye)
    if pose_results.pose_landmarks:
        right_ear = (int(lm[mp_pose.PoseLandmark.RIGHT_EAR].x * width), int(lm[mp_pose.PoseLandmark.RIGHT_EAR].y * height))
        right_hand = (int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * width), int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * height))
        if get_distance(right_hand, right_ear) < 50:
            cv2.putText(frame, "MediaPipe: Mobile Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            detected.append("mobile")

    # 🟡 Optionally draw landmarks
    mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return frame

