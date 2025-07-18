from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# 👇 Mapping of camera IDs to video sources
camera_map = {
    1: "/home/user/Downloads/cow.mp4",
    # 2: "0",  # Add other cams if needed
}

# 👇 Frame generator for a camera
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def gen_frames(cam_id):
    cap = cv2.VideoCapture(camera_map[cam_id])
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip and convert
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose and hand landmarks
        results_pose = pose.process(rgb)
        results_hands = hands.process(rgb)

        label = ""  # Default label

        # ========== Smoke Detection ==========
        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for finger_tip in [8, 12]:  # Index/Middle tips
                    x1 = hand_landmarks.landmark[finger_tip].x
                    y1 = hand_landmarks.landmark[finger_tip].y

                    x2 = results_pose.pose_landmarks.landmark[0].x  # Nose
                    y2 = results_pose.pose_landmarks.landmark[0].y

                    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if dist < 0.1:
                        label = "Smoking"
                        break

        # ========== Sleep Detection ==========
        if results_pose.pose_landmarks:
            nose_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
            neck_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            if nose_y > neck_y:  # Head lower than shoulders
                label = "Sleeping"

        # ========== Mobile Detection ==========
        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for finger_tip in [8, 4]:  # Index or Thumb tip
                    x1 = hand_landmarks.landmark[finger_tip].x
                    y1 = hand_landmarks.landmark[finger_tip].y

                    x2 = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x
                    y2 = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y

                    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if dist < 0.1:
                        label = "Using Mobile"
                        break

        # ========== Person/Object Detection ==========
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        person_detected = False
        if gray.mean() > 40:  # Just a basic threshold idea
            person_detected = True
            if label == "":
                label = "Person/Object Detected"

        # Draw label
        if label:
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

        # Encode
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# 👇 Home page showing camera tiles
@app.route('/')
def index():
    return render_template('index.html')

# 👇 Individual camera video feed
@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    return Response(gen_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

