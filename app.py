from flask import Flask, render_template, Response
import cv2
import time
import mediapipe as mp
import numpy as np
import threading

app = Flask(__name__)

# 👇 Mapping of camera IDs to video sources
camera_map = {
    1: "/home/user/Downloads/media.mp4",
    2: "/home/user/Downloads/media.mp4"
}

# 👇 Global dictionary to store total counts per camera
global_detection_counts = {
    1: {"smoking": 0, "sleeping": 0, "mobile": 0, "object": 0},
    2: {"smoking": 0, "sleeping": 0, "mobile": 0, "object": 0},
}
lock = threading.Lock()

# 👇 Frame generator per camera
def gen_frames(cam_id):
    cap = cv2.VideoCapture(camera_map[cam_id])

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

    prev_time = time.time()
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        results_pose = pose.process(rgb)
        results_hands = hands.process(rgb)

        label = ""
        box_coords = None

        # ========== Smoke Detection ==========
        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for finger_tip in [8, 12]:
                    x1 = int(hand_landmarks.landmark[finger_tip].x * w)
                    y1 = int(hand_landmarks.landmark[finger_tip].y * h)
                    x2 = int(results_pose.pose_landmarks.landmark[0].x * w)
                    y2 = int(results_pose.pose_landmarks.landmark[0].y * h)
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if dist < 40:
                        label = "Smoking"
                        with lock:
                            global_detection_counts[cam_id]["smoking"] += 1
                        box_coords = (min(x1, x2) - 60, min(y1, y2) - 60, max(x1, x2) + 60, max(y1, y2) + 60)
                        break

        # ========== Sleep Detection ==========
        if results_pose.pose_landmarks:
            nose_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
            neck_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            if nose_y > neck_y:
                label = "Sleeping"
                with lock:
                    global_detection_counts[cam_id]["sleeping"] += 1
                x = int(results_pose.pose_landmarks.landmark[0].x * w)
                y = int(results_pose.pose_landmarks.landmark[0].y * h)
                box_coords = (x - 80, y - 80, x + 80, y + 80)

        # ========== Mobile Detection ==========
        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for finger_tip in [8, 4]:
                    x1 = int(hand_landmarks.landmark[finger_tip].x * w)
                    y1 = int(hand_landmarks.landmark[finger_tip].y * h)
                    x2 = int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x * w)
                    y2 = int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y * h)
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if dist < 40:
                        label = "Using Mobile"
                        with lock:
                            global_detection_counts[cam_id]["mobile"] += 1
                        box_coords = (min(x1, x2) - 60, min(y1, y2) - 60, max(x1, x2) + 60, max(y1, y2) + 60)
                        break

        # ========== Object/Person Detection ==========
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() > 40 and label == "":
            label = "Person/Object Detected"
            with lock:
                global_detection_counts[cam_id]["object"] += 1
            box_coords = (20, 20, w - 20, h - 20)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Draw label and box
        if label:
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if box_coords:
                x1, y1, x2, y2 = box_coords
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 🎥 Overlay counts on frame
        with lock:
            c = global_detection_counts[cam_id]
            total = {
                "smoking": sum(c["smoking"] for c in global_detection_counts.values()),
                "sleeping": sum(c["sleeping"] for c in global_detection_counts.values()),
                "mobile": sum(c["mobile"] for c in global_detection_counts.values()),
                "object": sum(c["object"] for c in global_detection_counts.values()),
            }

        cv2.putText(frame, f"Smoking: {c['smoking']}", (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        cv2.putText(frame, f"Sleeping: {c['sleeping']}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
        cv2.putText(frame, f"Mobile: {c['mobile']}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        cv2.putText(frame, f"Object: {c['object']}", (200, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 200), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Print real-time counts
        print(f"[Camera {cam_id}] Smoking: {c['smoking']}, Sleeping: {c['sleeping']}, Mobile: {c['mobile']}, Object: {c['object']}")
        print(f"[TOTAL] => Smoking: {total['smoking']}, Sleeping: {total['sleeping']}, Mobile: {total['mobile']}, Object: {total['object']}")

        # Encode and stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    pose.close()
    hands.close()

# 👇 Home page
@app.route('/')
def index():
    return render_template('index.html')

# 👇 Individual camera feed
@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    return Response(gen_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

