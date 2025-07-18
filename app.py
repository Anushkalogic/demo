from flask import Flask, render_template_string, Response
import cv2
import time
import mediapipe as mp
import numpy as np
import threading

app = Flask(__name__)

camera_map = {
    1: "/home/user/Downloads/media.mp4",
    2: "/home/user/Downloads/cow.mp4",
    3: "/home/user/Downloads/media.mp4",
    4: "/home/user/Downloads/cow.mp4"
}

global_detection_counts = {
    cam_id: {"smoking": 0, "sleeping": 0, "mobile": 0, "object": 0}
    for cam_id in camera_map
}
lock = threading.Lock()

def gen_frames(cam_id):
    cap = cv2.VideoCapture(camera_map[cam_id])
    if not cap.isOpened():
        print(f"[ERROR] Cannot open Camera {cam_id}")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5)

    prev_time = time.time()
    frame_count = 0
    hand_history = []
    max_history = 5

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        results_pose = pose.process(rgb)
        results_hands = hands.process(rgb)

        label = ""
        box_coords = None
        cam_counts = global_detection_counts[cam_id]

        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                hx = int(hand_landmarks.landmark[8].x * w)
                hy = int(hand_landmarks.landmark[8].y * h)
                mx = int(results_pose.pose_landmarks.landmark[0].x * w)
                my = int(results_pose.pose_landmarks.landmark[0].y * h)

                hand_history.append((hx, hy))
                if len(hand_history) > max_history:
                    hand_history.pop(0)

                if np.sqrt((hx - mx)**2 + (hy - my)**2) < 50:
                    label = "Smoking"
                    with lock:
                        cam_counts["smoking"] += 1
                    box_coords = (hx-50, hy-50, hx+50, hy+50)
                elif len(hand_history) >= 2:
                    dx = hand_history[-1][0] - hand_history[0][0]
                    dy = hand_history[-1][1] - hand_history[0][1]
                    speed = np.sqrt(dx**2 + dy**2)
                    if speed > 80 and np.sqrt((hx - mx)**2 + (hy - my)**2) > 80:
                        label = "Smoking (Throw)"
                        with lock:
                            cam_counts["smoking"] += 1
                        box_coords = (hx-50, hy-50, hx+50, hy+50)

        if results_pose.pose_landmarks:
            nose_y = results_pose.pose_landmarks.landmark[0].y
            shoulder_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            if nose_y > shoulder_y:
                label = "Sleeping"
                with lock:
                    cam_counts["sleeping"] += 1
                x = int(results_pose.pose_landmarks.landmark[0].x * w)
                y = int(results_pose.pose_landmarks.landmark[0].y * h)
                box_coords = (x-50, y-50, x+50, y+50)

        if results_hands.multi_hand_landmarks and results_pose.pose_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for tip in [4, 8]:
                    hx, hy = int(hand_landmarks.landmark[tip].x * w), int(hand_landmarks.landmark[tip].y * h)
                    for ear_landmark in [mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR]:
                        ear = results_pose.pose_landmarks.landmark[ear_landmark]
                        ex, ey = int(ear.x * w), int(ear.y * h)
                        if np.sqrt((hx - ex)**2 + (hy - ey)**2) < 80:
                            label = "Using Mobile"
                            with lock:
                                cam_counts["mobile"] += 1
                            box_coords = (hx-50, hy-50, hx+50, hy+50)
                            break

        if label == "":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.mean() > 40:
                label = "Person/Object Detected"
                with lock:
                    cam_counts["object"] += 1
                box_coords = (20, 20, w-20, h-20)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()

        if label:
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if box_coords:
                x1, y1, x2, y2 = box_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        with lock:
            c = cam_counts.copy()
        print(f"[Camera {cam_id}] Smoking: {c['smoking']}, Sleeping: {c['sleeping']}, Mobile: {c['mobile']}, Object: {c['object']}")

        cv2.putText(frame, f"Smoking: {c['smoking']}", (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Sleeping: {c['sleeping']}", (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"Mobile: {c['mobile']}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    pose.close()
    hands.close()

@app.route('/')
def index():
    cams = camera_map.keys()
    return render_template_string("""
    <html>
    <head>
        <title>Live Detection Feeds</title>
        <style>
            body {
                background: #000;
                color: white;
                font-family: Arial;
            }
            h1 {
                text-align: center;
                padding: 10px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                padding: 10px;
            }
            .cam {
                border: 2px solid #ccc;
                padding: 5px;
                background: #111;
            }
            .cam h3 {
                margin: 0;
                color: #0f0;
            }
            img {
                width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <h1>Live Detection Feeds</h1>
        <div class="grid">
            {% for cam_id in cams %}
            <div class="cam">
                <h3>Camera {{ cam_id }}</h3>
                <img src="/video_feed/{{ cam_id }}">
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """, cams=cams)

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    return Response(gen_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

