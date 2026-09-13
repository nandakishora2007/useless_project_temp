import cv2
import time
import webbrowser
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import pygetwindow as gw

print("Booting Overhauled Anti-Productivity Dashboard...")
model = YOLO('yolov8n.pt') 

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

COOLDOWN_SECONDS = 25
WORK_OBJECTS = ['laptop', 'keyboard', 'book', 'mouse', 'scissors']
SLACK_KEYWORDS = ['youtube', 'netflix', 'twitch', 'discord', 'steam', 'game', 'reddit', 'spotify']
WORK_KEYWORDS = ['visual studio', 'vscode', 'github', 'stackoverflow', 'chatgpt', 'docs', 'python', 'terminal', 'cmd']

is_distracted = False
last_trigger_time = 0

# Create pinned topmost window
window_name = "Anti-Productivity Command Center"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow(window_name, 960, 540)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    person_is_working = False
    face_detected = False
    active_app_title = "Unknown"
    detected_objects = []

    # 1. Active Window Check
    try:
        active_win = gw.getActiveWindow()
        if active_win and active_win.title:
            active_app_title = active_win.title.lower()
    except Exception:
        active_app_title = "Desktop"

    already_slack_mode = any(kw in active_app_title for kw in SLACK_KEYWORDS)
    is_important_work = any(kw in active_app_title for kw in WORK_KEYWORDS)

    # 2. Face Detection
    fd_results = face_detection.process(image_rgb)
    if fd_results.detections:
        face_detected = True
        for detection in fd_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x1, y1 = int(bboxC.xmin * img_w), int(bboxC.ymin * img_h)
            w, h = int(bboxC.width * img_w), int(bboxC.height * img_h)
            
            # Subtle futuristic bracket for face
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 120), 1)

    # 3. YOLO Object Detection
    results = model(frame, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            obj_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            if conf > 0.4 and obj_name in WORK_OBJECTS:
                detected_objects.append(obj_name)
                person_is_working = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 1)
                cv2.putText(frame, f"{obj_name.upper()} {int(conf*100)}%", (x1, y1 - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 255), 1)

    if face_detected and is_important_work:
        person_is_working = True

    # 4. Trigger Logic & Suppression
    current_time = time.time()
    time_since_last = current_time - last_trigger_time

    if already_slack_mode:
        person_is_working = False
        status_mode_str = "SUPPRESSED (ALREADY SLACKING)"
        mode_color = (255, 180, 0) # Amber
    else:
        status_mode_str = "ACTIVE MONITORING"
        mode_color = (0, 255, 120) # Mint green

    if person_is_working and not is_distracted and not already_slack_mode:
        if time_since_last > COOLDOWN_SECONDS:
            webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            is_distracted = True
            last_trigger_time = current_time
    elif not person_is_working:
        is_distracted = False

    # ==========================================
    # 5. PROFESSIONAL HUD / UI OVERLAY RENDER
    # ==========================================
    # Draw a clean frosted glass sidebar panel on the left
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, img_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Header Branding
    cv2.putText(frame, "ANTI-PRODUCTIVITY HUD", (18, 30), font, 0.55, (255, 255, 255), 2)
    cv2.line(frame, (18, 42), (322, 42), (60, 60, 60), 1)

    # System Status Card
    cv2.putText(frame, "ENGINE STATE", (18, 70), font, 0.38, (140, 140, 140), 1)
    cv2.putText(frame, status_mode_str, (18, 92), font, 0.45, mode_color, 2)

    # Verdict Card
    cv2.putText(frame, "PRODUCTIVITY VERDICT", (18, 135), font, 0.38, (140, 140, 140), 1)
    if person_is_working:
        cv2.putText(frame, "WORKING (INTERRUPTING)", (18, 162), font, 0.5, (0, 50, 255), 2)
    else:
        cv2.putText(frame, "SAFE / IDLE", (18, 162), font, 0.5, (0, 255, 120), 2)

    # Active Window Card
    cv2.putText(frame, "ACTIVE WINDOW", (18, 205), font, 0.38, (140, 140, 140), 1)
    clean_title = (active_app_title[:32] + '..') if len(active_app_title) > 32 else active_app_title
    cv2.putText(frame, clean_title, (18, 227), font, 0.4, (220, 220, 220), 1)

    # Telemetry Data Card
    cv2.putText(frame, "TELEMETRY METRICS", (18, 270), font, 0.38, (140, 140, 140), 1)
    face_txt = "Detected" if face_detected else "Searching..."
    cv2.putText(frame, f"Face Tracking: {face_txt}", (18, 292), font, 0.4, (200, 200, 200), 1)
    
    obj_txt = ", ".join(set(detected_objects)) if detected_objects else "None"
    cv2.putText(frame, f"Desk Objects: {obj_txt}", (18, 315), font, 0.4, (200, 200, 200), 1)

    # Weapon Cooldown Progress Bar
    cv2.putText(frame, "DISTRACTION WEAPON COOLDOWN", (18, 375), font, 0.38, (140, 140, 140), 1)
    cv2.rectangle(frame, (18, 390), (322, 408), (45, 45, 45), -1)
    
    if time_since_last > COOLDOWN_SECONDS:
        cv2.rectangle(frame, (18, 390), (322, 408), (255, 0, 180), -1)
        cv2.putText(frame, "ARMED & READY", (105, 403), font, 0.4, (255, 255, 255), 1)
    else:
        progress = time_since_last / COOLDOWN_SECONDS
        fill_w = int(progress * 304)
        cv2.rectangle(frame, (18, 390), (18 + fill_w, 408), (80, 80, 80), -1)
        seconds_left = int(COOLDOWN_SECONDS - time_since_last)
        cv2.putText(frame, f"RECHARGING... {seconds_left}s", (95, 403), font, 0.4, (255, 255, 255), 1)

    # Render window pinned on top
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()