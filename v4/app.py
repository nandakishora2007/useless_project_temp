import cv2
import time
import webbrowser
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import pygetwindow as gw

print("Booting Context-Aware Anti-Productivity Engine...")
model = YOLO('yolov8n.pt') 

# Use MediaPipe's dedicated Face Detection (Foolproof bounding boxes)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

COOLDOWN_SECONDS = 25
NOTEBOOK_WORK_THRESHOLD = 2.5 
WORK_OBJECTS = ['laptop', 'keyboard', 'book', 'mouse', 'scissors']

# Keywords that indicate you are ALREADY slacking off (Skip triggering)
SLACK_KEYWORDS = ['youtube', 'netflix', 'twitch', 'discord', 'steam', 'game', 'reddit', 'spotify']

# Keywords that indicate high-importance work
WORK_KEYWORDS = ['visual studio', 'vscode', 'github', 'stackoverflow', 'chatgpt', 'docs', 'python', 'terminal', 'cmd']

is_distracted = False
last_trigger_time = 0
downward_look_start = None

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    person_is_working = False
    face_detected = False
    active_app_title = "Unknown"

    # ==========================================
    # 1. ACTIVE WINDOW & TAB IMPORTANCE CHECK
    # ==========================================
    try:
        active_win = gw.getActiveWindow()
        if active_win and active_win.title:
            active_app_title = active_win.title.lower()
    except Exception:
        active_app_title = "Desktop"

    # Check if user is already doing non-productive work
    already_slack_mode = any(keyword in active_app_title for keyword in SLACK_KEYWORDS)
    is_important_work = any(keyword in active_app_title for keyword in WORK_KEYWORDS)

    # ==========================================
    # 2. FOOLPROOF FACE DETECTION
    # ==========================================
    fd_results = face_detection.process(image_rgb)
    if fd_results.detections:
        face_detected = True
        for detection in fd_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x1 = int(bboxC.xmin * img_w)
            y1 = int(bboxC.ymin * img_h)
            w = int(bboxC.width * img_w)
            h = int(bboxC.height * img_h)
            
            # Draw clean face detection box
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, "Target Acquired", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ==========================================
    # 3. YOLO DESK & WORK OBJECT DETECTION
    # ==========================================
    results = model(frame, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            obj_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            if conf > 0.4 and obj_name in WORK_OBJECTS:
                person_is_working = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"WORK OBJ: {obj_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # If high importance work app is open + face is there, trigger productivity flag
    if face_detected and is_important_work:
        person_is_working = True

    # ==========================================
    # 4. CONTEXT-AWARE DISTRIBUION LOGIC
    # ==========================================
    current_time = time.time()
    time_since_last = current_time - last_trigger_time

    # SUPPRESSION RULE: If already slacking, do not trigger!
    if already_slack_mode:
        person_is_working = False
        status_override = "SUPPRESSED: ALREADY SLACKING"
    else:
        status_override = "ACTIVE MONITORING"

    if person_is_working and not is_distracted and not already_slack_mode:
        if time_since_last > COOLDOWN_SECONDS:
            webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            is_distracted = True
            last_trigger_time = current_time
    elif not person_is_working:
        is_distracted = False

    # ==========================================
    # 5. UI DASHBOARD OVERLAY
    # ==========================================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 180), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "CONTEXT-AWARE ANTI-PRODUCTIVITY", (15, 30), font, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Mode: {status_override}", (15, 60), font, 0.45, (0, 255, 255), 1)
    
    # Display Window Title snippet
    short_title = (active_app_title[:35] + '..') if len(active_app_title) > 35 else active_app_title
    cv2.putText(frame, f"Active Window: {short_title}", (15, 90), font, 0.4, (200, 200, 200), 1)

    verdict_str = "WORKING (WILL DISTRACT)" if person_is_working else "SAFE / SKIPPED"
    verdict_col = (0, 0, 255) if person_is_working else (0, 255, 0)
    cv2.putText(frame, f"Verdict: {verdict_str}", (15, 130), font, 0.6, verdict_col, 2)

    cv2.imshow("Smart Context Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()