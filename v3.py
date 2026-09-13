import cv2
import time
import webbrowser
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

print("Booting Anti-Productivity Engine...")
model = YOLO('yolov8n.pt') 

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Configuration & Timers
COOLDOWN_SECONDS = 20
NOTEBOOK_WORK_THRESHOLD = 2.5 
WORK_OBJECTS = ['laptop', 'keyboard', 'book', 'mouse', 'scissors']

is_distracted = False
last_trigger_time = 0
downward_look_start = None

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 1. Smooth out frame for UI (Mirror it)
    frame = cv2.flip(frame, 1) 
    img_h, img_w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # State tracking for this frame
    gaze_status = "No Face Detected"
    pitch, yaw = 0.0, 0.0
    detected_objects = []
    person_is_working = False
    elapsed_down_time = 0.0

    # ==========================================
    # 2. FACE MESH & POSE TRACKING
    # ==========================================
    fm_results = face_mesh.process(image_rgb)
    
    if fm_results.multi_face_landmarks:
        for face_landmarks in fm_results.multi_face_landmarks:
            # Subdued white wireframe (Better UX than thick green)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1) 
            )

            face_2d, face_3d = [], []
            key_indices = [1, 152, 33, 263, 61, 291]
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in key_indices:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d, face_3d = np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64)
            cam_matrix = np.array([[1 * img_w, 0, img_w / 2], [0, 1 * img_w, img_h / 2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            _, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch = angles[0] * 360
            yaw = angles[1] * 360

            # Posture Logic
            if pitch < -12:
                if downward_look_start is None:
                    downward_look_start = time.time()
                elapsed_down_time = time.time() - downward_look_start
                
                if elapsed_down_time >= NOTEBOOK_WORK_THRESHOLD:
                    gaze_status = "Focused on Desk!"
                    person_is_working = True
                else:
                    gaze_status = "Glancing Down..."
            elif yaw < -15 or yaw > 15:
                gaze_status = "Looking Away (Idle)"
                downward_look_start = None
            else:
                gaze_status = "Looking at Screen"
                person_is_working = True
                downward_look_start = None

            # Sleeker 3D Gaze Pointer (Arrow)
            nose_x, nose_y = int(face_landmarks.landmark[1].x * img_w), int(face_landmarks.landmark[1].y * img_h)
            p2 = (int(nose_x + yaw * 3), int(nose_y - pitch * 3))
            cv2.arrowedLine(frame, (nose_x, nose_y), p2, (0, 255, 255), 2, tipLength=0.2)
    else:
        downward_look_start = None

    # ==========================================
    # 3. YOLO OBJECT DETECTION
    # ==========================================
    results = model(frame, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            obj_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            if conf > 0.4:
                detected_objects.append(obj_name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if obj_name in WORK_OBJECTS:
                    person_is_working = True
                    color = (0, 0, 255) # Red for danger (work)
                else:
                    color = (0, 255, 150) # Mint green for safe
                
                # Draw thinner bounding box and solid background for text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + 140, y1), color, -1)
                cv2.putText(frame, f"{obj_name.upper()} {int(conf*100)}%", (x1 + 5, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # ==========================================
    # 4. DISTRACTION TRIGGER LOGIC
    # ==========================================
    current_time = time.time()
    time_since_last = current_time - last_trigger_time
    
    if person_is_working and not is_distracted:
        if time_since_last > COOLDOWN_SECONDS:
            webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            is_distracted = True
            last_trigger_time = current_time
    elif not person_is_working:
        is_distracted = False

    # ==========================================
    # 5. SLEEK UI / UX DASHBOARD
    # ==========================================
    # Dark semi-transparent sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (330, img_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Header
    cv2.putText(frame, "ANTI-PRODUCTIVITY AI", (15, 35), font, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (15, 50), (315, 50), (100, 100, 100), 1)

    # Verdict Module
    cv2.putText(frame, "SYSTEM VERDICT", (15, 80), font, 0.4, (150, 150, 150), 1)
    if person_is_working:
        cv2.putText(frame, "WORKING!", (15, 115), font, 1.3, (50, 50, 255), 3)
    else:
        cv2.putText(frame, "IDLE / SAFE", (15, 115), font, 1.3, (50, 255, 50), 3)

    # Telemetry Data
    cv2.putText(frame, "LIVE TELEMETRY", (15, 160), font, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, "Gaze:", (15, 185), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"{gaze_status}", (75, 185), font, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, "Desk:", (15, 215), font, 0.5, (200, 200, 200), 1)
    objs_str = ", ".join(set(detected_objects)) if detected_objects else "Clear"
    cv2.putText(frame, f"{objs_str}", (75, 215), font, 0.5, (255, 255, 255), 1)

    # Posture Charge Bar
    cv2.putText(frame, "Notebook Focus Timer:", (15, 260), font, 0.4, (150, 150, 150), 1)
    cv2.rectangle(frame, (15, 275), (315, 290), (50, 50, 50), -1) 
    if elapsed_down_time > 0:
        fill_width = int(min((elapsed_down_time / NOTEBOOK_WORK_THRESHOLD) * 300, 300))
        bar_color = (0, 165, 255) # Orange warning
        if fill_width == 300: bar_color = (50, 50, 255) # Red max
        cv2.rectangle(frame, (15, 275), (15 + fill_width, 290), bar_color, -1)

    # Cooldown Recharge Bar
    cv2.putText(frame, "Distraction Weapon:", (15, 330), font, 0.4, (150, 150, 150), 1)
    cv2.rectangle(frame, (15, 345), (315, 360), (50, 50, 50), -1) 
    
    if time_since_last > COOLDOWN_SECONDS:
        cv2.rectangle(frame, (15, 345), (315, 360), (255, 0, 255), -1) # Purple ready
        cv2.putText(frame, "ARMED & READY", (100, 356), font, 0.4, (255, 255, 255), 1)
    else:
        recharge_pct = time_since_last / COOLDOWN_SECONDS
        fill_width = int(recharge_pct * 300)
        cv2.rectangle(frame, (15, 345), (15 + fill_width, 360), (100, 100, 100), -1)
        cv2.putText(frame, f"RECHARGING... {int(COOLDOWN_SECONDS - time_since_last)}s", (90, 356), font, 0.4, (255, 255, 255), 1)

    # ==========================================
    # 6. RENDER
    # ==========================================
    cv2.imshow("Anti-Productivity AI Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()