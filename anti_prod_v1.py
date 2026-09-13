import cv2
import time
import webbrowser
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# 1. Initialize Models & Utilities
print("Loading AI Models and Telemetry...")
model = YOLO('yolov8n.pt') 

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# State Tracking
is_distracted = False
last_trigger_time = 0
cooldown_seconds = 20

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    img_h, img_w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    gaze_status = "No Face Detected"
    pitch, yaw = 0.0, 0.0
    detected_objects = []
    person_is_working = False

    # A. Face Mesh & Pose Tracking
    fm_results = face_mesh.process(image_rgb)
    
    if fm_results.multi_face_landmarks:
        for face_landmarks in fm_results.multi_face_landmarks:
            # Draw green face mesh grid
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Extract 2D and 3D points for Pose Estimation
            face_2d = []
            face_3d = []
            key_indices = [1, 152, 33, 263, 61, 291]

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in key_indices:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([
                [focal_length, 0, img_w / 2],
                [0, focal_length, img_h / 2],
                [0, 0, 1]
            ])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch = angles[0] * 360
            yaw = angles[1] * 360

            if pitch < -10:
                gaze_status = "Looking Down (Desk)"
                person_is_working = True
            elif yaw < -15 or yaw > 15:
                gaze_status = "Looking Away (Distracted)"
            else:
                gaze_status = "Looking Forward (Screen)"
                person_is_working = True

            # Draw gaze direction pointer from nose
            nose_x, nose_y = int(face_landmarks.landmark[1].x * img_w), int(face_landmarks.landmark[1].y * img_h)
            p1 = (nose_x, nose_y)
            p2 = (int(nose_x + yaw * 3), int(nose_y - pitch * 3))
            cv2.line(frame, p1, p2, (0, 255, 255), 3)

    # B. YOLO Object Detection
    results = model(frame, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            obj_name = model.names[cls_id]
            confidence = float(box.conf[0])
            
            if confidence > 0.4:
                detected_objects.append(obj_name)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if obj_name in ['laptop', 'book', 'keyboard'] else (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{obj_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if obj_name in ['laptop', 'book', 'keyboard']:
                    person_is_working = True

    # C. Trigger Logic
    current_time = time.time()
    time_since_last = current_time - last_trigger_time

    if person_is_working and not is_distracted:
        if time_since_last > cooldown_seconds:
            webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            is_distracted = True
            last_trigger_time = current_time
    elif not person_is_working:
        is_distracted = False

    # D. HUD Telemetry Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    verdict_text = "DANGER: WORKING!" if person_is_working else "SAFE: IDLE/SLACKING"
    verdict_color = (0, 0, 255) if person_is_working else (0, 255, 0)

    cv2.putText(frame, "=== TELEMETRY HUD ===", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Verdict: {verdict_text}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, verdict_color, 2)
    cv2.putText(frame, f"Gaze: {gaze_status}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Head Pose: Pitch={pitch:.1f} | Yaw={yaw:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Objects: {', '.join(set(detected_objects)) if detected_objects else 'None'}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cooldown_disp = f"READY" if time_since_last > cooldown_seconds else f"Cooldown: {int(cooldown_seconds - time_since_last)}s"
    cv2.putText(frame, f"Trigger Status: {cooldown_disp}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Anti-Productivity Telemetry Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()