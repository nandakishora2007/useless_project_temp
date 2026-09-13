import cv2
import time
import webbrowser
import mediapipe as mp
from ultralytics import YOLO

# 1. Initialize the Models
print("Loading AI Models...")
model = YOLO('yolov8n.pt') 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. Start the Built-in Laptop Webcam
cap = cv2.VideoCapture(0)

# 3. Distraction Logic Variables
is_distracted = False
last_trigger_time = 0
cooldown_seconds = 30 # Wait 30 seconds before opening another video

print("System active. Watching for productivity...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    person_is_working = False
    
    # Check 1: Are you looking forward at the screen? (MediaPipe)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fm_results = face_mesh.process(image_rgb)
    if fm_results.multi_face_landmarks:
        # Basic check: Face is visible and detected
        person_is_working = True 
        
    # Check 2: Are work objects visible? (YOLO)
    results = model(frame, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            object_name = model.names[cls_id]
            # If YOLO sees a laptop, keyboard, or book, you are likely working
            if object_name in ['laptop', 'book', 'keyboard']:
                person_is_working = True
                
    # 4. Trigger the Distraction
    current_time = time.time()
    
    if person_is_working and not is_distracted:
        if (current_time - last_trigger_time) > cooldown_seconds:
            print("🚨 PRODUCTIVITY DETECTED! Launching distraction... 🚨")
            
            # Open YouTube (or any link) in the default web browser
            webbrowser.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            
            is_distracted = True
            last_trigger_time = current_time
            
    # If you walk away or turn around, reset the flag so it can trigger again
    elif not person_is_working:
        is_distracted = False
        
    # 5. Show the camera view locally
    cv2.imshow("Anti-Productivity Cam", frame)
    
    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()