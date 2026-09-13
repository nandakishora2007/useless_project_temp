import sys
import time
import random
import webbrowser
import threading
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import pygetwindow as gw
import pyttsx3

from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFrame

class VisionHUD(QWidget):
    def __init__(self):
        super().__init__()
        
        # Window Flags: Frameless, Always on Top, Translucent Background
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.SubWindow
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.resize(420, 610)
        
        self.old_pos = QPoint()

        # AI Models & Audio Initialization
        print("Loading AI Engine & Audio Synthesizer...")
        self.model = YOLO('yolov8n.pt')
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        
        # Initialize Text-to-Speech Engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 170)

        self.cap = cv2.VideoCapture(0)

        # Logic States & Feature Lists
        self.cooldown = 25
        self.last_trigger = 0
        self.is_distracted = False
        
        # Meme Roulette Links
        self.distractions = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://thezen.zone/",
            "https://zoomquilt.org/",
            "https://pointerpointer.com/",
            "https://eelslap.com/"
        ]
        
        # Audio Shame Messages
        self.shame_phrases = [
            "Productivity detected! Drop your tools immediately!",
            "Warning. Overworking hazard detected. Cease coding at once.",
            "Hey! Stop being productive and look at a meme.",
            "Error 404: Fun not found. Deploying distraction!"
        ]

        self.work_objects = ['laptop', 'keyboard', 'book', 'mouse', 'scissors']
        self.slack_kw = ['youtube', 'netflix', 'twitch', 'discord', 'steam', 'game', 'reddit', 'spotify']
        self.work_kw = ['visual studio', 'vscode', 'github', 'stackoverflow', 'chatgpt', 'docs', 'python', 'terminal']

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def speak_warning(self):
        def run_speech():
            try:
                phrase = random.choice(self.shame_phrases)
                self.tts_engine.say(phrase)
                self.tts_engine.runAndWait()
            except Exception:
                pass
        threading.Thread(target=run_speech, daemon=True).start()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Glass Container Card
        self.card = QFrame(self)
        self.card.setStyleSheet("""
            QFrame {
                background-color: rgba(22, 24, 30, 215);
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 22px;
            }
        """)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(15, 15, 15, 15)

        # Title Bar / Header
        header_layout = QHBoxLayout()
        self.title_lbl = QLabel(" VISION HUD // AUDIO + ROULETTE", self)
        self.title_lbl.setStyleSheet("color: rgba(255, 255, 255, 200); font-weight: bold; font-size: 13px; border: none;")
        header_layout.addWidget(self.title_lbl)
        
        close_btn = QLabel("✕", self)
        close_btn.setStyleSheet("color: rgba(255, 255, 255, 150); font-weight: bold; font-size: 14px; border: none;")
        close_btn.mousePressEvent = lambda e: self.close()
        header_layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        card_layout.addLayout(header_layout)

        # Camera Display Screen Card
        self.cam_lbl = QLabel(self)
        self.cam_lbl.setFixedSize(360, 220)
        self.cam_lbl.setStyleSheet("background-color: rgba(10, 10, 15, 200); border-radius: 14px; border: 1px solid rgba(255,255,255,20);")
        self.cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.cam_lbl)

        # Telemetry Text Labels
        self.status_lbl = QLabel("Status: Initializing...", self)
        self.status_lbl.setStyleSheet("color: #00FF99; font-size: 12px; font-weight: bold; border: none;")
        card_layout.addWidget(self.status_lbl)

        self.window_lbl = QLabel("Active App: Scanning...", self)
        self.window_lbl.setStyleSheet("color: rgba(200, 200, 220, 180); font-size: 11px; border: none;")
        card_layout.addWidget(self.window_lbl)

        main_layout.addWidget(self.card)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        working = False
        face_found = False
        active_title = "Desktop"

        # 1. Active Window Check
        try:
            win = gw.getActiveWindow()
            if win and win.title:
                active_title = win.title.lower()
        except:
            pass

        slack_mode = any(k in active_title for k in self.slack_kw)
        important_work = any(k in active_title for k in self.work_kw)

        # 2. Face Detection
        fd = self.face_detection.process(rgb)
        if fd.detections:
            face_found = True
            for d in fd.detections:
                box = d.location_data.relative_bounding_box
                x, y, bw, bh = int(box.xmin*w), int(box.ymin*h), int(box.width*w), int(box.height*h)
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 120), 1)

        # 3. YOLO Detection
        results = self.model(frame, stream=True, verbose=False)
        for r in results:
            for b in r.boxes:
                obj_name = self.model.names[int(b.cls[0])]
                if float(b.conf[0]) > 0.4 and obj_name in self.work_objects:
                    working = True

        if face_found and important_work:
            working = True

        # 4. Trigger & Cooldown Logic with Audio & Meme Roulette
        now = time.time()
        elapsed = now - self.last_trigger

        if slack_mode:
            working = False
            status_text = "SUPPRESSED (ALREADY SLACKING)"
            status_color = "#FFAA00"
        elif working:
            status_text = "DANGER: WORKING DETECTED"
            status_color = "#FF3355"
            if not self.is_distracted and elapsed > self.cooldown:
                # Trigger Audio Shame & Random Distraction Link
                self.speak_warning()
                target_url = random.choice(self.distractions)
                webbrowser.open(target_url)
                self.is_distracted = True
                self.last_trigger = now
        else:
            status_text = "SAFE / IDLE"
            status_color = "#00FF99"
            self.is_distracted = False

        # Update UI Text Labels
        self.status_lbl.setText(status_text)
        self.status_lbl.setStyleSheet(f"color: {status_color}; font-size: 12px; font-weight: bold; border: none;")
        self.window_lbl.setText(f"Active App: {active_title[:35]}..")

        # Render Camera Frame to PyQt Label
        qt_img = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        self.cam_lbl.setPixmap(QPixmap.fromImage(qt_img).scaled(360, 220, Qt.AspectRatioMode.KeepAspectRatio))

    def mousePressEvent(self, event):
        self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        delta = event.globalPosition().toPoint() - self.old_pos
        self.move(self.pos() + delta)
        self.old_pos = event.globalPosition().toPoint()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    hud = VisionHUD()
    hud.show()
    sys.exit(app.exec())