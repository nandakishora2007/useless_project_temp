# Anti-Productivity

A context-aware desktop application designed to sabotage overworking by monitoring your physical activity and active software windows, automatically launching a distraction when you work too hard.

## Features

* **VisionOS Glassmorphism UI:** A sleek, frameless, semi-transparent PyQt6 widget that stays pinned **always-on-top** above all browser tabs and applications.
* **Context-Aware Intelligence:** Integrates `pygetwindow` to inspect your active window titles. If you are already slacking off, the system suppresses distractions to prevent redundancy.
* **Computer Vision Telemetry:** Combines **MediaPipe Face Detection** for head/presence tracking with **Ultralytics YOLOv8** to identify physical desk objects like laptops, keyboards, books, and mice.
* **Interactive Controls:** Features an integrated cooldown mechanism and mouse-draggable frame positioning.

## Tech Stack

* **Python**
* **PyQt6** (GUI & Translucent Window Rendering)
* **Ultralytics YOLOv8** (Object Detection)
* **MediaPipe** (Face Detection)
* **PyGetWindow** (Active Window Monitoring)

## Installation & Setup

1. Open your terminal inside the project directory and create a virtual environment:
```powershell
python -m vvenv .venv
.venv\Scripts\Activate.ps1

```


2. Install the required dependencies:
```powershell
pip install PyQt6 opencv-python ultralytics mediapipe pygetwindow numpy

```



## Usage

Run the main application script:

```powershell
python vision_hud.py

```

* Click and drag anywhere on the panel background to move the HUD freely across your desktop.