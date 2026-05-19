# AI Proctor Guard 🛡️🔍

An AI-powered monitoring tool designed to ensure interview integrity by tracking gaze, presence, and system activity.

## 🚀 Features

- **Gaze Tracking**: Detects if the person is looking left, right, or away from the screen using MediaPipe Iris Tracking.
- **Presence Detection**: Flags if multiple people are in the camera frame.
- **Window Monitoring**: Tracks active window changes and flags suspicious laptop searches (e.g., ChatGPT, Chrome, Search).
- **Automated Logging**: Generates a detailed `proctor_report.log` with timestamps for all suspicious activities.

## 🛠️ Technical Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- PyGetWindow
- NumPy

## 📦 Installation

1.  **Install Dependencies**:
    ```bash
    pip install opencv-python mediapipe numpy pygetwindow
    ```

2.  **Run the Monitor**:
    ```bash
    python main.py
    ```

## 📂 Project Structure

- `main.py`: The central hub integrating camera and system monitoring.
- `gaze_detector.py`: MediaPipe logic for iris and face tracking.
- `window_monitor.py`: System-level active window tracking.
- `logs/`: Contains session reports and activity history.

---
*Note: This tool is intended for educational and proctoring purposes with user consent.*
