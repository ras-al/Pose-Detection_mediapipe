# Dhrona Pose Detection

Dhrona is a hybrid pose detection and classification system that combines rule-based logic with a deep learning model to identify human poses and gestures in real-time. It uses MediaPipe for high-fidelity body landmark detection and TensorFlow for pose classification.

This repository contains utilizing code for both high-performance environments (Windows/Desktop) and edge devices (Raspberry Pi).

## Features

- **Hybrid Classification Engine**:
  - **Strict Logic**: rule-based detection for specific gestures like "Attention" (hands up), "Cancel" (cross arms), "Phone Call", and "Direction".
  - **AI Model**: Deep learning model for general postures like "Sit", "Squat", and "Stand".
- **Real-time Visualization**: Overlay of pose landmarks, classification labels, and confidence scores.
- **Cross-Platform**: Optimized scripts for both Windows and Raspberry Pi.
- **Movement Tracking**: rudimentary movement intensity calculation.

## Prerequisites

Ensure you have Python 3.7+ installed.

### Dependencies
Install the required Python packages using pip. (Note: `requirements.txt` is provided in the repo, but here are the core libraries).

```bash
pip install opencv-python mediapipe numpy tensorflow scikit-learn
```

> **Note for Raspberry Pi**: Installing TensorFlow and MediaPipe on Raspberry Pi may require specific pre-built binaries or system-level dependencies. Please refer to official guides for installing `tensorflow` and `mediapipe` on Raspberry Pi OS (e.g., 64-bit OS is recommended).

## Quick Start

### 1. Running on Windows
The Windows version runs with higher resolution input and standard processing adjustments.

**Command:**
```bash
python run_dhrona_hybrid.py
```
- **Input**: Uses the default webcam (starts with index 0, DSHOW backend).
- **Controls**: Press `q` to quit the application.

### 2. Running on Raspberry Pi
The Pi version is optimized for limited resources:
- Disables GPU usage (runs on CPU).
- Uses lower camera resolution (640x480).
- Simplified drawing models.

**Command:**
```bash
python run_dhrona_hybrid_pi.py
```
- **Input**: Uses the default camera interface.
- **Controls**: Press `q` to quit.

## Project Structure

Here is an overview of the key files in this repository:

### Core Execution Files
- **`run_dhrona_hybrid.py`**: **[Primary file for Windows]** Launches the full-featured pose detection system.
- **`run_dhrona_hybrid_pi.py`**: **[Primary file for Raspberry Pi]** Launches the optimized version for edge devices.

### Models & Data
- **`dhrona_model.h5`**: The trained TensorFlow/Keras model used for classification.
- **`scaler.pkl`**: Scikit-learn scaler object for normalizing input features.
- **`encoder.pkl`**: Label encoder to map model outputs to text labels.
- **`yolov3-tiny.*`**: (Optional) Object detection weights config (not currently used in the main hybrid scripts).
- **`dhrona_data.csv`**: The dataset used to train the pose model.

### Training & Tools
- **`pose_trainging_mediapipe.ipynb`**: Jupyter notebook used for training the `dhrona_model.h5`.
- **`1_video_to_csv.py`**: Utility script to extract landmarks from video to CSV for dataset creation.
- **`testing/`**: Directory containing various test scripts (e.g., `run_dhrona_final_v1.py`) used during development.

## How It Works

1.  **Capture**: Video frames are captured via OpenCV.
2.  **Landmark Extraction**: MediaPipe Pose extracts 33 3D landmarks from the body.
3.  **Hybrid Logic**:
    *   **Strict Rules**: Geometric checks (angles, positions) determine specific command gestures immediately (e.g., hands above nose = Attention).
    *   **AI Inference**: If no strict rule triggers, the landmarks are normalized and passed to the neural network to classify the general posture (Sit/Stand/Squat).
4.  **Feedback**: The result is smoothed over a history of frames to prevent flickering and displayed on the screen.

## Troubleshooting

- **"Model files not found"**: Ensure `dhrona_model.h5`, `scaler.pkl`, and `encoder.pkl` are in the same directory as the script you are running.
- **Camera not opening**: Check if another application is using the camera. On Pi, ensure the camera interface is enabled in `raspi-config`.
- **Slow performance**: On Windows, ensure you are not running heavy background tasks. On Pi, ensure you are using the `_pi.py` script which reduces resolution.
