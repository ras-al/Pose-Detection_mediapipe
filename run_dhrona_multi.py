import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import requests
from collections import deque, Counter
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.70
HISTORY_LENGTH = 5
MOVEMENT_THRESHOLD = 0.005 
MODEL_OBJ_PATH = 'efficientdet_lite0.tflite'

# --- 1. AUTO-DOWNLOAD OBJECT MODEL ---
if not os.path.exists(MODEL_OBJ_PATH):
    print(f"Downloading Object Model ({MODEL_OBJ_PATH})...")
    url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
    r = requests.get(url, allow_redirects=True)
    with open(MODEL_OBJ_PATH, 'wb') as f:
        f.write(r.content)
    print("Download Complete.")

# --- 2. LOAD GESTURE BRAIN ---
print("Loading Dhrona AI...")
try:
    model = load_model('dhrona_model.h5')
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print(f"Loaded! Detecting: {encoder.classes_}")
except:
    print("Error: Gesture model files not found.")
    exit()

# --- 3. INITIALIZE VISUAL SYSTEMS ---

# A. Pose System
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# B. Object Detection System
with open(MODEL_OBJ_PATH, 'rb') as f: model_content = f.read()
base_options = python.BaseOptions(model_asset_buffer=model_content)
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# --- 4. HELPER: CALCULATE MOVEMENT ---
def calculate_movement(current, previous):
    if previous is None: return 0.0
    key_indices = [11, 12, 23, 24, 25, 26]
    curr_pts = np.array(current).reshape(-1, 4)
    prev_pts = np.array(previous).reshape(-1, 4)
    
    movement = 0.0
    for i in key_indices:
        # Distance formula
        dist = np.linalg.norm(curr_pts[i, :2] - prev_pts[i, :2])
        movement += dist
    
    return movement / len(key_indices)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280); cap.set(4, 720)

history = deque(maxlen=HISTORY_LENGTH)
last_label = "Scanning..."
last_conf = 0.0
prev_landmarks = None

print("\n--- SYSTEM LIVE ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Prepare Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # PART 1: OBJECT DETECTION (The Crowd)
    detection_result = detector.detect(mp_image)
    
    for det in detection_result.detections:
        bbox = det.bounding_box
        category = det.categories[0]
        
        # Draw Box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        
        # Color: Green for Person, Yellow for Objects
        color = (0, 255, 0) if category.category_name == 'person' else (0, 255, 255)
        
        cv2.rectangle(frame, start_point, end_point, color, 2)
        cv2.putText(frame, f"{category.category_name} {int(category.score*100)}%", 
                   (bbox.origin_x, bbox.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # PART 2: GESTURE RECOGNITION (The Target)
    pose_results = pose.process(rgb)
    
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract Data
        row = []
        for lm in pose_results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
        # --- LOGIC: MOVEMENT CHECK ---
        # Calculate how much we moved since last frame
        move_score = calculate_movement(row, prev_landmarks)
        prev_landmarks = row
        
        # --- AI PREDICTION ---
        feat = scaler.transform([row])
        pred = model.predict(feat, verbose=0)
        idx = np.argmax(pred)
        conf = pred[0][idx]
        label = encoder.classes_[idx]
        
        if label == "walk" and move_score < MOVEMENT_THRESHOLD:
            label = "stand" 
            
        # Filter & Display
        if conf > CONFIDENCE_THRESHOLD:
            history.append(label)
            
            if label in ['attention', 'sos'] and last_label != label:
                print(f"EMERGENCY: {label.upper()}")
            elif label == 'cancel' and last_label != label:
                print(f"CANCEL SIGNAL")
        else:
            history.append("...")

        # Smoothing
        counts = Counter(history)
        most_common, count = counts.most_common(1)[0]
        if most_common != "...":
            last_label = most_common
            last_conf = conf

    # PART 3: UI DRAWING
    color = (0, 255, 0)
    if last_label in ['sos', 'attention']: color = (0, 0, 255)
    if last_label == 'cancel': color = (0, 165, 255)
    
    text = f"{last_label.upper()} ({int(last_conf*100)}%)"
    
    # Text Background
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (20, 20), (20 + tw + 20, 20 + th + 30), (0,0,0), -1)
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Dhrona AI (Obj + Gesture)", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()