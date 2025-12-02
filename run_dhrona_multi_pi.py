import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import requests
import sys
import time
from collections import deque, Counter
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.70
HISTORY_LENGTH = 5
MOVEMENT_THRESHOLD = 0.005 
MODEL_OBJ_PATH = 'efficientdet_lite0.tflite'

# --- 1. MODEL DOWNLOADER ---
if not os.path.exists(MODEL_OBJ_PATH) or os.path.getsize(MODEL_OBJ_PATH) < 1000:
    print(f"Downloading Object Model...")
    try:
        url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
        r = requests.get(url, allow_redirects=True)
        with open(MODEL_OBJ_PATH, 'wb') as f: f.write(r.content)
    except: sys.exit("Error: Check internet.")

# --- 2. LOAD BRAIN ---
print("Loading Dhrona AI...")
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = load_model('dhrona_model.h5', compile=False)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print(f"Loaded! Detecting: {encoder.classes_}")
except:
    sys.exit("Error: Gesture model files not found.")

# --- 3. INIT VISION ---
# Pose System
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5, 
                    model_complexity=0)

# Object Detector
with open(MODEL_OBJ_PATH, 'rb') as f: model_content = f.read()
base_options = python.BaseOptions(model_asset_buffer=model_content)
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# --- 4. HELPER: MOVEMENT ---
def calculate_movement(current, previous):
    if previous is None: return 0.0
    key_indices = [11, 12, 23, 24, 25, 26]
    curr_pts = np.array(current).reshape(-1, 4)
    prev_pts = np.array(previous).reshape(-1, 4)
    
    movement = 0.0
    for i in key_indices:
        dist = np.linalg.norm(curr_pts[i, :2] - prev_pts[i, :2])
        movement += dist
    return movement / len(key_indices)

def get_target_person_crop(detections):
    largest_area = 0
    target_box = None
    for det in detections:
        if det.categories[0].category_name == 'person':
            bbox = det.bounding_box
            area = bbox.width * bbox.height
            if area > largest_area:
                largest_area = area
                target_box = (int(bbox.origin_x), int(bbox.origin_y), 
                              int(bbox.width), int(bbox.height))
    return target_box

# --- MAIN LOOP ---
print("Starting Pi Camera...")
cap = cv2.VideoCapture(0) # Standard Pi Camera index
cap.set(3, 640) 
cap.set(4, 480)

history = deque(maxlen=HISTORY_LENGTH)
last_label = "Scanning..."
last_conf = 0.0
prev_landmarks = None

print("\n--- SYSTEM LIVE (Pi Mode) ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # 1. Object Detection
    detection_result = detector.detect(mp_image)
    
    for det in detection_result.detections:
        bbox = det.bounding_box
        cat = det.categories[0]
        color = (0, 255, 0) if cat.category_name == 'person' else (0, 255, 255)
        
        cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), 
                     (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), color, 2)
        cv2.putText(frame, cat.category_name, (bbox.origin_x, bbox.origin_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 2. Target Processing
    target_box = get_target_person_crop(detection_result.detections)

    if target_box:
        x, y, w, h = target_box
        h_img, w_img, _ = frame.shape
        x = max(0, x-20); y = max(0, y-20)
        w = min(w_img-x, w+40); h = min(h_img-y, h+40)
        
        person_crop = frame[y:y+h, x:x+w]
        
        if person_crop.size > 0:
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(crop_rgb)
            
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(person_crop, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                row = []
                for lm in pose_results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                # Logic
                move_score = calculate_movement(row, prev_landmarks)
                prev_landmarks = row
                
                # Predict
                feat = scaler.transform([row])
                pred = model.predict(feat, verbose=0)
                idx = np.argmax(pred)
                conf = pred[0][idx]
                label = encoder.classes_[idx]
                
                if label == "walk" and move_score < MOVEMENT_THRESHOLD:
                    label = "stand"
                    
                if conf > CONFIDENCE_THRESHOLD:
                    history.append(label)
                    if label in ['attention', 'sos'] and last_label != label:
                        print(f">>> EMERGENCY: {label.upper()}")
                else:
                    history.append("...")

                # Smoothing
                most_common = Counter(history).most_common(1)[0][0]
                if most_common != "...":
                    last_label = most_common
                    last_conf = conf
                
                # Update Visuals
                frame[y:y+h, x:x+w] = person_crop
                
                color = (0, 255, 0)
                if last_label in ['sos', 'attention']: color = (0, 0, 255)
                
                text = f"{last_label.upper()} ({int(last_conf*100)}%)"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    else:
        history.append("...")
        last_label = "Scanning..."

    # Top Bar
    cv2.rectangle(frame, (0,0), (640, 30), (0,0,0), -1)
    status = f"Status: {last_label.upper()} | Objects: {len(detection_result.detections)}"
    cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Dhrona Pi", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()