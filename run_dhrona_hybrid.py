import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import requests
import sys
from collections import deque, Counter
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.60 
HISTORY_LENGTH = 5           
MOVEMENT_THRESHOLD = 0.008  
MODEL_OBJ_PATH = 'efficientdet_lite0.tflite'

# --- 1. MODEL DOWNLOADER ---
if not os.path.exists(MODEL_OBJ_PATH) or os.path.getsize(MODEL_OBJ_PATH) < 1000:
    print(f"Downloading {MODEL_OBJ_PATH}...")
    try:
        url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
        r = requests.get(url, allow_redirects=True)
        with open(MODEL_OBJ_PATH, 'wb') as f: f.write(r.content)
    except: sys.exit("Error: Check internet.")

# --- 2. LOAD AI BRAIN ---
print("[1/4] Loading Dhrona Brain...")
try:
    gesture_model = load_model('dhrona_model.h5')
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print(f"SUCCESS. Classes: {encoder.classes_}")
except Exception as e: sys.exit(f"Error: {e}")

# --- 3. INIT VISION ---
print("[2/4] Init Vision...")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

with open(MODEL_OBJ_PATH, 'rb') as f: model_content = f.read()
base_options = python.BaseOptions(model_asset_buffer=model_content)
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# --- 4. GEOMETRIC LOGIC ENGINE (THE FIX) ---

def calculate_angle(a, b, c):
    """Calculates angle between three points."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def get_hybrid_prediction(landmarks, ai_label, movement_score):
    def get_pt(idx): return [landmarks[idx*4], landmarks[idx*4+1]]
    def get_y(idx): return landmarks[idx*4+1]

    nose_y = get_y(0)
    left_wrist_y = get_y(15)
    right_wrist_y = get_y(16)
    
    # --- RULE 1: ATTENTION (SOS) ---
    hands_up = (left_wrist_y < nose_y) and (right_wrist_y < nose_y)
    if hands_up:
        return "attention"

    # --- RULE 2: CANCEL ---
    one_hand_up = (left_wrist_y < nose_y) ^ (right_wrist_y < nose_y)
    if one_hand_up:
        return "cancel"

    # --- RULE 3: SIT vs SQUAT ---
    l_knee_ang = calculate_angle(get_pt(23), get_pt(25), get_pt(27))
    r_knee_ang = calculate_angle(get_pt(24), get_pt(26), get_pt(28))
    avg_knee_ang = (l_knee_ang + r_knee_ang) / 2

    # If knees are bent (< 140 degrees)
    if avg_knee_ang < 140:
        if ai_label in ['sit', 'squat']:
            return ai_label
        return "sit"

    # --- RULE 4: STAND vs WALK ---
    if avg_knee_ang > 150:
        if movement_score > MOVEMENT_THRESHOLD:
            return "walk"
        else:
            return "stand"

    # If no rules matched, trust the AI
    return ai_label

def calculate_movement(current, previous):
    if previous is None: return 0.0
    key_indices = [11, 12, 23, 24, 25, 26]
    curr_pts = np.array(current).reshape(-1, 4)
    prev_pts = np.array(previous).reshape(-1, 4)
    diff = 0.0
    for i in key_indices:
        diff += np.linalg.norm(curr_pts[i, :2] - prev_pts[i, :2])
    return diff / len(key_indices)

def get_target_person_crop(detections):
    largest_area = 0
    target_box = None
    for detection in detections:
        if detection.categories[0].category_name == 'person':
            bbox = detection.bounding_box
            area = bbox.width * bbox.height
            if area > largest_area:
                largest_area = area
                target_box = (int(bbox.origin_x), int(bbox.origin_y),
                              int(bbox.width), int(bbox.height))
    return target_box

# --- 5. MAIN LOOP ---
print("[3/4] Starting Camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280); cap.set(4, 720)

# Warmup
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened(): sys.exit("Error: No camera found.")

history = deque(maxlen=HISTORY_LENGTH)
last_label = "Scanning..."
prev_landmarks = None

print("\n--- SYSTEM LIVE (Hybrid Mode) ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    # 1. Draw Crowd
    for det in detection_result.detections:
        bbox = det.bounding_box
        if det.categories[0].category_name != 'person':
            cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), 
                         (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0, 255, 255), 2)

    # 2. Process Target
    target_box = get_target_person_crop(detection_result.detections)

    if target_box:
        x, y, w, h = target_box
        pad = 60
        h_img, w_img, _ = frame.shape
        x_pad = max(0, x - pad); y_pad = max(0, y - pad)
        w_pad = min(w_img - x_pad, w + 2*pad); h_pad = min(h_img - y_pad, h + 2*pad)
        
        person_crop = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        
        if person_crop.size > 0:
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(crop_rgb)
            
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(person_crop, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                row = []
                for lm in pose_results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                # --- HYBRID ENGINE ---
                move_score = calculate_movement(row, prev_landmarks)
                prev_landmarks = row
                
                # AI Guess
                feat = scaler.transform([row])
                pred = gesture_model.predict(feat, verbose=0)
                ai_label = encoder.classes_[np.argmax(pred)]
                
                # APPLY MATH RULES
                final_label = get_hybrid_prediction(row, ai_label, move_score)
                
                # Store History
                history.append(final_label)
                
                if final_label in ['attention', 'sos'] and last_label != final_label:
                    print(f"EMERGENCY DETECTED: {final_label.upper()}")

                # Smoothing
                most_common = Counter(history).most_common(1)[0][0]
                if Counter(history).most_common(1)[0][1] >= 3:
                    last_label = most_common
                
                # Update UI
                frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad] = person_crop
                
                color = (0, 255, 0)
                if last_label in ['sos', 'attention']: color = (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                text = f"{last_label.upper()}"
                # Smart Text Box
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                draw_y = y - 10 if y > 30 else y + th + 20
                cv2.rectangle(frame, (x, draw_y - th - 5), (x + tw + 10, draw_y + 5), (0,0,0), -1)
                cv2.putText(frame, text, (x + 5, draw_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    else:
        history.append("...")
        last_label = "Scanning..."
    cv2.rectangle(frame, (0,0), (1280, 40), (0,0,0), -1)
    status_msg = f"Status: {last_label.upper()} | Objects: {len(detection_result.detections)}"
    cv2.putText(frame, status_msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Dhrona Hybrid V2", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()