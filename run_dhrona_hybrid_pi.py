import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys
import time
from collections import deque, Counter
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION FOR PI ---
CONFIDENCE_THRESHOLD = 0.60
HISTORY_LENGTH = 4           # Lower history for faster reaction
MOVEMENT_THRESHOLD = 0.008
MODEL_OBJ_PATH = 'efficientdet_lite0.tflite'

# --- 1. LOAD AI BRAIN (Lightweight Mode) ---
print("[1/3] Loading Brain...")
try:
    # Disable GPU for TensorFlow on Pi (often causes issues)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    gesture_model = load_model('dhrona_model.h5', compile=False) # compile=False is faster
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print("SUCCESS.")
except Exception as e:
    sys.exit(f"Error: {e}")

# --- 2. INIT VISION ---
print("[2/3] Init Vision...")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Lower complexity for Pi (0 is fastest, 1 is balanced, 2 is heavy)
pose = mp_pose.Pose(min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5, 
                    model_complexity=0)

# Load Object Detector
with open(MODEL_OBJ_PATH, 'rb') as f: model_content = f.read()
base_options = python.BaseOptions(model_asset_buffer=model_content)
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# --- 3. MATH HELPERS ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def get_hybrid_prediction(landmarks, ai_label, movement_score):
    def get_pt(idx): return [landmarks[idx*4], landmarks[idx*4+1]]
    def get_y(idx): return landmarks[idx*4+1]

    nose_y = get_y(0)
    left_wrist_y = get_y(15); right_wrist_y = get_y(16)
    
    # RULE 1: ATTENTION (Hands High)
    if (left_wrist_y < nose_y) and (right_wrist_y < nose_y): return "attention"
    # RULE 2: CANCEL (One Hand High)
    if (left_wrist_y < nose_y) ^ (right_wrist_y < nose_y): return "cancel"
    
    # RULE 3: SIT vs STAND
    l_knee = calculate_angle(get_pt(23), get_pt(25), get_pt(27))
    r_knee = calculate_angle(get_pt(24), get_pt(26), get_pt(28))
    avg_knee = (l_knee + r_knee) / 2

    if avg_knee < 140: return "sit"
    if avg_knee > 150:
        if movement_score > MOVEMENT_THRESHOLD: return "walk"
        else: return "stand"

    return ai_label

def calculate_movement(current, previous):
    if previous is None: return 0.0
    key_indices = [11, 12, 23, 24, 25, 26]
    curr = np.array(current).reshape(-1, 4)
    prev = np.array(previous).reshape(-1, 4)
    # Simple Manhattan distance is faster than Euclidean on Pi
    diff = np.sum(np.abs(curr[key_indices, :2] - prev[key_indices, :2]))
    return diff / len(key_indices)

def get_target_crop(detections):
    largest = 0; target = None
    for d in detections:
        if d.categories[0].category_name == 'person':
            b = d.bounding_box
            area = b.width * b.height
            if area > largest:
                largest = area
                target = (b.origin_x, b.origin_y, b.width, b.height)
    return target

# --- 4. MAIN LOOP ---
print("[3/3] Starting Pi Camera...")
cap = cv2.VideoCapture(0) # Pi Camera is usually 0
cap.set(3, 640) # Width
cap.set(4, 480) # Height

history = deque(maxlen=HISTORY_LENGTH)
last_label = "Scanning..."
prev_landmarks = None
prev_time = 0

print("\n--- PI SYSTEM LIVE ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # 1. Object Detection (Full Frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)
    
    # Draw Objects
    for det in detection_result.detections:
        b = det.bounding_box
        if det.categories[0].category_name != 'person':
            cv2.rectangle(frame, (b.origin_x, b.origin_y), 
                         (b.origin_x + b.width, b.origin_y + b.height), (0,255,255), 2)

    # 2. Gesture Recognition (Target Only)
    target = get_target_crop(detection_result.detections)
    
    if target:
        x, y, w, h = target
        # Crop safely
        h_img, w_img, _ = frame.shape
        pad = 30
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(w_img, x+w+pad); y2 = min(h_img, y+h+pad)
        
        crop = frame[y1:y2, x1:x2]
        
        if crop.size > 0:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = pose.process(crop_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(crop, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                row = []
                for lm in results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                # Logic
                move_score = calculate_movement(row, prev_landmarks)
                prev_landmarks = row
                
                feat = scaler.transform([row])
                pred = gesture_model.predict(feat, verbose=0)
                ai_label = encoder.classes_[np.argmax(pred)]
                
                final_label = get_hybrid_prediction(row, ai_label, move_score)
                
                history.append(final_label)
                most_common = Counter(history).most_common(1)[0][0]
                if Counter(history).most_common(1)[0][1] >= 2:
                    last_label = most_common
                
                # Update visual
                frame[y1:y2, x1:x2] = crop
                
                # Draw Box
                color = (0,0,255) if last_label in ['attention','sos'] else (0,255,0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, last_label.upper(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Print Emergency
                if last_label in ['attention', 'sos']:
                    print(f"!!! EMERGENCY: {last_label} !!!")

    else:
        last_label = "Scanning..."

    # Status UI
    cv2.putText(frame, f"FPS: {int(fps)} | Status: {last_label.upper()}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Dhrona Pi", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()