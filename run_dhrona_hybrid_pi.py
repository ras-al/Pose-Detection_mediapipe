import cv2
import mediapipe as mp
import numpy as np
import pickle
import math
import os
import sys
import time
from collections import deque, Counter
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.60
HISTORY_LENGTH = 4 
MOVEMENT_THRESHOLD = 0.008 

# --- LOAD BRAIN ---
print("[1/2] Loading Brain...")
try:
    # Disable GPU for Pi
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = load_model('dhrona_model.h5', compile=False)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print(f"Loaded! Classes: {encoder.classes_}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit()

# --- MEDIAPIPE ---
print("[2/2] Init Vision...")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5, 
                    model_complexity=0)

# --- MATH HELPERS ---
def get_dist(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)

def calculate_movement(current, previous):
    if previous is None: return 0.0
    indices = [11, 12, 23, 24]
    move = 0.0
    for i in indices:
        move += get_dist(current[i], previous[i])
    return move / len(indices)

# --- STRICT LOGIC ENGINE ---
def get_strict_label(landmarks, movement_score):
    nose_y = landmarks[0].y
    l_wrist = landmarks[15]
    r_wrist = landmarks[16]
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_hip = landmarks[23]
    r_hip = landmarks[24]
    l_ear = landmarks[7]
    r_ear = landmarks[8]

    # 1. ATTENTION
    if l_wrist.y < nose_y and r_wrist.y < nose_y:
        return "attention"

    # 2. CANCEL
    l_up = l_wrist.y < nose_y
    r_up = r_wrist.y < nose_y
    l_down = l_wrist.y > l_hip.y
    r_down = r_wrist.y > r_hip.y

    if (l_up and r_down) or (r_up and l_down):
        return "cancel"

    # 3. PHONE CALL
    if get_dist(l_wrist, l_ear) < 0.15 or get_dist(r_wrist, r_ear) < 0.15:
        return "phonecall"

    # 4. DIRECTION
    shoulder_width = abs(l_shoulder.x - r_shoulder.x)
    l_ext = abs(l_wrist.x - l_shoulder.x) > shoulder_width * 1.2
    r_ext = abs(r_wrist.x - r_shoulder.x) > shoulder_width * 1.2
    l_low = l_wrist.y > l_shoulder.y
    r_low = r_wrist.y > r_shoulder.y

    if (l_ext and l_low) or (r_ext and r_low):
        if not (l_up or r_up): return "direction"

    # 5. WALK
    if movement_score > MOVEMENT_THRESHOLD:
        return "walk"

    return None

# --- MAIN LOOP ---
print("Starting Pi Camera...")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

history = deque(maxlen=HISTORY_LENGTH)
last_label = "Scanning..."
last_conf = 0.0
prev_landmarks_obj = None
prev_time = 0

print("\n--- PI SYSTEM LIVE ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 1. Movement
        curr_lms = results.pose_landmarks.landmark
        move_score = calculate_movement(curr_lms, prev_landmarks_obj)
        prev_landmarks_obj = curr_lms
        
        # 2. Strict Label
        strict_label = get_strict_label(curr_lms, move_score)
        
        # 3. AI Prediction
        row = []
        for lm in curr_lms:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        feat = scaler.transform([row])
        pred = model.predict(feat, verbose=0)
        ai_conf = pred[0][np.argmax(pred)]
        ai_label = encoder.classes_[np.argmax(pred)]

        # 4. Decision
        final_label = strict_label
        conf = 1.0
        
        if not final_label:
            if ai_label in ['sit', 'squat']:
                final_label = ai_label
                conf = ai_conf
            else:
                final_label = "stand"
                conf = ai_conf

        history.append(final_label)
        
        if final_label in ['attention', 'sos', 'cancel'] and last_label != final_label:
            print(f">>> SIGNAL: {final_label.upper()}")

        # Smoothing
        most_common = Counter(history).most_common(1)[0][0]
        if most_common != "...":
            last_label = most_common
            last_conf = conf

    # UI
    color = (0, 255, 0)
    if last_label in ['sos', 'attention']: color = (0, 0, 255)
    if last_label == 'cancel': color = (0, 165, 255)
    
    cv2.rectangle(frame, (0,0), (640, 40), (0,0,0), -1)
    status = f"FPS: {int(fps)} | {last_label.upper()} ({int(last_conf*100)}%)"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Dhrona Pi", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()