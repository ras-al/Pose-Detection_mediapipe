import cv2
import mediapipe as mp
import numpy as np
import pickle
import math
from collections import deque, Counter
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.60
HISTORY_LENGTH = 5
MOVEMENT_THRESHOLD = 0.008

# --- LOAD BRAIN ---
print("Loading Dhrona AI...")
try:
    model = load_model('dhrona_model.h5', compile=False)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print(f"Loaded! Classes: {encoder.classes_}")
except:
    print("Error: Model files not found. Using Logic Mode only.")
    exit()

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- MATH HELPERS ---
def get_dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

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

    # 1. ATTENTION: "Two hands above the nose"
    if l_wrist.y < nose_y and r_wrist.y < nose_y:
        return "attention"

    # 2. CANCEL: "One hand above nose and another at some angle from leg"
    l_up = l_wrist.y < nose_y
    r_up = r_wrist.y < nose_y
    l_down = l_wrist.y > l_hip.y
    r_down = r_wrist.y > r_hip.y

    if (l_up and r_down) or (r_up and l_down):
        return "cancel"

    # 3. PHONE CALL: "Hand is in ear"
    if get_dist(l_wrist, l_ear) < 0.12 or get_dist(r_wrist, r_ear) < 0.12:
        return "phonecall"

    # 4. DIRECTION: "One hand is below shoulder at a direction"
    shoulder_width = abs(l_shoulder.x - r_shoulder.x)
    l_pointing = (l_wrist.y > l_shoulder.y) and (abs(l_wrist.x - l_shoulder.x) > shoulder_width * 1.5)
    r_pointing = (r_wrist.y > r_shoulder.y) and (abs(r_wrist.x - r_shoulder.x) > shoulder_width * 1.5)

    if l_pointing or r_pointing:
        if not (l_up or r_up):
            return "direction"

    # 5. RUNNING/WALK: "Continuous movement"
    if movement_score > MOVEMENT_THRESHOLD:
        return "walk" 

    # 6. DEFAULT
    return None

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280); cap.set(4, 720)

history = deque(maxlen=HISTORY_LENGTH)
last_label = "Scanning..."
last_conf = 0.0
prev_landmarks_obj = None

print("\n--- SYSTEM LIVE (Single Person Strict) ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 1. Calculate Movement
        curr_landmarks_obj = results.pose_landmarks.landmark
        move_score = calculate_movement(curr_landmarks_obj, prev_landmarks_obj)
        prev_landmarks_obj = curr_landmarks_obj
        
        # 2. Get STRICT Label
        strict_label = get_strict_label(curr_landmarks_obj, move_score)
        
        # 3. Get AI Prediction
        row = []
        for lm in curr_landmarks_obj:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        feat = scaler.transform([row])
        pred = model.predict(feat, verbose=0)
        idx = np.argmax(pred)
        ai_conf = pred[0][idx]
        ai_label = encoder.classes_[idx]

        # 4. DECISION MERGE 
        if strict_label:
            final_label = strict_label
            confidence = 1.0
        else:
            if ai_label in ['sit', 'squat']:
                final_label = ai_label
                confidence = ai_conf
            else:
                final_label = "stand"
                confidence = ai_conf

        # 5. Display History
        history.append(final_label)
        
        # Console Log
        if final_label in ['attention', 'sos', 'cancel'] and last_label != final_label:
            print(f">>> SIGNAL: {final_label.upper()}")

        # Smoothing
        most_common = Counter(history).most_common(1)[0][0]
        if most_common != "...":
            last_label = most_common
            last_conf = confidence

    # --- UI DRAWING ---
    color = (0, 255, 0) # Green
    if last_label in ['sos', 'attention']: color = (0, 0, 255)
    if last_label == 'cancel': color = (0, 165, 255) 
    
    text = f"{last_label.upper()} ({int(last_conf*100)}%)"
    
    # Text Box
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (20, 20), (20 + tw + 20, 20 + th + 30), (0,0,0), -1)
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Debug: Movement
    cv2.putText(frame, f"Mov: {move_score:.4f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    cv2.imshow("Dhrona Pos", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()