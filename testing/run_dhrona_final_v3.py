import cv2
import mediapipe as mp
import numpy as np
import pickle
import math
from collections import deque, Counter
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.55  # Lowered to catch more gestures
HISTORY_LENGTH = 6           # Smoothing

# --- LOAD SYSTEM ---
print("Loading Dhrona AI...")
try:
    model = load_model('dhrona_model.h5')
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print(f"SUCCESS. Classes: {encoder.classes_}")
except:
    print("❌ Error: Missing model files.")
    exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- MATH HELPERS ---
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def get_dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

# --- RELAXED LOGIC ---
def apply_custom_logic(ai_label, lm):
    # Landmarks: 11/12=Shoulders, 13/14=Elbows, 15/16=Wrists, 7/8=Ears, 0=Nose
    
    # 1. ATTENTION (SOS) vs CANCEL
    # Rule: If BOTH hands are above the nose, it is AUTOMATICALLY Attention.
    # (Cancel is usually one hand)
    nose_y = lm[0].y
    l_hand_up = lm[15].y < nose_y
    r_hand_up = lm[16].y < nose_y
    
    if l_hand_up and r_hand_up:
        return "attention"  # Force Attention if both hands up
        
    if ai_label == "attention" and not (l_hand_up and r_hand_up):
        return "..." # Reject Attention if hands are down

    # 2. PHONE CALL
    # Rule: Wrist must be somewhat close to Ear.
    # Increased distance threshold from 0.20 -> 0.30 to be more forgiving
    l_dist = get_dist(lm[15], lm[7])
    r_dist = get_dist(lm[16], lm[8])
    
    if ai_label == "phonecall":
        # If hand is too far from head, reject it
        if l_dist > 0.3 and r_dist > 0.3:
            return "..." 

    # 3. DIRECTION (Pointing)
    # Rule: Arm must be somewhat straight (>120 degrees, lowered from 150)
    l_angle = calculate_angle(lm[11], lm[13], lm[15])
    r_angle = calculate_angle(lm[12], lm[14], lm[16])
    
    if ai_label == "direction":
        # If both arms are bent, it's not pointing
        if l_angle < 120 and r_angle < 120:
            return "..."

    return ai_label

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

history = deque(maxlen=HISTORY_LENGTH)
last_label = "Scanning..."
last_conf = 0.0

print("\n--- DHRONA V6 LIVE ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
        feat = scaler.transform([row])
        pred = model.predict(feat)
        idx = np.argmax(pred)
        conf = pred[0][idx]
        ai_label = encoder.classes_[idx]
        
        # Apply Relaxed Logic
        final_label = apply_custom_logic(ai_label, results.pose_landmarks.landmark)
        
        if conf > CONFIDENCE_THRESHOLD and final_label != "...":
            history.append(final_label)
            if final_label in ['attention', 'sos'] and last_label != final_label:
                print(f">>> 🚨 EMERGENCY: {final_label.upper()}")
        else:
            history.append("...")

        counts = Counter(history)
        most_common, count = counts.most_common(1)[0]
        
        # Fast reaction (requires 3 frames)
        if count >= 3 and most_common != "...":
            last_label = most_common
            last_conf = conf

    color = (0, 255, 0)
    if last_label in ['sos', 'attention']: color = (0, 0, 255)
    if last_label == "direction": color = (255, 255, 0)
    
    text = f"{last_label.upper()} ({int(last_conf*100)}%)"
    cv2.rectangle(frame, (20, 20), (350, 70), (0,0,0), -1)
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Dhrona AI (V6)", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()