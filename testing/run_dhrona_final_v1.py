import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.75  # Increased slightly to reduce noise
HISTORY_LENGTH = 5 

# --- LOAD BRAIN ---
print("Loading Dhrona AI...")
try:
    model = load_model('dhrona_model.h5', compile=False)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print("SUCCESS.")
except:
    print("ERROR: Model files missing. Download them from Colab!")
    exit()

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- MATH HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculates angle between three points (Shoulder-Elbow-Wrist)."""
    a = np.array(a) # Shoulder
    b = np.array(b) # Elbow
    c = np.array(c) # Wrist
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def verify_gesture_logic(label, landmarks):
    """
    Double-checks the AI prediction using physical geometry.
    Returns True if the gesture looks physically correct.
    """
    # Helper to get Y coordinate (0 is top, 1 is bottom)
    def get_y(idx): return landmarks[idx*4 + 1]
    # Helper to get [x,y] coordinate
    def get_pt(idx): return [landmarks[idx*4], landmarks[idx*4+1]]

    # --- FIX 1: ATTENTION (SOS) ---
    # Rule: Both wrists must be ABOVE the nose.
    if label == "attention":
        nose_y = get_y(0)
        left_wrist_y = get_y(15)
        right_wrist_y = get_y(16)
        
        # If either hand is below the nose, it's NOT Attention.
        if (left_wrist_y > nose_y) or (right_wrist_y > nose_y):
            return False

    # --- FIX 2: DIRECTION (Pointing) ---
    # Rule: At least one arm must be STRAIGHT (Angle > 140 degrees).
    if label == "direction":
        # Left Arm: Shoulder(11) - Elbow(13) - Wrist(15)
        angle_L = calculate_angle(get_pt(11), get_pt(13), get_pt(15))
        # Right Arm: Shoulder(12) - Elbow(14) - Wrist(16)
        angle_R = calculate_angle(get_pt(12), get_pt(14), get_pt(16))
        
        # If BOTH arms are bent, you are not pointing.
        if angle_L < 140 and angle_R < 140:
            return False

    return True

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

history = deque(maxlen=HISTORY_LENGTH)
last_label = "Scanning..."
last_conf = 0.0

print("\n--- SYSTEM LIVE (Press 'q' to quit) ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Process
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        # Draw Skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 2. Extract Data
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
        # 3. Predict
        feat = scaler.transform([row])
        pred = model.predict(feat, verbose=0) # verbose=0 stops lag
        idx = np.argmax(pred)
        conf = pred[0][idx]
        label = encoder.classes_[idx]
        
        # 4. Verify Logic (The Fix)
        is_valid = verify_gesture_logic(label, row)
        
        if conf > CONFIDENCE_THRESHOLD and is_valid:
            history.append(label)
            # Only print if it's a new steady detection
            if label != last_label and history.count(label) > 2:
                print(f">>> DETECTED: {label.upper()} ({int(conf*100)}%)")
        else:
            history.append("...")

        # 5. Smoothing
        most_common = Counter(history).most_common(1)[0][0]
        if most_common != "...":
            last_label = most_common
            last_conf = conf

    # --- UI DRAWING ---
    color = (0, 255, 0) # Green
    if last_label in ['sos', 'attention']: color = (0, 0, 255) # Red
    
    text = f"{last_label.upper()} ({int(last_conf*100)}%)"
    
    # Background Box
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (20, 20), (20 + tw + 20, 20 + th + 30), (0,0,0), -1)
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Dhrona AI System", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()