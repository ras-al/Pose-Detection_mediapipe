import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model

#CONFIGURATION
CONFIDENCE_THRESHOLD = 0.75
HISTORY_LENGTH = 8 

#LOAD BRAIN
print("Loading Dhrona Clean AI...")
try:
    model = load_model('dhrona_model.h5')
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print(f"SUCCESS. Classes: {encoder.classes_}")
except:
    print("ERROR: Model files missing. Download them from Colab!")
    exit()

#MEDIAPIPE SETUP
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#HELPER: GUARD RAILS
def check_validity(label, landmarks):
    # Need wrists visible for hand gestures
    if label in ['attention', 'cancel', 'phonecall', 'direction']:
        left_wrist_vis = landmarks[15*4 + 3]
        right_wrist_vis = landmarks[16*4 + 3]
        if left_wrist_vis < 0.5 and right_wrist_vis < 0.5:
            return False
    return True

#MAIN LOOP
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

history = deque(maxlen=HISTORY_LENGTH)
last_label = "Scanning..."
last_conf = 0.0

print("\n--- SYSTEM LIVE ---")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # MediaPipe Processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        # Draw Skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract Data
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
        # Predict
        feat = scaler.transform([row])
        pred = model.predict(feat)
        idx = np.argmax(pred)
        conf = pred[0][idx]
        label = encoder.classes_[idx]
        
        # Logic Checks
        if conf > CONFIDENCE_THRESHOLD and check_validity(label, row):
            history.append(label)
            
            # Print Emergency to CMD
            if label in ['attention', 'sos'] and last_label != label:
                print(f">>>EMERGENCY SIGNAL DETECTED: {label.upper()}")
        else:
            history.append("...")

        # Smoothing (Voting)
        counts = Counter(history)
        most_common, count = counts.most_common(1)[0]
        
        # Require 5 out of 8 frames to agree (Reduces flickering)
        if count >= 5 and most_common != "...":
            last_label = most_common
            last_conf = conf

    # DRAW UI
    color = (0, 255, 0)
    if last_label in ['sos', 'attention']: color = (0, 0, 255)
    
    # Text Box
    text = f"{last_label.upper()} ({int(last_conf*100)}%)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (20, 20), (20 + tw + 20, 20 + th + 30), (0,0,0), -1)
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Dhrona AI (Clean Version)", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()