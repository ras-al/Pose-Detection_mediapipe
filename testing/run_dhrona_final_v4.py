import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model

# CONFIGURATION
CONFIDENCE_THRESHOLD = 0.70
HISTORY_LENGTH = 5

# LOAD BRAIN
print("Loading Dhrona AI...")
try:
    model = load_model('dhrona_model.h5')
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print(f"Loaded! Detecting: {encoder.classes_}")
except:
    print("Error: Model files not found. Run train_dhrona.py first")
    exit()

# MEDIAPIPE
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# MAIN LOOP
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

history = deque(maxlen=HISTORY_LENGTH)
last_label = "Scanning..."
last_conf = 0.0

print("\nSYSTEM LIVE")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Vision Processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 2. Extract Data (33 points -> 132 values)
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
        # 3. AI Prediction
        feat = scaler.transform([row])
        pred = model.predict(feat)
        idx = np.argmax(pred)
        conf = pred[0][idx]
        label = encoder.classes_[idx]
        
        # 4. Filter & Display
        if conf > CONFIDENCE_THRESHOLD:
            history.append(label)
            
            # Console Log for critical gestures
            if label in ['attention', 'sos'] and last_label != label:
                print(f">>>EMERGENCY: {label.upper()}")
            elif label == 'cancel' and last_label != label:
                print(f">>>CANCEL SIGNAL")
        else:
            history.append("...")

        # Smoothing 
        counts = Counter(history)
        most_common, count = counts.most_common(1)[0]
        if most_common != "...":
            last_label = most_common
            last_conf = conf

    # UI DRAWING
    color = (0, 255, 0) # Green
    if last_label in ['sos', 'attention']: color = (0, 0, 255) # Red
    if last_label == 'cancel': color = (0, 165, 255) # Orange
    
    text = f"{last_label.upper()} ({int(last_conf*100)}%)"
    
    # Text Background
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (20, 20), (20 + tw + 20, 20 + th + 30), (0,0,0), -1)
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Dhrona AI", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()