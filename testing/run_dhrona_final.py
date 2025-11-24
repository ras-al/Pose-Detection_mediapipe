import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque, Counter
from tensorflow.keras.models import load_model

CONFIDENCE_THRESHOLD = 0.80 
HISTORY_LENGTH = 5 

#LOAD BRAIN
print("Loading Dhrona AI...")
try:
    model = load_model('dhrona_model.h5')
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)
    print("SUCCESS.")
except:
    print("ERROR: Model files missing. Download them from Colab!")
    exit()

#MEDIAPIPE SETUP
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#CAMERA
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
    
    # Process
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        # Draw Skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract Data (33 points x 4 values = 132)
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
        # Predict
        feat = scaler.transform([row])
        pred = model.predict(feat)
        idx = np.argmax(pred)
        conf = pred[0][idx]
        label = encoder.classes_[idx]
        
        # Logic
        if conf > CONFIDENCE_THRESHOLD:
            history.append(label)
            print(f">>> DETECTED: {label.upper()} ({int(conf*100)}%)")
        else:
            history.append("...")

        # Smoothing
        most_common = Counter(history).most_common(1)[0][0]
        if most_common != "...":
            last_label = most_common
            last_conf = conf

    # UI Design
    color = (0, 255, 0) # Green
    if last_label in ['sos', 'attention']: color = (0, 0, 255) # Red
    
    text = f"{last_label.upper()} ({int(last_conf*100)}%)"
    
    # Background Box for Text
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (20, 20), (20 + tw + 20, 20 + th + 30), (0,0,0), -1)
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Dhrona AI System", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()