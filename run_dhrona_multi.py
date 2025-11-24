import os
import sys
import time

# System file validation and auto-download for YOLO requirements
def download_file(filename, url):
    import requests
    try:
        r = requests.get(url, allow_redirects=True)
        with open(filename, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        print(f"Failed to download {filename}. Error: {e}")
        sys.exit()

# Ensure YOLO files exist
if not os.path.exists("yolov3-tiny.weights"):
    download_file("yolov3-tiny.weights", "https://pjreddie.com/media/files/yolov3-tiny.weights")

if not os.path.exists("yolov3-tiny.cfg"):
    download_file("yolov3-tiny.cfg", "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg")

# Ensure gesture model exists
if not os.path.exists("dhrona_model.h5"):
    print("Model file 'dhrona_model.h5' missing.")
    sys.exit()

# Core libraries
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# System configuration
CONFIDENCE_THRESHOLD = 0.60
SKIP_FRAMES = 2

# COCO class list for YOLO detection
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Load gesture recognition components
model = load_model('dhrona_model.h5')
with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)

# Load YOLO object detector
yolo_net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load MediaPipe pose estimator
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=0)

# Object detection function using YOLO
def get_all_objects(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)
    
    detections = []
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                cx, cy, bw, bh = (det[0:4] * np.array([w, h, w, h])).astype("int")
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append((class_ids[i], confidences[i], boxes[i]))
    return results

# Initialize camera with fallback
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit()

cap.set(3, 1280)
cap.set(4, 720)

frame_count = 0
last_detections = []

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Periodic YOLO execution
    if frame_count % SKIP_FRAMES == 0:
        last_detections = get_all_objects(frame)
    
    frame_count += 1
    
    # Process detections
    for (class_id, conf, box) in last_detections:
        x, y, w, h = box
        label_name = COCO_CLASSES[class_id]
        
        # Gesture recognition when person detected
        if label_name == "person":
            x = max(0, x); y = max(0, y)
            w = min(frame.shape[1]-x, w); h = min(frame.shape[0]-y, h)
            crop = frame[y:y+h, x:x+w]
            
            gesture_text = "Scanning"
            color = (0, 255, 0)
            
            if crop.size > 0:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_crop)
                
                if results.pose_landmarks:
                    row = []
                    for lm in results.pose_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z, lm.visibility])
                    
                    try:
                        feat = scaler.transform([row])
                        pred = model.predict(feat)
                        idx = np.argmax(pred)
                        ai_conf = pred[0][idx]
                        
                        if ai_conf > CONFIDENCE_THRESHOLD:
                            gesture_text = encoder.classes_[idx].upper()
                            if gesture_text in ['ATTENTION', 'SOS']: 
                                color = (0, 0, 255)
                    except:
                        pass
                    
                    # Skeleton overlay
                    for conn in mp_pose.POSE_CONNECTIONS:
                        start_idx = conn[0]
                        end_idx = conn[1]
                        s = results.pose_landmarks.landmark[start_idx]
                        e = results.pose_landmarks.landmark[end_idx]
                        cv2.line(frame,
                                 (int(x + s.x*w), int(y + s.y*h)),
                                 (int(x + e.x*w), int(y + e.y*h)),
                                 (255, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"PERSON: {gesture_text}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Non-person object labeling
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
            cv2.putText(frame, label_name.upper(), (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

    cv2.imshow("Dhrona Multi-Target", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
