import os
import sys
import time

# 1. AUTO-DOWNLOADER
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

# 2. IMPORTS
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 3. CONFIGURATION
CONFIDENCE_THRESHOLD = 0.75
SKIP_FRAMES = 3

# COCO class list
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

# 4. LOAD MODELS
print("Loading Dhrona Brain...")
model = load_model('dhrona_model.h5')
with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
with open('encoder.pkl', 'rb') as f: encoder = pickle.load(f)

print("Loading YOLO...")
yolo_net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

print("Loading MediaPipe...")
mp_pose = mp.solutions.pose
# Complexity 0 is faster for Pi else use 1
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=0)

# 5. HELPER FUNCTIONS
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

# 6. CAMERA SETUP (PI SPECIFIC)
print("Starting Camera...")

# Use V4L2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Use lower resolution for better FPS on Pi CPU
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Camera not detected.")
    sys.exit()

print("--- SYSTEM LIVE (HEADLESS MODE) ---")

frame_count = 0
last_detections = []

# 7. MAIN LOOP
try:
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Frame lost.")
            break
        
        # Periodic YOLO execution
        if frame_count % SKIP_FRAMES == 0:
            last_detections = get_all_objects(frame)
        
        frame_count += 1
        
        # Process detections
        for (class_id, conf, box) in last_detections:
            x, y, w, h = box
            label_name = COCO_CLASSES[class_id]
            
            # A. PERSON DETECTED -> CHECK GESTURE
            if label_name == "person":
                # Safe crop
                x = max(0, x); y = max(0, y)
                w = min(frame.shape[1]-x, w); h = min(frame.shape[0]-y, h)
                crop = frame[y:y+h, x:x+w]
                
                if crop.size > 0:
                    # Convert to RGB for MediaPipe
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
                                
                                # [PI CHANGE] Action Logic (Instead of Drawing)
                                if gesture_text in ['ATTENTION', 'SOS']:
                                    print(f">>> ALERT: SOS Detected! (Conf: {int(ai_conf*100)}%)")
                                    # TODO: vehicle.mode = VehicleMode("LOITER")
                                    
                                elif gesture_text == 'CANCEL':
                                    print(f">>> INFO: Cancel Signal Received.")
                                    # TODO: vehicle.mode = VehicleMode("AUTO")
                                    
                        except:
                            pass
            # B. OBJECT DETECTED
            else:
                # [PI CHANGE] Just print important objects found
                if label_name in ['backpack', 'bottle', 'cell phone']:
                    print(f">>> OBJECT: {label_name.upper()} found.")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    cap.release()
    print("System Shutdown.")