import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import glob

#CONFIGURATION
VIDEO_FOLDER = "training_videos"
DATA_FILE = "dhrona_data.csv"
SKIP_FRAMES = 2

#MEDIAPIPE SETUP
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def extract_landmarks(results):
    if results.pose_landmarks:
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        return row
    return None

#MAIN
data = []
video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))

if not video_files:
    print(f"No videos found in '{VIDEO_FOLDER}'.")
    exit()

print(f"Found {len(video_files)} videos. Processing...")

for video_path in video_files:
    label = os.path.splitext(os.path.basename(video_path))[0].lower()      
    print(f"--> Processing: {label.upper()}")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0: continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        row = extract_landmarks(results)
        if row:
            data.append([label] + row)
            
    cap.release()

if data:
    cols = ['label'] + [str(i) for i in range(132)]
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(DATA_FILE, index=False)
    print(f"\nCLEAN DATASET GENERATED: {len(df)} samples.")
    print(f"Classes: {df['label'].unique()}")