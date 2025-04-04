import cv2
import numpy as np
import torch
from torchvision import transforms

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    cap.release()
    return torch.stack(frames)

if __name__ == "__main__":
    video_tensor = preprocess_video("anomaly_demo.avi")
    print("Processed Video Shape:", video_tensor.shape)
