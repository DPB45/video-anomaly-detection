import cv2
import numpy as np
import random

def create_anomaly_video(output_path, frame_count=100, frame_size=(256, 256)):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, frame_size)

    for i in range(frame_count):
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        cv2.circle(frame, (random.randint(50, 200), random.randint(50, 200)), 10, (0, 255, 0), -1)

        if i % 20 == 0:  # Introduce an anomaly every 20 frames
            cv2.circle(frame, (random.randint(50, 200), random.randint(50, 200)), 15, (0, 0, 255), -1)

        out.write(frame)
    out.release()

create_anomaly_video('anomaly_demo.avi')
