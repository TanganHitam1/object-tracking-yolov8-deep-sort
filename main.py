import os
import random
import numpy as np

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from tracker import Tracker
import time

video_path = os.path.join('.', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'out3.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                        (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8x.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

output_ = list()
frame_id = 0

detection_threshold = 0.5
while ret:

    results = model(frame)

    for result in results:
        print(type(result))
        # result = Results(result)
        detections = []
        class_names = []
        print(result.speed)
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])
                class_name = model.names[class_id]
                # print(class_name, score)
                class_names.append(class_name)
        # print(len(detections))
        tracker.update(frame, detections)
        # print(len(tracker.tracks))
        # print(tracker.tracks)

        for i, track in enumerate(tracker.tracks):
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            w = x2 - x1
            h = y2 - y1
            output_.append([frame_id, float(track_id), float(x1), float(y1), float(w), float(h), 1, -1, -1 , -1])
            # print(output_)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            try:
                cv2.putText(frame, str(track_id) + ' ' + class_names[i], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass
    frame_id += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps
    fps_text = f"FPS: {frame_time:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cap_out.write(frame)
    cv2.imshow('YOLO Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
np.savetxt('./output3.txt', output_, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
