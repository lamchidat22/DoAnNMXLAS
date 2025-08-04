import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load models
yolo = YOLO('yolov8n.pt')
gender_model = load_model('Gender_model.h5')

# Class labels
gender_labels = ['Male', 'Female']

# Open video
cap = cv2.VideoCapture('video.mp4')

# Start video processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8
    results = yolo(frame)[0]
    count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # only detect people (COCO class 0 is 'person')
            continue

        count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = frame[y1:y2, x1:x2]

        # Resize and preprocess for gender model
        try:
            face_resized = cv2.resize(face, (256, 256))
        except:
            continue  # skip if invalid resize

        face_input = face_resized / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        # Predict gender
        gender_pred = gender_model.predict(face_input, verbose=0)[0]
        gender = gender_labels[np.argmax(gender_pred)]

        label = f"{gender}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show people count
    cv2.putText(frame, f"People: {count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('YOLOv8 + Gender', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
