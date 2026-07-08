import cv2
import numpy as np
from tensorflow.keras.models import load_model

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
COLORS = [
    (0, 0, 255),    # Angry
    (0, 140, 255),  # Disgust
    (0, 255, 255),  # Fear
    (0, 255, 0),    # Happy
    (255, 255, 0),  # Neutral
    (255, 0, 0),    # Sad
    (255, 0, 255),  # Surprise
]


def load_assets():
    model = load_model('emotion_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade


def annotate_frame(frame, model, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=[0, -1])

        preds = model.predict(roi, verbose=0)[0]
        emotion_idx = np.argmax(preds)
        emotion = EMOTIONS[emotion_idx]
        confidence = preds[emotion_idx]
        color = COLORS[emotion_idx]

        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Emotion label
        label = f"{emotion}: {confidence*100:.1f}%"
        cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+5, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Confidence bars
        bar_x = x + w + 10
        for i, (emo, prob) in enumerate(zip(EMOTIONS, preds)):
            bar_y = y + i * 28
            bar_len = int(prob * 120)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 120, bar_y + 20),
                          (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 20),
                          COLORS[i], -1)
            cv2.putText(frame, f"{emo[:3]} {prob*100:.0f}%",
                        (bar_x + 2, bar_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame
