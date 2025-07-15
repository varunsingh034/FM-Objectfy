import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/mask_detector.h5")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    face = cv2.resize(frame, (224, 224))
    face = np.expand_dims(face / 255.0, axis=0)
    pred = model.predict(face)[0][0]
    label = "With Mask" if pred < 0.5 else "Without Mask"
    color = (0, 255, 0) if pred < 0.5 else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
