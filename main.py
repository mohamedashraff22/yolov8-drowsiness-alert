import cv2
import math
import time
from ultralytics import YOLO
import cvzone

# 1. LOAD THE MODEL
model = YOLO("best.pt")

# 2. SETUP CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# 3. CONFIGURATION (FIXED)
# Instead of hardcoding, we get the names directly from the model
classNames = model.names
print(f"Model Classes Found: {classNames}")  # This will print what your model knows

CONFIDENCE_THRESHOLD = 0.5
DROWSY_FRAMES_THRESHOLD = 15

closed_eyes_counter = 0
alarm_triggered = False

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    eyes_closed_in_frame = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # SAFEGUARD: Ensure the class index exists
            if cls < len(classNames):
                current_class = classNames[cls]
            else:
                current_class = "Unknown"

            if conf > CONFIDENCE_THRESHOLD:
                # LIST OF DANGEROUS STATES
                # Add any label your dataset uses for "Drowsy" here:
                drowsy_labels = [
                    "closed_eye",
                    "eyes closed",
                    "Drowsy",
                    "drowsy",
                    "Sleepy",
                    "Yawn",
                ]

                is_drowsy = current_class in drowsy_labels

                # Red for drowsy, Green for awake
                color = (0, 0, 255) if is_drowsy else (0, 255, 0)

                cvzone.cornerRect(
                    img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=color
                )
                cvzone.putTextRect(
                    img,
                    f"{current_class} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1,
                    offset=3,
                    colorR=color,
                )

                if is_drowsy:
                    eyes_closed_in_frame = True

    # 4. DROWSINESS LOGIC
    if eyes_closed_in_frame:
        closed_eyes_counter += 1
    else:
        closed_eyes_counter = 0
        alarm_triggered = False

    if closed_eyes_counter > DROWSY_FRAMES_THRESHOLD:
        alarm_triggered = True
        cvzone.putTextRect(
            img,
            "DROWSINESS ALERT!",
            (50, 100),
            scale=3,
            thickness=3,
            colorR=(0, 0, 255),
            offset=10,
        )

    cv2.imshow("Smart Car Drowsiness Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
