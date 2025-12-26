import cv2
import math
import time
from ultralytics import YOLO
import cvzone
import paho.mqtt.client as mqtt

# SETUP

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "project/driver_sleep_system"


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ SUCCESS: Connected to MQTT Broker!")
    else:
        print(f"❌ FAILED: Connection failed with code {rc}")


client = mqtt.Client()
client.on_connect = on_connect

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
except Exception as e:
    print(f"❌ Error connecting to MQTT: {e}")

# MODEL & CAMERA

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

classNames = model.names
drowsy_labels = ["drowsy", "Drowsy", "closed_eye", "eyes closed", "sleepy", "yawn"]

CONFIDENCE_THRESHOLD = 0.5

# --- SETTINGS FOR SAFETY ---
DROWSY_TRIGGER_FRAMES = 10  # How fast to trigger alarm (approx 0.5 sec)
WAKEUP_CONFIRM_FRAMES = (
    25  # How long eyes must be OPEN to stop alarm (approx 1-1.5 sec)
)

# --- STATE VARIABLES ---
counter_eyes_closed = 0  # Counts consecutive closed frames
counter_eyes_open = 0  # Counts consecutive open frames
alarm_is_active = False  # Keeps track of the current system state
last_sent_state = None

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True, verbose=False)

    # We assume eyes are OPEN unless detected otherwise in this frame
    is_currently_closed = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > CONFIDENCE_THRESHOLD:
                current_class = classNames[cls]

                # Check if this class is in our drowsy list
                if current_class in drowsy_labels:
                    is_currently_closed = True
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Green

                # Draw bounding box
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

    # SAFETY LOGIC (HYSTERESIS)

    if is_currently_closed:
        counter_eyes_closed += 1
        counter_eyes_open = 0  # Reset open counter because eyes are closed
    else:
        counter_eyes_open += 1
        counter_eyes_closed = 0  # Reset closed counter because eyes are open

    # --- TRIGGERING THE ALARM ---
    # If eyes closed longer than threshold -> Turn ON Alarm
    if counter_eyes_closed > DROWSY_TRIGGER_FRAMES:
        alarm_is_active = True

    # --- STOPPING THE ALARM ---
    # Only turn OFF alarm if alarm is currently ON AND eyes have been open for a while
    # This prevents the "flicker" issue where 1 frame of detection failure restarts the car
    if alarm_is_active and counter_eyes_open > WAKEUP_CONFIRM_FRAMES:
        alarm_is_active = False

    # SEND TO CAR (MQTT)

    if alarm_is_active:
        cvzone.putTextRect(
            img,
            "ALARM! STOPPING CAR",
            (50, 100),
            scale=3,
            thickness=3,
            colorR=(0, 0, 255),
            offset=10,
        )
        current_mqtt_msg = "1"
    else:
        current_mqtt_msg = "0"

    # Optimization: Re-send signal periodically if it's "1" (STOP)
    # to ensure the car doesn't miss the stop command due to network lag.
    # But for state change, we stick to the basic logic:

    if current_mqtt_msg != last_sent_state:
        client.publish(MQTT_TOPIC, current_mqtt_msg)
        last_sent_state = current_mqtt_msg

        if current_mqtt_msg == "1":
            print("🔴 URGENT: Driver Asleep -> Sending STOP")
        else:
            print("🟢 SAFE: Driver Awake -> Sending GO")

    # Optional: Visual Debug bars
    # cv2.putText(img, f"Closed: {counter_eyes_closed}", (50, 600), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    # cv2.putText(img, f"Open: {counter_eyes_open}", (50, 640), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    cv2.imshow("Driver Monitoring System", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

client.loop_stop()
client.disconnect()
cap.release()
cv2.destroyAllWindows()
