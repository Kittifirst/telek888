#!/usr/bin/env python3

import cv2
from ultralytics import YOLO

MODEL_PATH = "/home/kittifirst/teelek/src/teelek/model/best_cabbage.pt"
CAM_ID = 2          # ใส่เลขที่ test แล้วเปิดได้จริง
CONF = 0.8

# โหลดโมเดล
model = YOLO(MODEL_PATH)
print("Class names:", model.names)

# เปิดกล้องแบบบังคับ V4L2 (สำคัญมากบน Ubuntu)
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

# ตั้งค่าความละเอียด
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("✅ Camera opened successfully")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Camera read failed")
        break

    # รัน YOLO
    frame = cv2.resize(frame, (320, 240))
    results = model(frame, conf=CONF, verbose=False, device='cpu')

    # วาด bounding box
    annotated = results[0].plot()

    # Debug จำนวน detection
    if results[0].boxes is not None:
        print("Detected:", len(results[0].boxes))

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            print("Class:", cls_id, "Conf:", round(conf_score, 2))

    cv2.imshow("YOLO Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()