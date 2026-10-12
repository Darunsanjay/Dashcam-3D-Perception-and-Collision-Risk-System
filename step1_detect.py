"""
STEP 1 + 2 - Read dashcam video and detect objects
===================================================
Goal: open a dashcam video, look at it frame by frame, and draw a box
around every car, person, truck, bike, etc. that the AI can see.

How to run (in Terminal, inside your project folder, with the venv active):
    python step1_detect.py

What you should see: a window playing your video with coloured boxes and
labels (e.g. "car 0.91"). Press  q  to quit.
"""

import time                 # used to measure speed (frames per second)
import cv2                  # OpenCV: reads video and draws on images
import torch                # the AI engine YOLO runs on (installed with ultralytics)
from ultralytics import YOLO  # YOLO: a ready-trained object detector

# On Apple Silicon Macs (M1/M2/M3/M4), "mps" uses the Mac's graphics chip,
# which is much faster than the normal processor ("cpu").
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# ---------------------------------------------------------------
# 1. SETTINGS - change these to experiment
# ---------------------------------------------------------------
VIDEO_PATH = "dashcam.mp4"   # put your dashcam video in the same folder
MODEL_NAME = "yolov8n.pt"    # 'n' = nano = smallest & fastest. Downloads automatically the first time.
CONFIDENCE = 0.4             # ignore detections the AI is less than 40% sure about

# The only classes we care about on a road (YOLO knows 80 classes in total).
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

# ---------------------------------------------------------------
# 2. LOAD THE AI MODEL
# ---------------------------------------------------------------
# A "model" is a file of numbers learned from millions of labelled images.
# We don't train anything ourselves yet - we reuse what YOLO already learned.
model = YOLO(MODEL_NAME)

# ---------------------------------------------------------------
# 3. OPEN THE VIDEO
# ---------------------------------------------------------------
# A video is just a stack of still pictures (frames), usually 25-30 per second.
cap = cv2.VideoCapture(VIDEO_PATH)   # use 0 instead of a file name to use a webcam
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}. Check the file name and folder.")

# ---------------------------------------------------------------
# 4. MAIN LOOP - one pass = one frame
# ---------------------------------------------------------------
while True:
    ok, frame = cap.read()           # frame = one picture, as a grid of pixel numbers
    if not ok:                       # no more frames -> video finished
        break

    start = time.time()

    # Ask YOLO: "what objects are in this picture?"
    results = model(frame, conf=CONFIDENCE, device=DEVICE, verbose=False)[0]

    # results.boxes holds every detection. Each one has:
    #   xyxy -> box corners (x1, y1) top-left and (x2, y2) bottom-right, in pixels
    #   conf -> how sure the AI is (0 to 1)
    #   cls  -> which class number (0 = person, 2 = car, ...)
    for box in results.boxes:
        class_name = model.names[int(box.cls)]
        if class_name not in ROAD_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf)

        # Draw the rectangle and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    # Show speed - "real time" means roughly 20+ frames per second
    fps = 1.0 / max(time.time() - start, 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Step 1 - Object detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):   # press q to stop
        break

cap.release()
cv2.destroyAllWindows()
