"""
STEP 3 - Track objects (give every car its own ID)
==================================================
Problem: in Step 1, YOLO looked at every frame separately. It had no idea
that the car in frame 10 is the SAME car as in frame 11.

Solution: a TRACKER. It matches each new box to the closest box from the
previous frame, so every object keeps a stable ID ("car #7") while it is
on screen. Once we can follow an object over time, we can measure how it
MOVES - and movement is what collision risk is all about.

New things you will see on screen:
  - "car #7"      -> the object's tracking ID
  - a trail line  -> the path the object took over the last second
  - APPROACHING   -> the box is growing fast, i.e. the object is getting closer

How to run:   python step3_track.py      (press q to quit)
"""

import time
from collections import defaultdict, deque   # handy containers for storing history

import cv2
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------
VIDEO_PATH = "dashcam.mp4"
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.4
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

TRAIL_LENGTH = 30        # remember the last 30 positions (about 1 second of video)
LOOKBACK = 10            # compare box size now vs 10 frames ago
GROWTH_WARNING = 1.15    # box 15% taller than 10 frames ago -> "approaching"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {DEVICE}")

model = YOLO(MODEL_NAME)

# ---------------------------------------------------------------
# 2. MEMORY FOR EACH OBJECT
# ---------------------------------------------------------------
# history[7] will hold the recent positions of object #7.
# Each entry is (centre_x, bottom_y, box_height).
# A deque with maxlen automatically throws away the oldest entry when full.
history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))


def colour_for_id(track_id):
    """Give every ID its own fixed colour so you can follow it by eye."""
    return ((track_id * 37) % 255, (track_id * 17) % 255, (track_id * 91) % 255)


# ---------------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}. Check the file name and folder.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    start = time.time()

    # The only big change from Step 1: model.track() instead of model(...)
    #   persist=True  -> "remember the objects from the previous frame"
    #   bytetrack     -> the tracking method (a popular, fast one)
    # Tracker options: "bytetrack.yaml" (fast, used here) or "botsort.yaml"
    results = model.track(frame, conf=CONFIDENCE, device=DEVICE, persist=True,
                          tracker="bytetrack.yaml", verbose=False)[0]

    # On some frames nothing is tracked yet, so the IDs can be missing
    if results.boxes.id is not None:
        track_ids = results.boxes.id.int().tolist()

        for box, track_id in zip(results.boxes, track_ids):
            class_name = model.names[int(box.cls)]
            if class_name not in ROAD_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            centre_x = (x1 + x2) // 2
            bottom_y = y2                 # bottom of the box = where the object touches the road
            height = y2 - y1              # bigger box = closer object (we'll use this idea a lot!)

            # Save this frame's position into the object's memory
            history[track_id].append((centre_x, bottom_y, height))
            points = history[track_id]

            # --- Is it getting closer? ---
            # Objects look bigger as they get nearer. If the box grew a lot
            # in the last LOOKBACK frames, the gap between us is shrinking.
            approaching = False
            if len(points) > LOOKBACK:
                old_height = points[-LOOKBACK - 1][2]
                if old_height > 0 and height / old_height > GROWTH_WARNING:
                    approaching = True

            colour = (0, 0, 255) if approaching else colour_for_id(track_id)
            label = f"{class_name} #{track_id}" + ("  APPROACHING" if approaching else "")

            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            # Draw the trail: a line joining the object's recent positions
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1][:2], points[i][:2], colour, 2)

    fps = 1.0 / max(time.time() - start, 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Step 3 - Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
