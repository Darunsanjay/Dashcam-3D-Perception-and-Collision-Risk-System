"""
STEP 6 - Collision risk (Time-To-Collision warnings)
====================================================
Step 4 told us HOW FAR each object is. But distance alone doesn't mean danger:
  - a car 10 m ahead driving at YOUR speed  -> safe (the gap isn't shrinking)
  - a car 30 m ahead that is stopped        -> dangerous (the gap shrinks fast)

What matters is how fast the GAP SHRINKS. So for every tracked object we:
  1. Remember its distance over time               (thanks to tracking IDs from Step 3)
  2. Work out the closing speed                     (metres the gap shrinks per second)
  3. Divide:  TIME-TO-COLLISION = distance / closing speed
        e.g. 12 m away, closing at 6 m/s  ->  12 / 6 = 2 seconds until impact
  4. Turn it into a risk level:  HIGH (red) / MEDIUM (orange) / LOW (green)

We also only worry properly about objects IN OUR PATH (the lane ahead).
A car in the next lane can be close without being a threat.

How to run:   python step6_collision_risk.py      (press q to quit)

REMINDER: a learning project built on AI distance *guesses*. Never use it
to make real driving decisions.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

# ---------------------------------------------------------------
# 1. SETTINGS (same as Step 4)
# ---------------------------------------------------------------
VIDEO_PATH = "dashcam.mp4"
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.4
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_EVERY = 3          # run the depth AI every 3 frames
SHRINK = 0.3             # measure distance in the middle of the box (see Step 4)
NEAR_M = 8               # anything in our path closer than this is at least MEDIUM risk

# ---------------------------------------------------------------
# 2. NEW SETTINGS FOR RISK
# ---------------------------------------------------------------
LOOKBACK = 10            # compare distance now vs 10 frames ago to get the closing speed
MIN_CLOSING_SPEED = 0.5  # gap shrinking slower than 0.5 m/s -> treat as "not approaching"

# Time-to-collision thresholds in seconds: below HIGH_TTC -> HIGH risk, below MEDIUM_TTC -> MEDIUM
HIGH_TTC = 2.0
MEDIUM_TTC = 4.0

# Fraction of the image width (centred) treated as our lane
IN_PATH_WIDTH = 0.4

# Blend factor for new distance readings: 1.0 = no smoothing, lower = smoother but slower to react
SMOOTHING = 0.3

RISK_COLOURS = {"HIGH": (0, 0, 255), "MEDIUM": (0, 165, 255), "LOW": (0, 200, 0)}  # BGR
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {DEVICE}")

model = YOLO(MODEL_NAME)
print("Loading depth model...")
depth_ai = pipeline("depth-estimation", model=DEPTH_MODEL, device=DEVICE)

# Each object's notebook now stores (time in seconds, smoothed distance in metres)
history = defaultdict(lambda: deque(maxlen=60))


# ---------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------
def estimate_depth(frame):
    """Same as Step 4: picture in, grid of metres out."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = depth_ai(Image.fromarray(rgb))
    depth = result["predicted_depth"].squeeze().float().cpu().numpy()
    height, width = frame.shape[:2]
    return cv2.resize(depth, (width, height))


def distance_in_box(depth, x1, y1, x2, y2):
    """Same as Step 4: median depth in the middle of the box."""
    box_w, box_h = x2 - x1, y2 - y1
    patch = depth[y1 + int(box_h * SHRINK): y2 - int(box_h * SHRINK),
                  x1 + int(box_w * SHRINK): x2 - int(box_w * SHRINK)]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def time_to_collision(points):
    """
    Look at the object's notebook and answer: how many seconds until we meet?
    Returns (ttc_seconds or None, closing_speed in m/s).
    """
    if len(points) <= LOOKBACK:
        return None, 0.0                       # not enough history yet

    t_old, d_old = points[-LOOKBACK - 1]       # where it was a moment ago
    t_now, d_now = points[-1]                  # where it is now
    seconds_passed = t_now - t_old
    if seconds_passed <= 0:
        return None, 0.0

    # How many metres did the gap shrink, per second?
    closing_speed = (d_old - d_now) / seconds_passed

    if closing_speed < MIN_CLOSING_SPEED:      # gap steady or growing -> no collision course
        return None, closing_speed

    return d_now / closing_speed, closing_speed   # THE key formula: time = distance / speed


def risk_level(distance, ttc, in_path):
    """Turn the numbers into a simple traffic-light decision."""
    if in_path and ttc is not None and ttc < HIGH_TTC:
        return "HIGH"
    if in_path and ((ttc is not None and ttc < MEDIUM_TTC) or distance < NEAR_M):
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------
# 4. MAIN LOOP
# ---------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}. Check the file name and folder.")

# IMPORTANT: we measure time using the VIDEO's clock, not the computer's.
# Our program runs slower than real time, but a car's speed must be measured
# in real-world seconds - and the video knows how many frames = 1 second.
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"Video records {video_fps:.0f} frames per second")

depth = None
frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    start = time.time()
    frame_h, frame_w = frame.shape[:2]
    video_time = frame_number / video_fps            # seconds since the video started

    results = model.track(frame, conf=CONFIDENCE, device=DEVICE, persist=True,
                          tracker="bytetrack.yaml", verbose=False)[0]

    if depth is None or frame_number % DEPTH_EVERY == 0:
        depth = estimate_depth(frame)

    # "Our path" = a strip in the middle of the picture
    path_left = int(frame_w * (0.5 - IN_PATH_WIDTH / 2))
    path_right = int(frame_w * (0.5 + IN_PATH_WIDTH / 2))
    cv2.line(frame, (path_left, frame_h), (path_left, int(frame_h * 0.45)), (255, 255, 255), 1)
    cv2.line(frame, (path_right, frame_h), (path_right, int(frame_h * 0.45)), (255, 255, 255), 1)

    worst_risk, worst_text = "LOW", "All clear"

    if results.boxes.id is not None:
        for box, track_id in zip(results.boxes, results.boxes.id.int().tolist()):
            class_name = model.names[int(box.cls)]
            if class_name not in ROAD_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_distance = distance_in_box(depth, x1, y1, x2, y2)
            if raw_distance is None:
                continue

            # --- Smooth the distance: blend the new guess with the previous one ---
            points = history[track_id]
            if points:
                distance = SMOOTHING * raw_distance + (1 - SMOOTHING) * points[-1][1]
            else:
                distance = raw_distance
            points.append((video_time, distance))

            # --- The risk maths ---
            ttc, closing_speed = time_to_collision(points)
            centre_x = (x1 + x2) // 2
            in_path = path_left <= centre_x <= path_right
            risk = risk_level(distance, ttc, in_path)

            # Remember the most dangerous object for the top banner
            if RISK_ORDER[risk] > RISK_ORDER[worst_risk]:
                worst_risk = risk
                worst_text = (f"{class_name} #{track_id} in {ttc:.1f} s" if ttc is not None
                              else f"{class_name} #{track_id} only {distance:.0f} m ahead")

            # Draw the box and label, e.g. "car #7 12m  TTC 2.4s"
            colour = RISK_COLOURS[risk]
            label = f"{class_name} #{track_id} {distance:.0f}m"
            if ttc is not None:
                label += f"  TTC {ttc:.1f}s"
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3 if risk == "HIGH" else 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    # --- Top banner showing the overall risk ---
    cv2.rectangle(frame, (0, 0), (frame_w, 40), RISK_COLOURS[worst_risk], -1)
    cv2.putText(frame, f"RISK: {worst_risk}  -  {worst_text}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    fps = 1.0 / max(time.time() - start, 1e-6)
    cv2.putText(frame, f"FPS {fps:.0f}", (frame_w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Step 6 - Collision risk", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_number += 1

cap.release()
cv2.destroyAllWindows()
