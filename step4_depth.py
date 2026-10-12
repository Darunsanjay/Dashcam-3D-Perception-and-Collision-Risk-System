"""
STEP 4 - Estimate distance in metres (depth estimation)
=======================================================
Problem: one camera can't measure distance. Your two eyes can (each eye sees
a slightly different picture), but a dashcam has only one "eye".

Solution: a DEPTH AI. It has studied millions of road photos and learned
clues humans use too: things higher up the picture are further away, roads
get narrower in the distance, far cars look small and blurry. From those
clues it guesses a distance, in metres, for EVERY pixel of the picture.

That guess is called a DEPTH MAP. We then look inside each car's box,
read the depth there, and write the distance on screen: "car #7  12.3 m".

You will see TWO windows:
  1. Your video with boxes coloured by distance (red = near, orange = medium, green = far)
  2. The depth map itself (bright = near, dark = far)

How to run:   python step4_depth.py      (press q to quit)
First run downloads the depth model (about 100 MB) - be patient.

IMPORTANT: these are AI *estimates*, not measurements. They can be off by
several metres. Great for learning - never for real driving decisions.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # lets the Mac chip fall back to CPU for rare operations

import time

import cv2
import numpy as np                  # NumPy: fast maths on big grids of numbers
import torch
from PIL import Image               # converts pictures into the format the depth AI wants
from transformers import pipeline   # Hugging Face: easy access to thousands of AI models
from ultralytics import YOLO

# ---------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------
VIDEO_PATH = "dashcam.mp4"
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.4
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

# The depth AI: "Metric" = answers in real metres, "Outdoor" = trained on
# road scenes, "Small" = the fastest size.
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

# The depth model is slow, so it runs every N frames and the last map is reused in between
DEPTH_EVERY = 3

# Box colours: red if closer than NEAR_M metres, orange if closer than MID_M, green otherwise
NEAR_M = 10
MID_M = 20

# Fraction trimmed off each side of a box before reading the depth, so the edges
# (road and sky behind the object) don't skew the result. Must stay below 0.5
SHRINK = 0.3

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# ---------------------------------------------------------------
# 2. LOAD BOTH AI MODELS
# ---------------------------------------------------------------
model = YOLO(MODEL_NAME)                       # AI #1: finds objects (from Step 1)
print("Loading depth model (first time downloads ~100 MB)...")
depth_ai = pipeline("depth-estimation", model=DEPTH_MODEL, device=DEVICE)  # AI #2: guesses distance


# ---------------------------------------------------------------
# 3. HELPER FUNCTIONS (small machines that each do one job)
# ---------------------------------------------------------------
def estimate_depth(frame):
    """Give the depth AI one picture, get back a grid of distances in metres."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # OpenCV stores colours as BGR; the AI wants RGB
    result = depth_ai(Image.fromarray(rgb))
    depth = result["predicted_depth"].squeeze().float().cpu().numpy()
    height, width = frame.shape[:2]
    return cv2.resize(depth, (width, height))         # same size as the video, so pixels line up


def distance_in_box(depth, x1, y1, x2, y2):
    """Read the distance inside a box: take the middle part, then the median."""
    box_w, box_h = x2 - x1, y2 - y1
    inner_x1 = x1 + int(box_w * SHRINK)
    inner_x2 = x2 - int(box_w * SHRINK)
    inner_y1 = y1 + int(box_h * SHRINK)
    inner_y2 = y2 - int(box_h * SHRINK)

    patch = depth[inner_y1:inner_y2, inner_x1:inner_x2]  # cut out that rectangle of the depth map
    if patch.size == 0:
        return None
    # MEDIAN = the middle value when sorted. Unlike the average, a few
    # weird pixels (a bit of sky) can't drag it far off.
    return float(np.median(patch))


def colour_for_distance(distance):
    """Traffic-light colours (remember: OpenCV order is Blue, Green, Red)."""
    if distance < NEAR_M:
        return (0, 0, 255)       # red
    if distance < MID_M:
        return (0, 165, 255)     # orange
    return (0, 200, 0)           # green


def depth_to_picture(depth):
    """Turn the grid of metres into a colourful picture so humans can see it."""
    near_is_big = 1.0 / np.clip(depth, 0.5, None)       # flip it: near -> big number -> bright
    scaled = cv2.normalize(near_is_big, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    picture = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    return cv2.resize(picture, None, fx=0.5, fy=0.5)    # half size so both windows fit on screen


# ---------------------------------------------------------------
# 4. MAIN LOOP
# ---------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}. Check the file name and folder.")

depth = None          # no depth map yet
frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    start = time.time()

    # AI #1: find and track objects (same as Step 3, using ByteTrack)
    results = model.track(frame, conf=CONFIDENCE, device=DEVICE, persist=True,
                          tracker="bytetrack.yaml", verbose=False)[0]

    # AI #2: refresh the depth map every DEPTH_EVERY frames
    if depth is None or frame_number % DEPTH_EVERY == 0:
        depth = estimate_depth(frame)
    frame_number += 1

    if results.boxes.id is not None:
        for box, track_id in zip(results.boxes, results.boxes.id.int().tolist()):
            class_name = model.names[int(box.cls)]
            if class_name not in ROAD_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            distance = distance_in_box(depth, x1, y1, x2, y2)
            if distance is None:
                continue

            colour = colour_for_distance(distance)
            label = f"{class_name} #{track_id}  {distance:.1f} m"

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    fps = 1.0 / max(time.time() - start, 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Step 4 - Distance", frame)
    cv2.imshow("Step 4 - Depth map (bright = near)", depth_to_picture(depth))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
