"""
STEP 5 - Bird's-eye map (from 2D picture to 3D positions)
=========================================================
So far every object has:
  - a box in the PICTURE (pixels)        -> from YOLO (Step 1)
  - a distance in METRES                  -> from the depth AI (Step 4)

Now we combine them to answer: WHERE exactly is it on the road?
  - Z = how far AHEAD of us (metres)       -> the depth
  - X = how far LEFT or RIGHT of us (metres) -> worked out with camera maths

Then we draw a top-down map, like looking down from a drone:
  - our car at the bottom-middle
  - every object as a rectangle at its real position
  - grid lines every 10 metres

THE CAMERA MATHS (one formula!):
    X = (pixel_x - centre_of_picture) * Z / focal_length

Why? An object 100 pixels right of centre could be 0.5 m to the right (if
it's close) or 5 m to the right (if it's far). Multiplying by the distance
Z fixes that. The focal length is how "zoomed in" the camera is - we
estimate it from the camera's field of view (how wide it sees).

How to run:   python step5_birdseye.py      (press q to quit)
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

# ---------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------
VIDEO_PATH = "dashcam.mp4"
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.4
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_EVERY = 3
SHRINK = 0.3

# Horizontal field of view of the camera in degrees (an estimate; real dashcams are often 100-140)
CAMERA_FOV_DEG = 80

# Map scale: how many map pixels make one metre
PIXELS_PER_M = 10

# Depth estimates get unreliable far away, so objects beyond this are left off the map
MAX_RANGE_M = 50

# Map canvas size (pixels) and where OUR car sits on it
MAP_W, MAP_H = 400, 600
EGO_X, EGO_Y = MAP_W // 2, MAP_H - 30

# Typical real sizes in metres (width, length) so shapes look right on the map
REAL_SIZE = {"car": (1.8, 4.5), "truck": (2.5, 8.0), "bus": (2.5, 12.0),
             "person": (0.6, 0.6), "bicycle": (0.6, 1.8), "motorcycle": (0.8, 2.0)}

# One colour per type of object (BGR)
CLASS_COLOURS = {"car": (255, 180, 0), "truck": (200, 100, 255), "bus": (0, 200, 255),
                 "person": (0, 255, 0), "bicycle": (255, 255, 0), "motorcycle": (255, 0, 255)}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {DEVICE}")
model = YOLO(MODEL_NAME)
print("Loading depth model...")
depth_ai = pipeline("depth-estimation", model=DEPTH_MODEL, device=DEVICE)


# ---------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------
def estimate_depth(frame):
    """Same as Step 4."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = depth_ai(Image.fromarray(rgb))
    depth = result["predicted_depth"].squeeze().float().cpu().numpy()
    h, w = frame.shape[:2]
    return cv2.resize(depth, (w, h))


def distance_in_box(depth, x1, y1, x2, y2):
    """Same as Step 4."""
    bw, bh = x2 - x1, y2 - y1
    patch = depth[y1 + int(bh * SHRINK): y2 - int(bh * SHRINK),
                  x1 + int(bw * SHRINK): x2 - int(bw * SHRINK)]
    return float(np.median(patch)) if patch.size else None


def focal_length_pixels(frame_width):
    """
    How 'zoomed in' the camera is, in pixels.
    Picture half the image width as one side of a triangle, and half the
    field of view as its angle: tan(angle) = opposite / adjacent.
    """
    half_fov = math.radians(CAMERA_FOV_DEG / 2)
    return (frame_width / 2) / math.tan(half_fov)


def pixel_to_road(pixel_x, distance, frame_width, focal):
    """THE 3D step: picture position + distance -> metres left/right and ahead."""
    centre = frame_width / 2
    sideways_m = (pixel_x - centre) * distance / focal   # X: negative = left, positive = right
    ahead_m = distance                                     # Z: forward
    return sideways_m, ahead_m


def road_to_map(sideways_m, ahead_m):
    """Metres on the road -> pixels on our map picture."""
    map_x = int(EGO_X + sideways_m * PIXELS_PER_M)
    map_y = int(EGO_Y - ahead_m * PIXELS_PER_M)    # minus, because 'up' on the map = further ahead
    return map_x, map_y


def draw_empty_map(focal, frame_width):
    """Dark background, distance rings, the camera's view cone, and our own car."""
    canvas = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # Grid line every 10 m ahead, with a label
    for metres in range(10, 200, 10):
        y = EGO_Y - metres * PIXELS_PER_M
        if y < 0:
            break
        cv2.line(canvas, (0, y), (MAP_W, y), (70, 70, 70), 1)
        cv2.putText(canvas, f"{metres} m", (5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # The camera only sees inside a cone: draw its two edges
    far = 200
    for edge_pixel in (0, frame_width):
        sx, sz = pixel_to_road(edge_pixel, far, frame_width, focal)
        cv2.line(canvas, (EGO_X, EGO_Y), road_to_map(sx, sz), (90, 90, 90), 1)

    # Our own car (1.8 m wide, 4.5 m long), just below the camera point
    w, l = 1.8 * PIXELS_PER_M, 4.5 * PIXELS_PER_M
    cv2.rectangle(canvas, (int(EGO_X - w / 2), EGO_Y), (int(EGO_X + w / 2), int(EGO_Y + l)),
                  (255, 255, 255), -1)
    cv2.putText(canvas, "YOU", (EGO_X - 14, MAP_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return canvas


# ---------------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}. Check the file name and folder.")

depth = None
frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    start = time.time()
    frame_w = frame.shape[1]
    focal = focal_length_pixels(frame_w)

    results = model.track(frame, conf=CONFIDENCE, device=DEVICE, persist=True,
                          tracker="bytetrack.yaml", verbose=False)[0]
    if depth is None or frame_number % DEPTH_EVERY == 0:
        depth = estimate_depth(frame)

    bird_map = draw_empty_map(focal, frame_w)

    if results.boxes.id is not None:
        for box, track_id in zip(results.boxes, results.boxes.id.int().tolist()):
            class_name = model.names[int(box.cls)]
            if class_name not in ROAD_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            distance = distance_in_box(depth, x1, y1, x2, y2)
            if distance is None or distance > MAX_RANGE_M:
                continue

            # --- From picture to road ---
            centre_x = (x1 + x2) / 2
            sideways_m, ahead_m = pixel_to_road(centre_x, distance, frame_w, focal)

            # --- Draw on the map: a rectangle the object's real size ---
            # The camera sees the BACK of a car, so the rectangle starts at
            # the measured distance and stretches further away by its length.
            obj_w, obj_l = REAL_SIZE[class_name]
            top_left = road_to_map(sideways_m - obj_w / 2, ahead_m + obj_l)
            bottom_right = road_to_map(sideways_m + obj_w / 2, ahead_m)
            colour = CLASS_COLOURS[class_name]
            cv2.rectangle(bird_map, top_left, bottom_right, colour, -1)
            cv2.putText(bird_map, f"#{track_id}", (bottom_right[0] + 3, bottom_right[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

            # Same ID on the video, so you can match map dots to cars
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{class_name} #{track_id}  {ahead_m:.0f}m ahead, {sideways_m:+.1f}m side",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

    fps = 1.0 / max(time.time() - start, 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Step 5 - Camera view", frame)
    cv2.imshow("Step 5 - Bird's-eye map", bird_map)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_number += 1

cap.release()
cv2.destroyAllWindows()
