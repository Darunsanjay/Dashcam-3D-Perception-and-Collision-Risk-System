"""
STEP 7 - The dashboard (everything together on one screen)
===========================================================
This is the finished system. It combines every step you built:

  Step 1-2  YOLO finds objects
  Step 3    ByteTrack gives each one an ID
  Step 4    The depth AI estimates distance in metres
  Step 5    Camera maths places each object on a bird's-eye map
  Step 6    Time-To-Collision turns movement into a risk level

The screen layout:
  +-----------------------------+--------------+
  |  CAMERA VIEW                |  BIRD'S-EYE  |
  |  boxes coloured by risk     |  MAP         |
  +-----------------------------+--------------+
  |  RISK BANNER  +  stats  +  risk timeline (last 10 s) |
  +--------------------------------------------+

ONE UPGRADE over Step 6: "our path" is now measured in METRES on the road
(e.g. within 1.8 m left/right of our car), instead of a strip of pixels.
That's only possible because Step 5 gave us real 3D positions!

How to run:   python step7_dashboard.py      (press q to quit)
Set SAVE_VIDEO = True to record the dashboard to "dashboard_output.mp4".

REMINDER: a learning project built on AI distance *guesses*. Never use it
to make real driving decisions.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

# ---------------------------------------------------------------
# 1. SETTINGS (same meaning as in the earlier steps)
# ---------------------------------------------------------------
VIDEO_PATH = "dashcam.mp4"
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.4
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

DEPTH_EVERY = 3          # run the depth model every 3 frames (Step 4)
SHRINK = 0.3             # measure depth in the middle of each box (Step 4)
CAMERA_FOV_DEG = 80      # camera field of view in degrees (Step 5)
MAX_RANGE_M = 50         # ignore objects further than this (Step 5)
PIXELS_PER_M = 10        # map scale (Step 5)
HIGH_TTC = 2.0           # seconds to impact for HIGH risk (Step 6)
MEDIUM_TTC = 4.0         # seconds to impact for MEDIUM risk (Step 6)
SMOOTHING = 0.3          # distance smoothing (Step 6)
NEAR_M = 8
LOOKBACK = 10
MIN_CLOSING_SPEED = 0.5

# An object counts as in our lane if it is within this many metres to our left or right
LANE_HALF_WIDTH = 1.8

# Set to True to record the dashboard to dashboard_output.mp4
SAVE_VIDEO = False

# How many seconds of history the risk timeline shows
TIMELINE_SECONDS = 10

# Screen layout sizes (pixels)
VIEW_H = 600                 # camera view height (the video is resized to this)
MAP_W, MAP_H = 400, VIEW_H   # bird's-eye map, same height so they sit side by side
PANEL_H = 130                # bottom info panel
EGO_X, EGO_Y = MAP_W // 2, MAP_H - 30

RISK_COLOURS = {"HIGH": (0, 0, 255), "MEDIUM": (0, 165, 255), "LOW": (0, 200, 0)}
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
REAL_SIZE = {"car": (1.8, 4.5), "truck": (2.5, 8.0), "bus": (2.5, 12.0),
             "person": (0.6, 0.6), "bicycle": (0.6, 1.8), "motorcycle": (0.8, 2.0)}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {DEVICE}")
model = YOLO(MODEL_NAME)
print("Loading depth model...")
depth_ai = pipeline("depth-estimation", model=DEPTH_MODEL, device=DEVICE)


# ---------------------------------------------------------------
# 2. HELPER FUNCTIONS (all from earlier steps)
# ---------------------------------------------------------------
def estimate_depth(frame):                                   # Step 4
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = depth_ai(Image.fromarray(rgb))
    depth = result["predicted_depth"].squeeze().float().cpu().numpy()
    h, w = frame.shape[:2]
    return cv2.resize(depth, (w, h))


def distance_in_box(depth, x1, y1, x2, y2):                  # Step 4
    bw, bh = x2 - x1, y2 - y1
    patch = depth[y1 + int(bh * SHRINK): y2 - int(bh * SHRINK),
                  x1 + int(bw * SHRINK): x2 - int(bw * SHRINK)]
    return float(np.median(patch)) if patch.size else None


def focal_length_pixels(frame_width):                        # Step 5
    return (frame_width / 2) / math.tan(math.radians(CAMERA_FOV_DEG / 2))


def pixel_to_road(pixel_x, distance, frame_width, focal):    # Step 5
    return (pixel_x - frame_width / 2) * distance / focal, distance


def road_to_map(sideways_m, ahead_m):                        # Step 5
    return int(EGO_X + sideways_m * PIXELS_PER_M), int(EGO_Y - ahead_m * PIXELS_PER_M)


def time_to_collision(points):                               # Step 6
    if len(points) <= LOOKBACK:
        return None
    t_old, d_old = points[-LOOKBACK - 1]
    t_now, d_now = points[-1]
    if t_now - t_old <= 0:
        return None
    closing_speed = (d_old - d_now) / (t_now - t_old)
    return d_now / closing_speed if closing_speed >= MIN_CLOSING_SPEED else None


def risk_level(distance, ttc, in_path):                      # Step 6
    if in_path and ttc is not None and ttc < HIGH_TTC:
        return "HIGH"
    if in_path and ((ttc is not None and ttc < MEDIUM_TTC) or distance < NEAR_M):
        return "MEDIUM"
    return "LOW"


def draw_empty_map(focal, frame_width):                      # Step 5 + lane lines
    canvas = np.full((MAP_H, MAP_W, 3), 30, dtype=np.uint8)
    for metres in range(10, 200, 10):
        y = EGO_Y - metres * PIXELS_PER_M
        if y < 0:
            break
        cv2.line(canvas, (0, y), (MAP_W, y), (70, 70, 70), 1)
        cv2.putText(canvas, f"{metres} m", (5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    for edge_pixel in (0, frame_width):                      # camera view cone
        cv2.line(canvas, (EGO_X, EGO_Y), road_to_map(*pixel_to_road(edge_pixel, 200, frame_width, focal)),
                 (90, 90, 90), 1)
    for side in (-LANE_HALF_WIDTH, LANE_HALF_WIDTH):         # NEW: our lane, in metres
        x = road_to_map(side, 0)[0]
        cv2.line(canvas, (x, 0), (x, EGO_Y), (200, 200, 200), 1)
    w, l = 1.8 * PIXELS_PER_M, 4.5 * PIXELS_PER_M
    cv2.rectangle(canvas, (int(EGO_X - w / 2), EGO_Y), (int(EGO_X + w / 2), int(EGO_Y + l)), (255, 255, 255), -1)
    cv2.putText(canvas, "YOU", (EGO_X - 14, MAP_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return canvas


def draw_panel(width, worst_risk, worst_text, n_objects, closest, fps, timeline):
    """The bottom strip: banner, numbers, and the risk timeline."""
    panel = np.full((PANEL_H, width, 3), 20, dtype=np.uint8)

    # Big coloured risk banner
    cv2.rectangle(panel, (0, 0), (width, 40), RISK_COLOURS[worst_risk], -1)
    cv2.putText(panel, f"RISK: {worst_risk}  -  {worst_text}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Stats line
    closest_text = f"{closest:.1f} m" if closest is not None else "-"
    cv2.putText(panel, f"Objects: {n_objects}     Closest in lane: {closest_text}     FPS: {fps:.0f}",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    # Risk timeline: one thin bar per frame, taller + redder = more risk
    cv2.putText(panel, f"Last {TIMELINE_SECONDS} s", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    left, right, bottom, tallest = 110, width - 10, PANEL_H - 10, 40
    if timeline:
        bar_w = (right - left) / timeline.maxlen
        for i, risk in enumerate(timeline):
            height = int(tallest * (RISK_ORDER[risk] + 1) / 3)
            x = int(left + i * bar_w)
            cv2.rectangle(panel, (x, bottom - height), (int(x + max(bar_w, 1)), bottom), RISK_COLOURS[risk], -1)
    return panel


# ---------------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}. Check the file name and folder.")
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

history = defaultdict(lambda: deque(maxlen=60))              # each object's (time, distance) notebook
timeline = deque(maxlen=int(TIMELINE_SECONDS * video_fps))   # worst risk of each frame
writer = None
depth = None
frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    start = time.time()

    # Shrink the video to VIEW_H pixels tall: fits the screen AND makes the AIs faster
    scale = VIEW_H / frame.shape[0]
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), VIEW_H))
    frame_w = frame.shape[1]
    focal = focal_length_pixels(frame_w)
    video_time = frame_number / video_fps

    results = model.track(frame, conf=CONFIDENCE, device=DEVICE, persist=True,
                          tracker="bytetrack.yaml", verbose=False)[0]
    if depth is None or frame_number % DEPTH_EVERY == 0:
        depth = estimate_depth(frame)

    bird_map = draw_empty_map(focal, frame_w)
    worst_risk, worst_text = "LOW", "All clear"
    n_objects, closest = 0, None

    if results.boxes.id is not None:
        for box, track_id in zip(results.boxes, results.boxes.id.int().tolist()):
            class_name = model.names[int(box.cls)]
            if class_name not in ROAD_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_distance = distance_in_box(depth, x1, y1, x2, y2)
            if raw_distance is None or raw_distance > MAX_RANGE_M:
                continue
            n_objects += 1

            # Step 6: smooth the distance and remember it
            points = history[track_id]
            distance = SMOOTHING * raw_distance + (1 - SMOOTHING) * points[-1][1] if points else raw_distance
            points.append((video_time, distance))

            # Step 5: where is it on the road?
            sideways_m, ahead_m = pixel_to_road((x1 + x2) / 2, distance, frame_w, focal)
            in_path = abs(sideways_m) < LANE_HALF_WIDTH          # the 3D upgrade!

            # Step 6: how risky?
            ttc = time_to_collision(points)
            risk = risk_level(distance, ttc, in_path)
            colour = RISK_COLOURS[risk]

            if in_path and (closest is None or distance < closest):
                closest = distance
            if RISK_ORDER[risk] > RISK_ORDER[worst_risk]:
                worst_risk = risk
                worst_text = (f"{class_name} #{track_id} in {ttc:.1f} s" if ttc is not None
                              else f"{class_name} #{track_id} only {distance:.0f} m ahead")

            # Draw on the camera view
            label = f"{class_name} #{track_id} {distance:.0f}m" + (f" TTC {ttc:.1f}s" if ttc is not None else "")
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3 if risk == "HIGH" else 2)
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            # Draw on the map (same colour = same risk)
            obj_w, obj_l = REAL_SIZE[class_name]
            cv2.rectangle(bird_map, road_to_map(sideways_m - obj_w / 2, ahead_m + obj_l),
                          road_to_map(sideways_m + obj_w / 2, ahead_m), colour, -1)
            label_x, label_y = road_to_map(sideways_m + obj_w / 2, ahead_m)
            cv2.putText(bird_map, f"#{track_id}", (label_x + 3, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    timeline.append(worst_risk)
    fps = 1.0 / max(time.time() - start, 1e-6)

    # Glue the three pieces into one picture
    top_row = np.hstack([frame, bird_map])
    panel = draw_panel(top_row.shape[1], worst_risk, worst_text, n_objects, closest, fps, timeline)
    dashboard = np.vstack([top_row, panel])

    if SAVE_VIDEO:
        if writer is None:
            writer = cv2.VideoWriter("dashboard_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                     video_fps, (dashboard.shape[1], dashboard.shape[0]))
        writer.write(dashboard)

    cv2.imshow("Dashcam Perception & Collision Risk", dashboard)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_number += 1

cap.release()
if writer is not None:
    writer.release()
    print("Saved dashboard_output.mp4")
cv2.destroyAllWindows()
