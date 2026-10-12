"""
STEP A7 - 3D mapping of the whole drive (visual odometry + mapping, "SLAM-lite")
================================================================================
Until now, every frame was analysed on its own: "car #7 is 12 m ahead of me".
But WHERE is "me"? As we drive, our camera moves. Real 3D mapping answers
two questions at the same time:

  1. LOCALISATION -> Where is my camera right now?   (its path = the red line)
  2. MAPPING      -> What does the world around me look like in 3D?
                     (a coloured "point cloud" of road, buildings, kerbs, trees)

Doing both at once is called SLAM (Simultaneous Localisation And Mapping).

HOW WE DO IT, IN 4 IDEAS:
  A. FEATURES:  find ~2000 tiny, easy-to-recognise spots in each picture
                (corners of windows, road markings, sign edges) with ORB.
  B. MATCHING:  find the same spots in the next picture.
  C. MOTION:    the depth AI tells us how far each spot was (in metres).
                Knowing where spots were in 3D, and where they appear now in
                2D, OpenCV's solvePnP works out how the camera moved.
                (Like working out where you're standing from landmarks you recognise.)
  D. MAP:       every few frames, turn every pixel into a 3D point using its
                depth (the Step 5 formula!), move it into "world" position
                using the camera's location, and add it to the map.

We REMOVE cars/people/bikes from both features and map: they move on their
own, so using them would confuse our sense of our own motion and smear
"ghost cars" across the map.

You view the result in RERUN, a free 3D viewer that opens automatically.

How to run:   python stepA7_3d_mapping.py
Install first: pip install rerun-sdk

HONEST LIMITS: this is "SLAM-lite". Real SLAM systems also fix errors by
recognising places they've been before ("loop closure") and by re-optimising
the whole map ("bundle adjustment"). Without those, small errors add up,
so the path slowly DRIFTS. That's normal - and a great thing to observe.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math

import cv2
import numpy as np
import rerun as rr                   # the 3D viewer
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

PROCESS_WIDTH = 640      # shrink frames to this width: much faster, still detailed enough
FRAME_STEP = 2           # use every 2nd frame (the camera moves enough between them to measure)
KEYFRAME_EVERY = 5       # add points to the map every 5 processed frames
MIN_INLIERS = 30         # need at least 30 trustworthy matches to believe a motion estimate
MAX_STEP_M = 3.0         # ignore impossible jumps (> 3 m between two processed frames)
SHRINK = 0.3

# Horizontal field of view in degrees. A wrong value bends the 3D map (straight roads look curved)
CAMERA_FOV_DEG = 80

# Map density: keep 1 pixel in every POINT_STEP, and drop points further than MAX_MAP_DEPTH metres
POINT_STEP = 6
MAX_MAP_DEPTH = 30

# Ignore cars and people when estimating motion and building the map (they move on their own)
MASK_MOVING_OBJECTS = True

# Typical real sizes (width, height, length) in metres, for 3D boxes
REAL_SIZE = {"car": (1.8, 1.5, 4.5), "truck": (2.5, 3.0, 8.0), "bus": (2.5, 3.0, 12.0),
             "person": (0.6, 1.7, 0.6), "bicycle": (0.6, 1.5, 1.8), "motorcycle": (0.8, 1.5, 2.0)}
CLASS_COLOURS = {"car": (0, 180, 255), "truck": (255, 100, 200), "bus": (255, 200, 0),
                 "person": (0, 255, 0), "bicycle": (255, 255, 0), "motorcycle": (255, 0, 255)}  # RGB

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {DEVICE}")
model = YOLO(MODEL_NAME)
print("Loading depth model...")
depth_ai = pipeline("depth-estimation", model=DEPTH_MODEL, device=DEVICE)

orb = cv2.ORB_create(nfeatures=2000)                       # A. the feature finder
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # B. the feature matcher


# ---------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------
def set_frame(i):
    """Tell Rerun which moment in time we're logging (works on old and new versions)."""
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=i)
    else:
        rr.set_time_sequence("frame", i)


def estimate_depth(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = depth_ai(Image.fromarray(rgb))
    depth = result["predicted_depth"].squeeze().float().cpu().numpy()
    h, w = frame.shape[:2]
    return cv2.resize(depth, (w, h))


def distance_in_box(depth, x1, y1, x2, y2):
    bw, bh = x2 - x1, y2 - y1
    patch = depth[y1 + int(bh * SHRINK): y2 - int(bh * SHRINK),
                  x1 + int(bw * SHRINK): x2 - int(bw * SHRINK)]
    return float(np.median(patch)) if patch.size else None


def find_moving_objects(frame):
    """Run YOLO. Return a mask (white = static world, black = objects) and the detections."""
    results = model(frame, conf=CONFIDENCE, device=DEVICE, verbose=False)[0]
    mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
    detections = []
    for box in results.boxes:
        name = model.names[int(box.cls)]
        if name not in ROAD_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        mask[max(y1, 0):y2, max(x1, 0):x2] = 0
        detections.append((name, x1, y1, x2, y2))
    return mask, detections


def estimate_motion(prev, kp, des, K, focal, cx, cy):
    """
    C. How did the camera move between the previous frame and this one?
    Returns ((R, t), n_inliers) where a point in the previous camera's view
    lands at  R @ point + t  in the current camera's view. Or (None, n).
    """
    if prev["des"] is None or des is None or len(prev["kp"]) < 10 or len(kp) < 10:
        return None, 0

    matches = matcher.match(prev["des"], des)              # B. same spots in both pictures
    h, w = prev["depth"].shape
    points_3d, points_2d = [], []
    for m in matches:
        u, v = prev["kp"][m.queryIdx].pt                   # where the spot was (pixels)
        z = prev["depth"][min(int(v), h - 1), min(int(u), w - 1)]  # how far it was (metres)
        if not (1.0 < z < MAX_MAP_DEPTH):
            continue
        points_3d.append([(u - cx) * z / focal, (v - cy) * z / focal, z])  # Step 5 formula, in 3D
        points_2d.append(kp[m.trainIdx].pt)               # where the spot is NOW (pixels)

    if len(points_3d) < MIN_INLIERS:
        return None, len(points_3d)

    # "I know where these landmarks are in 3D, and where I see them now.
    #  Where must my camera be?"  RANSAC ignores the bad matches (outliers).
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.float32(points_3d), np.float32(points_2d), K, None,
        iterationsCount=100, reprojectionError=3.0)
    if not ok or inliers is None or len(inliers) < MIN_INLIERS:
        return None, 0
    if np.linalg.norm(tvec) > MAX_STEP_M:                  # impossible jump -> don't trust it
        return None, len(inliers)

    R, _ = cv2.Rodrigues(rvec)                             # turn the rotation into a 3x3 matrix
    return (R, tvec.ravel()), len(inliers)


def pixels_to_points(depth, rgb, mask, focal, cx, cy):
    """D. Turn (some) pixels into coloured 3D points, in the camera's own view."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:POINT_STEP, 0:w:POINT_STEP]        # a grid of pixel positions
    z = depth[v, u]
    keep = (z > 1.0) & (z < MAX_MAP_DEPTH) & (mask[v, u] > 0)
    u, v, z = u[keep], v[keep], z[keep]
    x = (u - cx) * z / focal                               # right (+) / left (-)
    y = (v - cy) * z / focal                               # down (+) / up (-)
    return np.stack([x, y, z], axis=1), rgb[v, u]


# ---------------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}. Check the file name and folder.")
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

rr.init("Dashcam 3D map", spawn=True)                      # opens the Rerun viewer window
rr.log("world", rr.ViewCoordinates.RDF, static=True)       # x = Right, y = Down, z = Forward

pose = np.eye(4)        # where the camera is in the world (starts at the origin, facing forward)
trajectory = []         # list of camera positions = the red path
prev = None
frame_index, processed = 0, 0
speed_kmh = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_index += 1
    if (frame_index - 1) % FRAME_STEP:
        continue

    # Prepare the frame and the camera numbers (same maths as Step 5)
    scale = PROCESS_WIDTH / frame.shape[1]
    frame = cv2.resize(frame, (PROCESS_WIDTH, int(frame.shape[0] * scale)))
    h, w = frame.shape[:2]
    focal = (w / 2) / math.tan(math.radians(CAMERA_FOV_DEG / 2))
    cx, cy = w / 2, h / 2
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)  # camera matrix
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    depth = estimate_depth(frame)
    object_mask, detections = find_moving_objects(frame)
    use_mask = object_mask if MASK_MOVING_OBJECTS else np.full((h, w), 255, np.uint8)

    # A. find features (only on the static world, if masking is on)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, use_mask)

    # C. update where the camera is
    if prev is not None:
        motion, n_inliers = estimate_motion(prev, kp, des, K, focal, cx, cy)
        if motion is not None:
            R, t = motion
            cur_from_prev = np.eye(4)
            cur_from_prev[:3, :3], cur_from_prev[:3, 3] = R, t
            pose = pose @ np.linalg.inv(cur_from_prev)     # chain this step onto the journey so far
    prev = {"kp": kp, "des": des, "depth": depth}

    R_wc, t_wc = pose[:3, :3], pose[:3, 3]
    trajectory.append(t_wc.copy())

    # Our own speed (a taste of A6!): distance moved / time passed
    if len(trajectory) > 1:
        step_m = np.linalg.norm(trajectory[-1] - trajectory[-2])
        instant = step_m / (FRAME_STEP / video_fps) * 3.6
        speed_kmh = 0.8 * speed_kmh + 0.2 * instant
    if processed % 15 == 0:
        print(f"frame {frame_index:5d} | estimated speed {speed_kmh:5.1f} km/h | map keyframes {processed // KEYFRAME_EVERY}")

    # ---- Send everything to the 3D viewer ----
    set_frame(processed)
    rr.log("world/camera", rr.Transform3D(translation=t_wc, mat3x3=R_wc))
    rr.log("world/camera/image", rr.Pinhole(focal_length=focal, width=w, height=h))
    rr.log("world/camera/image", rr.Image(rgb))
    rr.log("world/trajectory", rr.LineStrips3D([np.array(trajectory)], colors=[(255, 60, 60)]))

    # D. add a slice of the world to the map
    if processed % KEYFRAME_EVERY == 0:
        points_cam, colours = pixels_to_points(depth, rgb, use_mask, focal, cx, cy)
        points_world = points_cam @ R_wc.T + t_wc          # camera view -> world position
        rr.log(f"world/map/keyframe_{processed}",
               rr.Points3D(points_world, colors=colours, radii=0.04), static=True)

    # Objects as 3D boxes, attached to the camera (so they move with it)
    centres, half_sizes, colours, labels = [], [], [], []
    for name, x1, y1, x2, y2 in detections:
        d = distance_in_box(depth, x1, y1, x2, y2)
        if d is None or d > MAX_MAP_DEPTH:
            continue
        ow, oh, ol = REAL_SIZE[name]
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        centres.append([(u - cx) * d / focal, (v - cy) * d / focal, d + ol / 2])
        half_sizes.append([ow / 2, oh / 2, ol / 2])
        colours.append(CLASS_COLOURS[name])
        labels.append(f"{name} {d:.0f}m")
    if centres:
        rr.log("world/camera/objects", rr.Boxes3D(centers=centres, half_sizes=half_sizes,
                                                   colors=colours, labels=labels))
    else:
        rr.log("world/camera/objects", rr.Clear(recursive=False))

    processed += 1

cap.release()
print("Done! Explore the 3D map in the Rerun window (it stays open).")
