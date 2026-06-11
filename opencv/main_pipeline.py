import math
import os

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# --- 1. System Initialization ---
print("Initializing Security Gate Pipeline...")
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print("Error loading YOLO. Did you run pip install ultralytics?")
    exit()

# Setup Tracking Memory
active_tracks = {}
next_id = 1


def get_centroid(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# Load the base image to simulate a video feed
img_path = "sample_car.jpg"
if not os.path.exists(img_path):
    print(f"Error: {img_path} missing.")
    exit()
base_img = cv2.imread(img_path)

# --- 2. The Main Pipeline Loop ---
# We simulate a 10-frame video by shifting the image 20 pixels to the right each frame
print("Starting Video Stream Processing...")

for frame_num in range(10):
    # --- A. Frame Acquisition ---
    # Create a blank frame, then paste the car slightly shifted to simulate movement
    frame = np.zeros_like(base_img)
    shift = frame_num * 20
    # Ensure we don't slice out of bounds
    w = base_img.shape[1] - shift
    frame[:, shift : shift + w] = base_img[:, 0:w]

    # We will draw our dashboard on this copy
    display_frame = frame.copy()
    current_frame_tracks = {}

    # --- B. Object Detection ---
    results = model(frame, conf=0.5, verbose=False)
    detections = results[0].boxes

    for box in detections:
        class_id = int(box.cls[0])
        # We only care about cars/trucks (COCO class 2 or 7)
        if class_id not in [2, 7]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centroid = get_centroid(x1, y1, x2, y2)

        # --- C. Tracking Logic ---
        assigned_id = None
        for track_id, prev_centroid in active_tracks.items():
            dist = math.hypot(
                centroid[0] - prev_centroid[0], centroid[1] - prev_centroid[1]
            )
            if dist < 60:  # Match found
                assigned_id = track_id

                # Rule Check: Speeding
                if dist > 40:
                    cv2.putText(
                        display_frame,
                        "ALERT: SPEEDING!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )

                del active_tracks[track_id]
                break

        if assigned_id is None:
            assigned_id = next_id
            next_id += 1

        current_frame_tracks[assigned_id] = centroid

        # --- D. Crop & Preprocess for OCR ---
        # Heuristic: The license plate is usually in the bottom middle of the car box.
        plate_x1 = x1 + int((x2 - x1) * 0.3)
        plate_x2 = x1 + int((x2 - x1) * 0.7)
        plate_y1 = y1 + int((y2 - y1) * 0.7)
        plate_y2 = y2

        # Ensure coordinates are within image bounds
        plate_x1, plate_y1 = max(0, plate_x1), max(0, plate_y1)
        plate_x2, plate_y2 = (
            min(frame.shape[1], plate_x2),
            min(frame.shape[0], plate_y2),
        )

        plate_crop = frame[plate_y1:plate_y2, plate_x1:plate_x2]

        plate_text = "Reading..."
        if plate_crop.size > 0:
            # Preprocess
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

            # OCR
            # In a real app we'd run this less frequently to save CPU, but we run it here for demo
            raw_text = pytesseract.image_to_string(binary, config="--psm 8").strip()
            # Clean non-alphanumeric
            plate_text = "".join(e for e in raw_text if e.isalnum())

        # --- E. Draw Visual Overlay (Dashboard) ---
        # Draw Car Bounding Box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw Tracking ID and OCR result
        label = f"ID:{assigned_id} Plate:{plate_text if plate_text else 'Unread'}"
        cv2.putText(
            display_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Draw OCR Crop Box to show what we are looking at
        cv2.rectangle(
            display_frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 255, 255), 2
        )

    # Update tracker memory
    active_tracks = current_frame_tracks

    # --- F. Render Frame ---
    display_frame = cv2.resize(display_frame, (800, 600))
    cv2.imshow("Gate Monitor System", display_frame)

    # Wait 500ms between frames so we can see what's happening
    if cv2.waitKey(500) & 0xFF == ord("q"):
        break

print("\nPipeline finished.")
cv2.destroyAllWindows()
