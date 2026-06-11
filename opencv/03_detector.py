import os
import urllib.request

import cv2
import numpy as np

# We use the 'ultralytics' library which contains the modern YOLOv8 models.
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: The 'ultralytics' library is not installed.")
    print("Please install it by running: pip install ultralytics")
    exit()


img_path = "sample_car.jpg"
if not os.path.exists(img_path):
    print("Downloading sample image...")
    url = "https://images.unsplash.com/photo-1504381270825-025726abb1de?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8bnVtYmVyJTIwcGxhdGV8ZW58MHx8MHx8fDA%3D"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(img_path, "wb") as out_file:
        out_file.write(response.read())


# --- 1. Load Pre-trained Model ---
# 'yolov8n.pt' is the "nano" version. It is very fast and lightweight.
# The first time you run this, it will download the model file (~6MB).
print("Loading YOLOv8 nano model...")
model = YOLO("yolov8n.pt")

# --- 2. Load the Image ---
img = cv2.imread(img_path)

if img is None:
    print("Error downloading image.")
    exit()

# --- 3. Run Inference (Detection) ---
# Pass the image to the model.
# We set conf=0.5 to tell the model: "Only return detections you are >50% sure about"
print("Running object detection...")
results = model(img, conf=0.3)

# --- 4. Parse the Results ---
# The model returns a list of result objects (one for each image processed).
# Since we passed 1 image, we look at results[0].
detections = results[0].boxes

print(f"\nFound {len(detections)} object(s) in the image.")

# Make a copy of the image so we can draw on it without destroying the original
drawn_img = img.copy()

# --- 5. Extract Data and Draw ---
for box in detections:
    # A bounding box has 4 coordinates: [x_min, y_min, x_max, y_max]
    # We extract them and convert them to standard integers for drawing
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Confidence score (0.0 to 1.0)
    conf = float(box.conf[0])

    # Class ID (an integer). 2 represents "car" in the standard COCO dataset.
    class_id = int(box.cls[0])
    class_name = model.names[class_id]  # Look up the string name using the dictionary

    print(
        f"Detected: {class_name} | Confidence: {conf:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]"
    )

    # Draw the bounding box (Blue color, 2 pixels thick)
    cv2.rectangle(drawn_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw the label above the box
    label = f"{class_name} {conf:.2f}"
    cv2.putText(
        drawn_img,
        label,
        (x1, max(y1 - 10, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )


# --- 6. Conceptual Exercise: Intersection over Union (IoU) ---
# When models predict, they often predict multiple boxes for the same object.
# IoU is the math used to delete duplicates.
def calculate_iou(boxA, boxB):
    # Determine coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute area of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# Example of two highly overlapping boxes
dummy_box1 = [100, 100, 200, 200]
dummy_box2 = [110, 110, 210, 210]
print(
    f"\nDummy IoU for highly overlapping boxes: {calculate_iou(dummy_box1, dummy_box2):.2f}"
)
print(
    "If IoU is > 0.5, the system deletes one to avoid duplicate counts (Non-Maximum Suppression)."
)

# --- 7. Visualization ---
cv2.imwrite("assets/Detected Objects.png", drawn_img)
