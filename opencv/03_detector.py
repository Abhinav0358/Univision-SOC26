import os
import urllib.request
import cv2
import numpy as np

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

# Load Pre-trained Model
# Utilizing YOLOv8 nano ('yolov8n.pt') which provides lightweight inference
print("Loading YOLOv8 nano model...")
model = YOLO("yolov8n.pt")

# Load image
img = cv2.imread(img_path)
if img is None:
    print("Error downloading image.")
    exit()

# Run Inference
# Filtering detections below a confidence threshold (e.g., 0.3)
print("Running object detection...")
results = model(img, conf=0.3)

# Parse prediction outputs
detections = results[0].boxes
print(f"\nFound {len(detections)} object(s) in the image.")

drawn_img = img.copy()

# Draw detected bounding boxes on a copy of the image
for box in detections:
    # Bounding box coordinates: [x_min, y_min, x_max, y_max]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    class_id = int(box.cls[0])
    class_name = model.names[class_id]

    print(
        f"Detected: {class_name} | Confidence: {conf:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]"
    )

    # Draw box border
    cv2.rectangle(drawn_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw confidence label
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


# Intersection over Union (IoU) Calculation
# Computes spatial overlap to filter multiple overlapping predictions (Non-Maximum Suppression)
def calculate_iou(boxA, boxB):
    # Overlap coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute individual areas
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute union overlap ratio
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# Example evaluation
dummy_box1 = [100, 100, 200, 200]
dummy_box2 = [110, 110, 210, 210]
print(
    f"\nDummy IoU for highly overlapping boxes: {calculate_iou(dummy_box1, dummy_box2):.2f}"
)
print(
    "Detections with high IoU are pruned via Non-Maximum Suppression to remove redundant boxes."
)

cv2.imwrite("assets/Detected Objects.png", drawn_img)
