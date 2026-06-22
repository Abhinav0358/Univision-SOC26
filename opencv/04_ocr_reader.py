import os
import re
import cv2
import numpy as np

# optical character recognition setup
try:
    import pytesseract
except ImportError:
    print("Error: 'pytesseract' is not installed. Run: pip install pytesseract")
    print(
        "Note: You also need the Tesseract engine installed on your OS (e.g., sudo apt install tesseract-ocr)"
    )
    exit()

# Load the target image
img_path = "sample_car.jpg"
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found. Please run 01 & 02")
    exit()

img = cv2.imread(img_path)

# Simulate bounding box crop (as if predicted by object detection model)
# Crop the image using array slicing [y1:y2, x1:x2]
cropped_plate = img[1420:1570, 1400:1950]

# Preprocess cropped plate for OCR
# Convert to grayscale to remove irrelevant color channels
gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce high-frequency pixel noise
blurred = cv2.GaussianBlur(gray, (21, 21), 0)

# equalized = cv2.equalizeHist(blurred)
# cv2.imshow("equalized Plate", equalized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Use global thresholding to binarize the image (black text on white background)
_, binary_plate = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

# adaptive thresholding
# adaptive_plate = cv2.adaptiveThreshold(
#     equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
# )
# cv2.imshow("adaptive Plate", adaptive_plate)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Run OCR using Tesseract
print("Running OCR Engine...")
# --psm 8 configures Tesseract to treat the image as a single alphanumeric word
raw_text = pytesseract.image_to_string(binary_plate, config="--psm 8")

# Strip whitespace and line breaks
clean_text = raw_text.strip().replace(" ", "")

print(f"\nRaw OCR Output:   '{raw_text.strip()}'")
print(f"Cleaned Output:   '{clean_text}'")

# Regular Expression Validation
# Validate the OCR text against a expected license plate pattern.
# Expected: 1 Letter, 3 Numbers, 3 Letters (e.g., J389NLT)
pattern = r"^[A-Z]{1}\d{3}[A-Z]{3}$"

print("\n--- Validation Step ---")
if re.match(pattern, clean_text):
    print(f"✅ SUCCESS: '{clean_text}' matches a valid format.")
else:
    print(f"❌ ERROR: '{clean_text}' does not match the expected format.")
    print("   (The OCR may have failed, or the vehicle has an alternative plate pattern.)")

img = cv2.resize(img, (400, 300))
cv2.imshow("Original", img)
cv2.imshow("Cropped Plate", cropped_plate)
cv2.imshow("gray Plate", gray)
cv2.imshow("blurred Plate", blurred)
cv2.imshow("binary Plate", binary_plate)
cv2.waitKey(0)
cv2.destroyAllWindows()
