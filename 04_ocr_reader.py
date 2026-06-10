import os
import re

import cv2
import numpy as np

# We will use pytesseract for Optical Character Recognition
try:
    import pytesseract
except ImportError:
    print("Error: 'pytesseract' is not installed. Run: pip install pytesseract")
    print(
        "Note: You also need the Tesseract engine installed on your OS (e.g., sudo apt install tesseract-ocr)"
    )
    exit()

# --- 1. Load the Noisy Image (from File 2) ---
img_path = "noisy_plate.jpg"
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found. Please run 02_enhancement.py first.")
    exit()

img = cv2.imread(img_path)

# --- 2. Simulate Bounding Box Crop (Linking File 3 to File 4) ---
# Imagine our YOLO model from File 3 told us the license plate is at these coordinates:
x1, y1, x2, y2 = 30, 40, 480, 180

# Crop the image using array slicing [y1:y2, x1:x2]
cropped_plate = img[y1:y2, x1:x2]

# --- 3. Preprocess for OCR (Applying File 2 knowledge) ---
# OCR engines are easily confused by noise and color.
gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Use thresholding to make text pure white and background pure black
_, binary_plate = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

# --- 4. Run OCR ---
print("Running OCR Engine...")
# --psm 8 tells Tesseract to treat the image as a single word/line (good for plates)
raw_text = pytesseract.image_to_string(binary_plate, config="--psm 8")

# Clean up the text (remove newlines and random spaces)
clean_text = raw_text.strip().replace(" ", "")

print(f"\nRaw OCR Output:   '{raw_text.strip()}'")
print(f"Cleaned Output:   '{clean_text}'")

# --- 5. Validation using Regular Expressions (Regex) ---
# OCR makes mistakes. "0" becomes "O", "8" becomes "B", etc.
# We validate the text against a known pattern. Let's assume valid plates are:
# 3 Letters, a dash, 4 Numbers (e.g., ABC-1234)
pattern = r"^[A-Z]{3}-\d{4}$"

print("\n--- Validation Step ---")
if re.match(pattern, clean_text):
    print(f"✅ SUCCESS: '{clean_text}' is a valid license plate format.")
else:
    print(f"❌ ERROR: '{clean_text}' does NOT match the expected format.")
    print("   (This means the OCR failed or the car has an invalid plate.)")

# --- 6. Visualization ---
cv2.imshow("1. Cropped Plate", cropped_plate)
cv2.imshow("2. Binary (What OCR sees)", binary_plate)
cv2.waitKey(0)
cv2.destroyAllWindows()
