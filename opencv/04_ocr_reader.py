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
img_path = "sample_car.jpg"
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found. Please run 01 & 02")
    exit()

img = cv2.imread(img_path)

# --- 2. Simulate Bounding Box Crop (Linking File 3 to File 4) ---
# Imagine our YOLO model from File 3 told us the license plate is at these coordinates:

# Crop the image using array slicing [y1:y2, x1:x2]
cropped_plate = img[1420:1570, 1400:1950]


# --- 3. Preprocess for OCR (Applying File 2 knowledge) ---
# OCR engines are easily confused by noise and color.
gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(gray, (21, 21), 0)


# equalized = cv2.equalizeHist(blurred)

# cv2.imshow("equalized Plate", equalized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # Use thresholding to make text pure white and background pure black
_, binary_plate = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)


# adaptive thresholding
# adaptive_plate = cv2.adaptiveThreshold(
#     equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
# )

# cv2.imshow("adaptive Plate", adaptive_plate)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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
# 1 Letter, 3 Numbers, 3 letters (e.g., J389NLT)
pattern = r"^[A-Z]{1}\d{3}[A-Z]{3}$"

print("\n--- Validation Step ---")
if re.match(pattern, clean_text):
    print(f"✅ SUCCESS: '{clean_text}' is a valid license plate format.")
else:
    print(f"❌ ERROR: '{clean_text}' does NOT match the expected format.")
    print("   (This means the OCR failed or the car has an invalid plate.)")

img = cv2.resize(img, (400, 300))
cv2.imshow("Original", img)

cv2.imshow("Cropped Plate", cropped_plate)
cv2.imshow("gray Plate", gray)
cv2.imshow("blurred Plate", blurred)
cv2.imshow("binary Plate", binary_plate)
cv2.waitKey(0)
cv2.destroyAllWindows()
