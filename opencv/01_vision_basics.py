import os
import urllib.request

import cv2
import numpy as np

# --- 0. Setup: Fetch a sample image of a car ---
img_path = "sample_car.jpg"
if not os.path.exists(img_path):
    print("Downloading sample image...")
    url = "https://images.unsplash.com/photo-1504381270825-025726abb1de?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8bnVtYmVyJTIwcGxhdGV8ZW58MHx8MHx8fDA%3D"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(img_path, "wb") as out_file:
        out_file.write(response.read())

# --- 1. Load the Image ---
# OpenCV loads images as NumPy arrays.
img = cv2.imread(img_path)

if img is None:
    print("Error: Could not load image. Check the path.")
    exit()

# --- 2. Inspect Shape ---
# An image is just a 3D grid of numbers.
# Shape returns: (Height, Width, Channels)
print(f"1. Original Shape:  {img.shape}")

# --- 3. Crop Region of Interest (ROI) ---
# Because it's a NumPy array, we crop using standard array slicing [Y-start:Y-end, X-start:X-end]
cropped_img = img[1200:1800, 1300:2100]  # Grabbing roughly the grill/plate area
print(f"2. Cropped Shape:   {cropped_img.shape}")

# --- 4. Resize the Image ---
# Neural networks require fixed input sizes (e.g., 640x640 for YOLO).
# Note: This will squish/distort the aspect ratio!
# Interpolation between pixels
resized_img = cv2.resize(img, (1000, 1000))
print(f"3. Resized Shape:   {resized_img.shape}")

# --- 5. Convert Color Space ---
# OpenCV loads in BGR (Blue, Green, Red). We can drop it to 1 channel (Grayscale) to save compute.
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"4. Grayscale Shape: {gray_img.shape} -> Notice the '3' channel is gone!")

# --- 6. Normalize Values ---
# Raw pixels are integers from 0-255. AI models learn better with floats from 0.0 to 1.0.
# reducing intensity by 4x
normalized_img = resized_img.astype(np.float32) / 255.0

print("\n--- Pixel Data Look ---")
print(f"Raw Pixel (0,0) (BGR): {resized_img[500, 500]}")
print(f"Normalized Pixel:      {normalized_img[500, 500]}")

# --- 7. Visualization ---
# Let's see what we did! (Press any key on the image window to close)
print("\nOpening image windows... Click on a window and press any key to exit.")
# cv2.imshow("1. Original", img)
# cv2.imshow("2. Cropped", cropped_img)
# cv2.imshow("3. Grayscale", gray_img)

# save image to currrent directory
cv2.imwrite("assets/original.png", img)
cv2.imwrite("assets/gray_img.png", gray_img)
cv2.imwrite("assets/resized_img.png", resized_img)
cv2.imwrite("assets/cropped.png", cropped_img)
cv2.imwrite("assets/normalized_img.png", normalized_img)
cv2.imwrite("assets/blurred.png", cv2.GaussianBlur(img, (101, 101), 0))
cv2.imwrite("assets/equalized.png", cv2.equalizeHist(gray_img))
cv2.waitKey(0)  # Pauses script until a key is pressed
cv2.destroyAllWindows()
