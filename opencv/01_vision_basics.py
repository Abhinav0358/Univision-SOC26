import os
import urllib.request
import cv2
import numpy as np

# Setup: Download a sample image if it doesn't exist
img_path = "sample_car.jpg"
if not os.path.exists(img_path):
    print("Downloading sample image...")
    url = "https://images.unsplash.com/photo-1504381270825-025726abb1de?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8bnVtYmVyJTIwcGxhdGV8ZW58MHx8MHx8fDA%3D"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(img_path, "wb") as out_file:
        out_file.write(response.read())

# Load the Image
# OpenCV loads images as NumPy arrays in BGR format
img = cv2.imread(img_path)

if img is None:
    print("Error: Could not load image. Check the path.")
    exit()

# Inspect Shape
# Images are 3D grids: (Height, Width, Channels)
print(f"1. Original Shape:  {img.shape}")

# Crop Region of Interest (ROI)
# Using standard NumPy array slicing [Y-start:Y-end, X-start:X-end]
cropped_img = img[1200:1800, 1300:2100]  
print(f"2. Cropped Shape:   {cropped_img.shape}")

# Resize the Image
# Interpolating pixels to reach the target resolution (1000x1000)
resized_img = cv2.resize(img, (1000, 1000))
print(f"3. Resized Shape:   {resized_img.shape}")

# Convert Color Space
# Convert BGR representation to 1-channel Grayscale to optimize computational cost
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"4. Grayscale Shape: {gray_img.shape} -> Notice the channels dimension is dropped")

# Normalize Values
# Scale integer pixel values [0-255] to floating-point values in the range [0.0, 1.0]
normalized_img = resized_img.astype(np.float32) / 255.0

print("\n--- Pixel Data Comparison ---")
print(f"Raw Pixel (500,500) (BGR): {resized_img[500, 500]}")
print(f"Normalized Pixel:         {normalized_img[500, 500]}")

# Save the processed outputs
cv2.imwrite("assets/original.png", img)
cv2.imwrite("assets/gray_img.png", gray_img)
cv2.imwrite("assets/resized_img.png", resized_img)
cv2.imwrite("assets/cropped.png", cropped_img)
cv2.imwrite("assets/normalized_img.png", normalized_img)
cv2.imwrite("assets/blurred.png", cv2.GaussianBlur(img, (101, 101), 0))
cv2.imwrite("assets/equalized.png", cv2.equalizeHist(gray_img))
cv2.waitKey(0)  
cv2.destroyAllWindows()
