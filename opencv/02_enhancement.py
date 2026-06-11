import os
import urllib.request

import cv2
import numpy as np

img_path = "sample_car.jpg"
if not os.path.exists(img_path):
    print("Downloading sample image...")
    url = "https://images.unsplash.com/photo-1504381270825-025726abb1de?fm=jpg&q=60&w=3000&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8bnVtYmVyJTIwcGxhdGV8ZW58MHx8MHx8fDA%3D"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(img_path, "wb") as out_file:
        out_file.write(response.read())


# --- 0. Setup: Create a "Messy" Synthetic License Plate ---
# Instead of downloading, let's programmatically generate a bad-quality image.
# It will be dark, low-contrast, and noisy—exactly what a bad camera produces.
img_path = "noisy_plate.jpg"
if not os.path.exists(img_path):
    # Create a dark background
    base_img = np.ones((200, 500, 3), dtype=np.uint8) * 40
    # Add dark, hard-to-read text
    cv2.putText(
        base_img, "ABC-1234", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (90, 90, 90), 8
    )

    # Add random camera grain/noise
    noise = np.random.normal(0, 15, base_img.shape).astype(np.uint8)
    noisy_img = cv2.add(base_img, noise)
    cv2.imwrite(img_path, noisy_img)

# --- 1. Load the Image & Grayscale ---
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 2. Denoising (Gaussian Blur) ---
# Blur removes the random "grain" (noise) by averaging pixels with their neighbors.
# The (5, 5) is the kernel size (how big of an area to average). Must be odd numbers.
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- 3. Contrast Enhancement (Histogram Equalization) ---
# The image is too dark. Histogram equalization stretches the pixel values
# so the darkest parts become 0 and the brightest become 255.
equalized = cv2.equalizeHist(blurred)

carimg = cv2.imread("sample_car.jpg")
carimg = cv2.cvtColor(carimg, cv2.COLOR_BGR2GRAY)
# --- 4. Thresholding (Binarization) ---
# AI/OCR models love black-and-white (binary) images.
# We turn pixels into purely 0 (black) or 255 (white).
# If a pixel is > 120, it becomes 255. Otherwise, 0.
_, global_thresh = cv2.threshold(equalized, 120, 255, cv2.THRESH_BINARY)
_, car_binarythresh = cv2.threshold(carimg, 120, 255, cv2.THRESH_BINARY)
cv2.imwrite("assets/car_binarythresh.png", car_binarythresh)

# Adaptive Thresholding: Better for uneven lighting. It calculates the threshold
# for small regions rather than one global number for the whole image.
adaptive_thresh = cv2.adaptiveThreshold(
    equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

caradaptive_thresh = cv2.adaptiveThreshold(
    carimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
cv2.imwrite("assets/car_adaptive_thresh.png", caradaptive_thresh)

# --- 5. Edge Detection (Canny) ---
# Sometimes we just want the outlines of the characters.
# The numbers 50 and 150 are the lower and upper bounds for detecting sharp changes in pixels.
edges = cv2.Canny(blurred, 50, 150)
cv2.imwrite("assets/car_canny_edges.png", cv2.Canny(carimg, 50, 150))
# --- 6. Visualization ---
print("Opening preprocessing comparison...")
print("Look at how the text becomes easier for a machine to read at each step!")

# We can use cv2.imshow for each, but let's arrange them smartly if we want.
cv2.imshow("1. Original Messy", img)
cv2.imshow("2. Grayscale + Blur", blurred)
cv2.imshow("3. Contrast Enhanced", equalized)
cv2.imshow("4. Global Threshold", global_thresh)
cv2.imshow("5. Adaptive Threshold", adaptive_thresh)
cv2.imshow("6. Canny Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
