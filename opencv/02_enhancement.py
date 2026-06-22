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

# Programmatically generate a low-quality, noisy synthetic license plate image
img_path = "noisy_plate.jpg"
if not os.path.exists(img_path):
    # Dark gray background
    base_img = np.ones((200, 500, 3), dtype=np.uint8) * 40
    # Add dark text to simulate poor contrast
    cv2.putText(
        base_img, "ABC-1234", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (90, 90, 90), 8
    )

    # Inject Gaussian noise to simulate camera grain
    noise = np.random.normal(0, 15, base_img.shape).astype(np.uint8)
    noisy_img = cv2.add(base_img, noise)
    cv2.imwrite(img_path, noisy_img)

# Load the generated noisy image and convert to grayscale
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Denoising: Gaussian blur averages local neighborhood values to suppress pixel grain
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Contrast Enhancement: Histogram equalization stretches pixel intensities
# to cover the full [0, 255] dynamic range
equalized = cv2.equalizeHist(blurred)

carimg = cv2.imread("sample_car.jpg")
carimg = cv2.cvtColor(carimg, cv2.COLOR_BGR2GRAY)

# Thresholding (Binarization): Maps pixels to pure black (0) or white (255)
# Global Thresholding: Uses a fixed cutoff threshold (e.g., 120) across the entire image
_, global_thresh = cv2.threshold(equalized, 120, 255, cv2.THRESH_BINARY)
_, car_binarythresh = cv2.threshold(carimg, 120, 255, cv2.THRESH_BINARY)
cv2.imwrite("assets/car_binarythresh.png", car_binarythresh)

# Adaptive Thresholding: Computes localized threshold cuts based on local window means,
# producing cleaner results under uneven or gradients of illumination
adaptive_thresh = cv2.adaptiveThreshold(
    equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

caradaptive_thresh = cv2.adaptiveThreshold(
    carimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
cv2.imwrite("assets/car_adaptive_thresh.png", caradaptive_thresh)

# Edge Detection: Canny filter isolates pixels with high intensity gradients
# The parameters 50 and 150 define the hysteresis threshold limits
edges = cv2.Canny(blurred, 50, 150)
cv2.imwrite("assets/car_canny_edges.png", cv2.Canny(carimg, 50, 150))

# Display preprocessed outputs
print("Displaying image enhancement steps...")
cv2.imshow("1. Original Noisy", img)
cv2.imshow("2. Grayscale & Blurred", blurred)
cv2.imshow("3. Equalized Contrast", equalized)
cv2.imshow("4. Global Thresholding", global_thresh)
cv2.imshow("5. Adaptive Thresholding", adaptive_thresh)
cv2.imshow("6. Canny Edge Maps", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
