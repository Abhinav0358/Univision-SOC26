"""
Computer Vision and Signal Processing Fundamentals

Demonstrates basic image convolutions, traditional spatial descriptors (HOG), 
and the signal processing pipeline for speech features (MFCCs).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage.feature import hog

# Convolutional Neural Network (CNN) Basics
# - Kernel (Filter): A sliding matrix that detects local features (edges, textures).
# - Convolution: Element-wise multiplication and summation of kernel weights over pixels.
# - Padding: Surrounds the image border to control spatial output size.
# - Stride: Step size of the kernel as it traverses the input matrix.
# - Pooling (e.g., Max Pooling): Aggregates window values to achieve spatial translation invariance.

# Create synthetic square object
image = np.zeros((20, 20))
image[5:15, 5:15] = 1

# Sobel filter for vertical edge detection
sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# Apply spatial convolution
edges = signal.convolve2d(image, sobel_v, mode="same")

# Histogram of Oriented Gradients (HOG)
# Pre-deep-learning descriptor representing structural appearance by collecting
# gradients of intensity across localized cells. Common in pedestrian detection.
fd, hog_viz = hog(image, orientations=8, pixels_per_cell=(4, 4), visualize=True)

# Audio Signal Processing: MFCCs (Mel-Frequency Cepstral Coefficients)
# Converts continuous wave data into perceptual speech features:
# - Short-Time Fourier Transform (STFT): Maps time-domain signals to frequency-domain.
# - Mel Scale: Non-linear mapping of frequency to match human acoustic sensitivity.
# - Discrete Cosine Transform (DCT): De-correlates Mel spectral bands into cepstral features.
print("--- Audio Preprocessing Pipeline ---")
print("Time Domain -> STFT (Spectrogram) -> Mel-Scale Filtering -> Log Power -> DCT -> MFCCs")

# Visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Binary Image")

plt.subplot(1, 3, 2)
plt.imshow(np.abs(edges), cmap="hot")
plt.title("Sobel Filter Convolved (Edges)")

plt.subplot(1, 3, 3)
plt.imshow(hog_viz, cmap="gray")
plt.title("HOG Visual Representation")

plt.tight_layout()
plt.savefig("lab10_results.png")
print("\n[SUCCESS] CV and signal processing lab completed. Plots saved to 'lab10_results.png'.")
