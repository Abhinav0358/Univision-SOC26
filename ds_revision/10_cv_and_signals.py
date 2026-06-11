"""
Lab 10: Computer Vision and Signal Processing - Deep Dive Tutorial
==================================================================
Syllabus Topics: CNN Layers (Kernel, Padding, Stride, Pooling), HOG, MFCCs.

This script explains how computers "see" spatial patterns in images
and "hear" spectral patterns in audio.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage.feature import hog

# =============================================================================
# 1. COMPUTER VISION: CONVOLUTIONAL NEURAL NETWORKS (CNN)
# =============================================================================
# How it works:
# 1. KERNEL (Filter): A small matrix (e.g., 3x3) that slides over the image.
# 2. CONVOLUTION: Multiplies the kernel with the image pixels.
#    - Different kernels find different things (Edges, Corners, Textures).
# 3. PADDING: Adding a border of zeros to keep the image size the same after convolution.
# 4. STRIDE: How many pixels the kernel "jumps" each time. (Stride 2 means skip every other pixel).
# 5. POOLING (MAX POOLING): Reducing image size by keeping only the maximum value in a window.
#    - This makes the model "Translation Invariant" (It doesn't care exactly WHERE the eye is).

# Creating a toy square image
image = np.zeros((20, 20))
image[5:15, 5:15] = 1

# SOBEL KERNEL: Designed to find vertical edges.
sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# Perform Convolution
edges = signal.convolve2d(image, sobel_v, mode="same")

# =============================================================================
# 2. TRADITIONAL DESCRIPTORS: HOG
# =============================================================================
# Before deep learning (CNNs), we used HOG (Histogram of Oriented Gradients).
# Logic: Shapes are defined by the distribution of directions (gradients) of intensity.
# Count how many times each orientation appears in a local cell.
# Very powerful for Human Detection (Pedestrian tracking).

fd, hog_viz = hog(image, orientations=8, pixels_per_cell=(4, 4), visualize=True)

# =============================================================================
# 3. AUDIO SIGNALS: MFCCs (Mel-Frequency Cepstral Coefficients)
# =============================================================================
# Computers don't "hear" waves; they "hear" frequencies.
# - STFT (Short-Time Fourier Transform): Converts time signal into frequency signal.
# - MEL SCALE: Human hearing is more sensitive to changes in low frequencies than high.
#   The Mel scale warps the frequency axis to match human perception.
# - MFCCs: The "Gold Standard" feature for speech recognition (Alexa/Siri).
#   It's a compressed representation of the "shape" of the sound.

print("--- Audio Signal Chain ---")
print("Raw Wave -> STFT (Spectrogram) -> Mel Scale -> Log Power -> DCT -> MFCCs")

# =============================================================================
# 4. VISUALIZATION
# =============================================================================
plt.figure(figsize=(15, 5))

# Plot 1: Original Image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("1. Original Matrix (Image)")

# Plot 2: Convolution (Feature Map)
plt.subplot(1, 3, 2)
plt.imshow(np.abs(edges), cmap="hot")
plt.title("2. CNN Kernel: Finding Edges")

# Plot 3: HOG (Shape Descriptor)
plt.subplot(1, 3, 3)
plt.imshow(hog_viz, cmap="gray")
plt.title("3. HOG: Finding Shape Patterns")

plt.tight_layout()
plt.savefig("lab10_results.png")
print("\n[SUCCESS] Lab 10 Complete. Features extracted!")
