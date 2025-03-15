"""
Name: Mark William
ID: 120210348
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import njit, prange

@njit(parallel=True)
def calculate_energy(image):
    # Convert image to grayscale manually
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    
    # Compute gradients using finite differences
    gradient_x = np.abs(gray[:, 1:] - gray[:, :-1]) 
    gradient_y = np.abs(gray[1:, :] - gray[:-1, :]) 
    
    # Pad gradients to match the original image size
    rows, cols = gray.shape
    padded_gradient_x = np.zeros((rows, cols), dtype=np.float64)
    padded_gradient_y = np.zeros((rows, cols), dtype=np.float64)
    
    # Pad gradient_x (right column remains 0)
    padded_gradient_x[:, :-1] = gradient_x
    
    # Pad gradient_y (bottom row remains 0)
    padded_gradient_y[:-1, :] = gradient_y
    
    # Total energy
    energy = padded_gradient_x + padded_gradient_y
    return energy

@njit(parallel=True)
def find_seam(energy):
    rows, cols = energy.shape
    dp = energy.copy()

    # Fill the DP table
    for i in range(1, rows):
        for j in prange(cols):
            if j == 0:
                dp[i, j] += min(dp[i-1, j], dp[i-1, j+1])
            elif j == cols - 1:
                dp[i, j] += min(dp[i-1, j-1], dp[i-1, j])
            else:
                dp[i, j] += min(dp[i-1, j-1], dp[i-1, j], dp[i-1, j+1])

    # Backtrack to find the seam
    seam = []
    j = np.argmin(dp[-1, :])
    for i in range(rows-1, -1, -1):
        seam.append((i, j))
        if i == 0:
            break
        if j == 0:
            j = np.argmin(dp[i-1, j:j+2]) + j
        elif j == cols - 1:
            j = np.argmin(dp[i-1, j-1:j+1]) + j - 1
        else:
            j = np.argmin(dp[i-1, j-1:j+2]) + j - 1
    return seam

def remove_seam(image, seam):
    # Remove the seam from the image.
    rows, cols, _ = image.shape
    new_image = np.zeros((rows, cols-1, 3), dtype=np.uint8)
    for i, j in seam:
        new_image[i, :j] = image[i, :j]
        new_image[i, j:] = image[i, j+1:]
    return new_image

def visualize_seams(image, seams):
    # Visualize the seams on the original image.
    for seam in seams:
        for i, j in seam:
            image[i, j] = [0, 0, 255]  # Mark seam in red
    return image

def seam_carving(image, new_width):
    # Resize the image by removing seams (optimized).
    seams = []
    for _ in range(image.shape[1] - new_width):
        energy = calculate_energy(image)
        seam = find_seam(energy)
        seams.append(seam)
        image = remove_seam(image, seam)
    return image, seams

# Load the input image
input_image = cv2.imread('input_image.jpg')

# Resize the image using seam carving
new_width = input_image.shape[1] // 2
resized_image, seams = seam_carving(input_image, new_width)

# Visualize the seams on the original image
seam_visualization = visualize_seams(input_image.copy(), seams)

# Save the resized image and seam visualization
cv2.imwrite('resized_image_bonus.jpg', resized_image)
cv2.imwrite('seam_visualization.jpg', seam_visualization)

# Display the results using matplotlib
plt.figure(figsize=(10, 5))

# Original image with seams
plt.subplot(1, 2, 1)
plt.title("Original Image with Seams")
plt.imshow(cv2.cvtColor(seam_visualization, cv2.COLOR_BGR2RGB))

# Resized image
plt.subplot(1, 2, 2)
plt.title("Resized Image")
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

plt.show()

print("Seam carving with bonus completed successfully!")