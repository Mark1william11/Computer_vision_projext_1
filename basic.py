"""
Name: Mark William
ID: 120210348
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_energy(image):
    """
    Calculate the energy of each pixel using the provided formula:
    e = |dI/dx| + |dI/dy|
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    dx = np.abs(np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1))
    dy = np.abs(np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0))
    energy = dx + dy 
    return energy

def find_seam(energy):
    # Find the seam with the least energy using dynamic programming.
    rows, cols = energy.shape
    dp = energy.astype(np.float64) 

    # Fill the DP table
    for i in range(1, rows):
        for j in range(cols):
            if j == 0:
                dp[i, j] += min(dp[i-1, j], dp[i-1, j+1])  # Left edge
            elif j == cols - 1:
                dp[i, j] += min(dp[i-1, j-1], dp[i-1, j])  # Right edge
            else:
                dp[i, j] += min(dp[i-1, j-1], dp[i-1, j], dp[i-1, j+1])  # Middle

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
            if 0 <= i < image.shape[0] and 0 <= j < image.shape[1]:
                image[i, j] = [0, 0, 255] 
    return image

def seam_carving(image, new_width, new_height):
    # Resize the image by removing seams (both horizontal and vertical).
    seams = [] 
    original_image = image.copy()

    # Resize horizontally
    for _ in range(image.shape[1] - new_width):
        energy = calculate_energy(image)
        seam = find_seam(energy)
        seams.append(seam)
        image = remove_seam(image, seam)

    # Resize vertically
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    for _ in range(image.shape[1] - new_height):
        energy = calculate_energy(image)
        seam = find_seam(energy)
        seams.append(seam)
        image = remove_seam(image, seam)
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image, seams, original_image

# Load the input image
input_image = cv2.imread('input_image.jpg')

# Resize the image using seam carving
new_width = input_image.shape[1] // 2
new_height = input_image.shape[0] // 2
resized_image, seams, original_image = seam_carving(input_image, new_width, new_height)

# Visualize the seams on the original image
seam_visualization = visualize_seams(original_image.copy(), seams)

# Save the resized image and seam visualization
cv2.imwrite('resized_image.jpg', resized_image)
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

print("Seam carving completed successfully!")