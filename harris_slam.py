import cv2
import numpy as np
import os

# Input and output directories
input_folder = "KITTI_images/"
output_folder = "Harris_output/"
os.makedirs(output_folder, exist_ok=True)

# Process all images
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
        gray = np.float32(image)

        # Apply Harris Corner Detector
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)  # Better visualization

        # Mark detected corners in red
        image[dst > 0.01 * dst.max()] = [0, 0, 255]

        # Save the result
        cv2.imwrite(os.path.join(output_folder, filename), image)

print("Harris Corner Detection applied to all images.")
