import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# Get the paths/directories for the image datasets
folder_dir1 = "/home/nallurn1/Anime_Code/YogiBear/"
folder_dir2 = "/home/nallurn1/Anime_Code/Simpson/"

fraction1 = 1
fraction2 = .5

# Get a list of all the image files in the directory
image_files = os.listdir(folder_dir1)
# Calculate the number of images to be read
num_images = int(len(image_files) * fraction2)
# Randomly sample a fraction of the image files
image_files = random.sample(image_files, num_images)
# Read the selected images into a list
images = []
for file in image_files:
    image_path = os.path.join(image_dir, file)
    image = cv2.imread(image_path)
    images.append(image)

# Initialize variables to store the probability distributions
merged_hist = np.zeros(256)


#change to images[]
# Loop through the images in the first directory
for filename1 in os.listdir(folder_dir1):
    # Check if the file ends with ".png"
    if filename1.endswith(".png"):
        # Load the image and calculate its histogram
        img1 = cv2.imread(os.path.join(folder_dir1, filename1), cv2.IMREAD_GRAYSCALE)
        hist1, _ = np.histogram(img1.ravel(), 256, [0, 256])

        # Normalize the histogram
        hist1_norm = hist1 / np.sum(hist1)

        # Loop through the images in the second directory
        for filename2 in os.listdir(folder_dir2):
            # Check if the file ends with ".png"
            if filename2.endswith(".png"):
                print("HERE")
                # Load the image and calculate its histogram
                img2 = cv2.imread(os.path.join(folder_dir2, filename2), cv2.IMREAD_GRAYSCALE)
                hist2, _ = np.histogram(img2.ravel(), 256, [0, 256])

                #hist2 is a numpy array, length 256?  Values are # of pixels with that intensity
                x_bar = range(256)*hist2/len(hist2) # mean value, (1/N) * sum(prod(i, hist2[i]))
                x_sigma_sq =  (range(256)-x_bar)**2.0 * hist2/len(hist2) # sample variance, (1/N) * sum(prod((i-x_bar)**2, hist2[i]))
                x_sigma = sqrt(x_sigma_sq) # this can be used as a measure of contrast

                # Normalize the histogram
                hist2_norm = hist2 / np.sum(hist2)

                # Merge the two histograms
                merged_hist += hist1_norm * hist2_norm

# Normalize the merged histogram
merged_hist_norm = merged_hist / np.sum(merged_hist)

# Plot the merged histogram
# plt.hist(np.arange(256), bins=256, weights=merged_hist_norm)
# plt.show()
# Plot the probability distributions for each dataset and the merged distribution
plt.figure(figsize=(10, 6))
plt.plot(hist1_norm, label="YogiBear")
plt.plot(hist2_norm, label="Simpson")
plt.plot(merged_hist_norm, label="Merged")
plt.title("Probability Distribution of YogiBear and Simpson Datasets")
plt.xlabel("Pixel Intensity")
plt.ylabel("Normalized Frequency")
plt.legend()
plt.show()