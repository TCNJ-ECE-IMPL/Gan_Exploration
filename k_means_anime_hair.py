import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import shutil

# Define the path to the dataset
dataset_path = '/home/nallurn1/Anime_Code/avatar_crops'

# Create a directory to store the filtered images
filtered_images_dir = os.path.join(dataset_path, 'filtered_images')
if not os.path.exists(filtered_images_dir):
    os.makedirs(filtered_images_dir)

# Load the dataset
images = []
for i in range(1, 10):
    image_path = os.path.join(dataset_path, 'image{}.png'.format(i))
    image = cv2.imread(image_path)
    # Display the image
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if image is not None:
        images.append(image)
    else:
        print("Error: Could not load image {}.".format(i))

# Calculate the STD values for each image
std_values = []
for image in images:
    std_value = np.std(image)
    std_values.append(std_value)

# Convert the STD values to a 2D array
std_values = np.array(std_values).reshape(-1, 1)

if std_values.size == 0:
    print("Error: STD values array is empty.")
    exit()

# Perform K-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=0).fit(std_values)

# Get the cluster labels for each image
cluster_labels = kmeans.labels_

# Filter out images in the cluster with the highest STD values
max_cluster_label = np.argmax(np.bincount(cluster_labels))
filtered_images = [image for i, image in enumerate(images) if cluster_labels[i] != max_cluster_label]

# Save the filtered images to the directory
for i, image in enumerate(filtered_images):
    image_name = 'filtered_image{}.jpg'.format(i)
    image_path = os.path.join(filtered_images_dir, image_name)
    cv2.imwrite(image_path, image)

# Move the filtered images to their own folder
filtered_images_folder = os.path.join(dataset_path, 'filtered_images_folder')
if not os.path.exists(filtered_images_folder):
    os.makedirs(filtered_images_folder)
for i, image in enumerate(filtered_images):
    image_name = 'filtered_image{}.jpg'.format(i)
    source_path = os.path.join(filtered_images_dir, image_name)
    destination_path = os.path.join(filtered_images_folder, image_name)
    shutil.move(source_path, destination_path)

# Remove the empty directory
os.rmdir(filtered_images_dir)
