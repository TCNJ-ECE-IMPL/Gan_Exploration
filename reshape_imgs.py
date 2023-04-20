from PIL import Image
import os
import numpy as np

# Set the input and output directory paths
#input_dir = "/home/nallurn1/Anime_model/Gan_Exploration/animes"
#output_dir = "/home/nallurn1/Anime_model/Gan_Exploration/anime_faces_reshape_imgs"
#input_dir = "/home/nallurn1/Anime_model/Gan_Exploration/avatar_crops"
#output_dir = "/home/nallurn1/Anime_model/Gan_Exploration/avatar_crops_reshape_imgs"
input_dir = "/home/nallurn1/Anime_model/Gan_Exploration/naruto_crop"
output_dir = "/home/nallurn1/Anime_model/Gan_Exploration/naruto_crop_reshape_imgs"

# Set the size of the resized image
img_size = (28, 28)

# Create an empty list to store the resized images
resized_images = []

# Create the new directory if it doesn't exist
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    # Open the image file
    img = Image.open(os.path.join(input_dir, filename))
    # Resize the image
    img = img.resize(img_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add the image array to the list
    resized_images.append(img_array)
    # Save the resized image to the output directory
    img.save(os.path.join(output_dir, filename))

# Reshape the array
resized_images = np.array(resized_images).reshape((-1, 28, 28, 1)).astype('float32')

