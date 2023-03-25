import os
import shutil
import numpy as np
from PIL import Image

# Set the path to the folder containing the images
image_folder_path = "/home/nallurn1/Anime_Code/Simpson/"

# Create the path to the usable image folder
usable_image_folder_path = os.path.join(os.path.dirname(image_folder_path), os.path.basename(image_folder_path) + "_useable")

# Create the usable image folder if it doesn't already exist
if not os.path.exists(usable_image_folder_path):
    os.makedirs(usable_image_folder_path)

# Set the threshold for the standard deviation of the pixel values
#90: anime faces, darker hair
std_threshold = 50

# Loop through all the images in the folder
for filename in os.listdir(image_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        
        # Load the image
        img = Image.open(os.path.join(image_folder_path, filename))
        
        # Convert the image to a numpy array
        img_array = np.array(img)
        
        # Calculate the standard deviation of the pixel values
        std = np.std(img_array)
        
        # Check if the STD is below the threshold
        #This need to be reversed
        #Filters out light hair 
        if std < std_threshold:
            #Display the removed images
            img.show()
            img.close()
            # Delete the dark image
            os.remove(os.path.join(image_folder_path, filename))
            print(f"{filename} deleted because it is too dark.")
        
        else:
            # Move the usable image to the usable image folder
            shutil.move(os.path.join(image_folder_path, filename), os.path.join(usable_image_folder_path, filename))
            print(f"{filename} moved to {usable_image_folder_path} because it is usable.")