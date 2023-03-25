import os
import cv2

# Define input and output directories
input_dir = "/home/nallurn1/Anime_Code/Simpson"
output_dir = "/home/nallurn1/Anime_Code/Simpson_fixed"

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    # Check that the file is an image file
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Load the image from file
        img = cv2.imread(os.path.join(input_dir, filename))

        # Split the image into its color channels
        b, g, r = cv2.split(img)

        # Apply histogram equalization to each color channel
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)

        # Merge the equalized color channels back into a single image
        equalized = cv2.merge((b_eq, g_eq, r_eq))

        # Save the equalized image to the output directory
        cv2.imwrite(os.path.join(output_dir, filename), equalized)

print("Histogram equalization complete!")
