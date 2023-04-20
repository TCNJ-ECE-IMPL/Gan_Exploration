#One Face
import cv2
import os

# Set the paths for the input and output directories
input_dir = "/home/nallurn1/Anime_model/Gan_Exploration/avatar"
output_dir = "/home/nallurn1/Anime_model/Gan_Exploration/avatar_crop"

# Load the pre-trained face detection model
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# Loop through all the files in the input directory
for filename in os.listdir(input_dir):
    # Load the image
    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Could not load image at path {image_path}")
        continue  # Skip this image and move on to the next one

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through the faces and crop the image to include only the faces
    for (x, y, w, h) in faces:
        # Expand the detected face region to include the hair
        x = max(0, x - int(w * 0.2))
        y = max(0, y - int(h * 0.3))
        w = min(image.shape[1] - x, int(w * 1.4))
        h = min(image.shape[0] - y, int(h * 1.5))

        # Crop the image to include only the face and hair
        cropped = image[y:y+h, x:x+w]

        # Save the cropped image to the output directory
        output_filename = os.path.splitext(filename)[0] + "_cropped.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cropped)

