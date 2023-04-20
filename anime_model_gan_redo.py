import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Define the parameters of the model
latent_dim = 100
num_classes = 3
batch_size = 32
num_epochs_discriminator = 50
num_epochs_generator = 10
img_shape = (28, 28, 3)

# Define the file paths and labels for each dataset
dataset_paths = ['/home/nallurn1/Anime_model/Gan_Exploration/anime_faces_reshape_imgs',
                 '/home/nallurn1/Anime_model/Gan_Exploration/avatar_crops_reshape_imgs','/home/nallurn1/Anime_model/Gan_Exploration/naruto_crop_reshape_imgs']
dataset_labels = [0, 1, 2]

# Load the images and labels from each dataset
images = []
labels = []
for i, dataset_path in enumerate(dataset_paths):
    for image_filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_filename)
        image = keras.preprocessing.image.load_img(image_path, target_size=img_shape[:2])
        image = keras.preprocessing.image.img_to_array(image)
        image = (image - 127.5) / 127.5  # Normalize the input images to [-1, 1]
        images.append(image)
        labels.append(dataset_labels[i])

print("Data type:", np.shape(images))
print("Shape:", np.shape(labels))

images = np.array(images)
labels = np.array(labels)

# Define the generator model
noise_input = keras.layers.Input(shape=(latent_dim,))
label_input = keras.layers.Input(shape=(1,))
label_embedding = keras.layers.Embedding(num_classes, latent_dim)(label_input)
label_embedding = keras.layers.Flatten()(label_embedding)
model_input = keras.layers.multiply([noise_input, label_embedding])
x = keras.layers.Dense(128 * 7 * 7, activation="relu")(model_input)
x = keras.layers.Reshape((7, 7, 128))(x)
x = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Conv2D(3, (3, 3), activation="tanh", padding="same")(x)
generator = keras.models.Model(inputs=[noise_input, label_input], outputs=x)

#Define the discriminator model
img_input = keras.layers.Input(shape=img_shape)
label_input = keras.layers.Input(shape=(1,))
label_embedding = keras.layers.Embedding(num_classes, np.prod(img_shape))(label_input)
label_embedding = keras.layers.Reshape(img_shape)(label_embedding)
model_input = keras.layers.concatenate([img_input, label_embedding], axis=-1)
x = keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")(model_input)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Dropout(0.25)(x)
x = keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Dropout(0.25)(x)
x = keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Dropout(0.25)(x)
x = keras.layers.Flatten()(x)
model_input = keras.layers.concatenate([x, label_embedding], axis=-1)
model_output = keras.layers.Dense(1, activation="sigmoid")(model_input)
discriminator = keras.models.Model(inputs=[img_input, label_input], outputs=model_output)

#Compile the models
discriminator.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"])
generator.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")

#Define the combined model
noise_input = keras.layers.Input(shape=(latent_dim,))
label_input = keras.layers.Input(shape=(1,))
model_input = [noise_input, label_input]
generated_image = generator(model_input)
discriminator.trainable = False
validity = discriminator([generated_image, label_input])
combined_model = keras.models.Model(inputs=model_input, outputs=validity)
combined_model.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")

#Train the GAN
for epoch in range(num_epochs_discriminator):
    for i in range(len(images) // batch_size):
        # Select a random batch of images and labels
        batch_images = images[i * batch_size:(i + 1) * batch_size]
        batch_labels = labels[i * batch_size:(i + 1) * batch_size]
        # Generate a batch of noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Generate a batch of fake images
        gen_images = generator.predict([noise, batch_labels])

        # Train the discriminator on real and fake images
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch([batch_images, batch_labels], real_labels)
        d_loss_fake = discriminator.train_on_batch([gen_images, batch_labels], fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

print(f"Discriminator loss: {d_loss[0]}, Accuracy: {100*d_loss[1]}%")

for epoch in range(num_epochs_generator):
    for i in range(len(images) // batch_size):
        # Generate a batch of noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
            # Get a batch of real images and their labels
    batch_images = images[i * batch_size : (i + 1) * batch_size]
    batch_labels = labels[i * batch_size : (i + 1) * batch_size]
    batch_labels = batch_labels.reshape(-1, 1)

    # Train the generator to generate images that the discriminator classifies as real
    g_loss = combined.train_on_batch([noise, batch_labels], np.ones((batch_size, 1)))

print(f"Generator loss: {g_loss}")

os.makedirs('gen_images_redo', exist_ok=True)
# Save a sample of generated images for each epoch
noise = np.random.normal(0, 1, (num_classes, latent_dim))
gen_images = generator.predict([noise, np.arange(num_classes)])
gen_images = 0.5 * gen_images + 0.5  # Scale the output images from [-1, 1] to [0, 1]
fig, axs = plt.subplots(nrows=1, ncols=num_classes, figsize=(num_classes * 3, 3))
for i in range(num_classes):
    axs[i].imshow(gen_images[i])
    axs[i].axis("off")
    axs[i].set_title(f"Class {i}")
plt.savefig(f"gen_images_redo/generated_images_epoch_{epoch}.png")
plt.close()

       
