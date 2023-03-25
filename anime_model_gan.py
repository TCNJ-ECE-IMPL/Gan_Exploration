import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the parameters of the model
latent_dim = 100
num_classes = 3
batch_size = 128
num_epochs_discriminator = 1
num_epochs_generator = 2
img_shape = (28, 28, 1)

# Define the generator model
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(7*7*256, input_dim=latent_dim),
    tf.keras.layers.Reshape((7, 7, 256)),
    tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='tanh')
])

# Define the discriminator model
discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes + 1, activation='softmax')
])

# Define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
classification_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define the discriminator loss function
def discriminator_loss(real_output, fake_output, real_labels, fake_labels):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    classification_loss_real = classification_loss(real_labels, real_output[:, :-1])
    classification_loss_fake = classification_loss(fake_labels, fake_output[:, :-1])
    total_loss = real_loss + fake_loss + classification_loss_real + classification_loss_fake
    return total_loss

# Define the generator loss function
def generator_loss(fake_output, fake_labels):
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    classification_loss_fake = classification_loss(fake_labels, fake_output[:, :-1])
    total_loss = fake_loss + classification_loss_fake
    return total_loss

# Define the optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define a function to load and preprocess images
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = plt.imread(os.path.join(folder_path, filename))
            img = np.expand_dims(img, axis=-1)
            img = img.astype('float32') / 255
            images.append(img)
    return np.array(images)

def train_gan(gan, discriminator, generator, x_train, y_train, epochs, batch_size):
    # Define labels for real and fake images
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # Loop over the epochs
    for epoch in range(epochs):

        # Shuffle the training data
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        # Loop over the batches
        for i in range(0, len(x_train), batch_size):

            # Get a batch of real images and labels
            x_batch_real = x_train[i:i+batch_size]
            y_batch_real = y_train[i:i+batch_size]

            # Generate a batch of fake images and labels
            noise = tf.random.normal([batch_size, 100])
            y_batch_fake = keras.utils.to_categorical(np.random.randint(0, num_classes, size=batch_size),
                                                      num_classes=num_classes)
            x_batch_fake = generator.predict([noise, y_batch_fake])

            # Concatenate the real and fake images and labels
            x_batch = np.concatenate([x_batch_real, x_batch_fake])
            y_batch = np.concatenate([y_batch_real, y_batch_fake])
            combined_labels = tf.concat([real_labels, fake_labels], axis=0)
            y_batch = [y_batch, combined_labels]

            # Train the discriminator on the batch
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(x_batch, y_batch)

            # Train the generator
            noise = tf.random.normal([batch_size, 100])
            y_generated = keras.utils.to_categorical(np.random.randint(0, num_classes, size=batch_size),
                                                      num_classes=num_classes)
            y_generated_labels = np.ones((batch_size, 1))
            g_loss = combined.train_on_batch([noise, y_generated], [y_generated_labels, y_generated])

        # Print the loss and accuracy for each epoch
        print(f"Epoch {epoch+1}/{epochs}: Discriminator Loss: {d_loss[0]}, Discriminator Accuracy: {d_loss[3]}, Generator Loss: {g_loss[0]}")

# Load the images from the three folders and combine them into one
dataset1_path = "/home/nallurn1/Anime_Code/anime_faces_crop"
dataset2_path = "/home/nallurn1/Anime_Code/avatar_crops"
dataset3_path = "/home/nallurn1/Anime_Code/naruto_crop"

#Concatenate the three sets of images into one
images1 = load_images(dataset1_path)
images2 = load_images(dataset2_path)
images3 = load_images(dataset3_path)
images = np.concatenate((images1, images2, images3))

# Normlize pixel values to be between -1 and 1
images = (images - 127.5) / 127.5

##Reshape the images to match the input shape of the generator
images = np.reshape(images, (-1, 28, 28, 1)).astype('float32')

#Define the labels for the three datasets
num_images1 = len(images1)
num_images2 = len(images2)
num_images3 = len(images3)
labels1 = np.zeros(num_images1)
labels2 = np.ones(num_images2)
labels3 = np.ones(num_images3) * 2

#Concatenate the labels for the three datasets into one
labels = np.concatenate((labels1, labels2, labels3))

#Convert the labels to one-hot encoding
num_classes = 3
labels = keras.utils.to_categorical(labels, num_classes=num_classes)

#Shuffle the images and labels
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

#Split the data into training and testing sets
num_train = int(0.8 * len(images))
x_train, x_test = images[:num_train], images[num_train:]
y_train, y_test = labels[:num_train], labels[num_train:]

#Compile the discriminator
discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
optimizer=optimizer,
metrics=['accuracy'])

#Compile the combined model
combined.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
optimizer=optimizer)

#Train the GAN
train_gan(gan, discriminator, generator, x_train, y_train, epochs=100, batch_size=128)