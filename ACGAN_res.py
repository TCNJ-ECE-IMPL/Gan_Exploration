import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

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
    tf.keras.layers.BatchNorm
    lization(),
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
    real_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))

    # Iterate over the number of epochs
    for epoch in range(epochs):
        # Iterate over the batches of the dataset
        for batch in range(x_train.shape[0] // batch_size):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of images from the training set
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]
            
            # Generate a batch of fake images using the generator
            noise = np.random.normal(0, 1, (batch_size, gan.latent_dim))
            fake_images = generator.predict(noise)
            
            # Train the discriminator on real and fake images
            discriminator_real_loss = discriminator.train_on_batch(real_images, real_label)
            discriminator_fake_loss = discriminator.train_on_batch(fake_images, fake_label)
            
            # Calculate the total discriminator loss
            discriminator_loss = 0.5 * np.add(discriminator_real_loss, discriminator_fake_loss)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Generate a batch of noise vectors
            noise = np.random.normal(0, 1, (batch_size, gan.latent_dim))
            
            # Train the generator (via the gan model)
            gan_loss = gan.train_on_batch(noise, real_label)
            
            # Plot the progress
            print("Epoch {0}/{1}, Batch {2}/{3}, Discriminator Loss: {4}, Generator Loss: {5}".format(epoch+1, epochs, batch+1, x_train.shape[0] // batch_size, discriminator_loss, gan_loss))

    return gan, discriminator, generator

# Load the images from the three folders and combine them into one
dataset1_path = "/home/nallurn1/Anime_Code/anime_faces_crop"
dataset2_path = "/home/nallurn1/Anime_Code/avatar_crops"
dataset3_path = "/home/nallurn1/Anime_Code/naruto_crop"

images1 = load_images(dataset1_path)
images2 = load_images(dataset2_path)
images3 = load_images(dataset3_path)

# Resize the images to a common shape
size = (28, 28)
images1_resized = [np.array(Image.fromarray(img).resize(size)) for img in images1]
images2_resized = [np.array(Image.fromarray(img).resize(size)) for img in images2]
images3_resized = [np.array(Image.fromarray(img).resize(size)) for img in images3]

# Concatenate the three sets of images into one
images = np.concatenate((images1, images2, images3))

# Normlize pixel values to be between -1 and 1
images = (images - 127.5) / 127.5

# Reshape the images to match the input shape of the generator
images = np.reshape(images, (-1, 28, 28, 1)).astype('float32')

# Define the labels for the three datasets
num_images1 = len(images1)
num_images2 = len(images2)
num_images3 = len(images3)
labels1 = np.zeros(num_images1)
labels2 = np.ones(num_images2)
labels3 = np.ones(num_images3) * 2

# Concatenate the labels for the three datasets into one
labels = np.concatenate((labels1, labels2, labels3))

# Convert the labels to one-hot encoding
num_classes = 3
labels = keras.utils.to_categorical(labels, num_classes=num_classes)

# Shuffle the images and labels
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

# Split the data into training and testing sets
num_train = int(0.8 * len(images))
x_train, x_test = images[:num_train], images[num_train:]
y_train, y_test = labels[:num_train], labels[num_train:]

# Define the input shape of the generator
input_shape = (28, 28, 1)

# Create the generator
generator = make_generator_model(input_shape)

# Create the discriminator
discriminator = make_discriminator_model(input_shape, num_classes)

# Create the GAN
gan = make_gan_model(generator, discriminator)

# Compile the discriminator
discriminator.compile(
    loss=['binary_crossentropy', 'categorical_crossentropy'],
    optimizer=optimizer,
    metrics=['accuracy']
)

# Compile the combined model
combined.compile(
    loss=['binary_crossentropy', 'categorical_crossentropy'],
    optimizer=optimizer
)

# Train the GAN
train_gan(gan, discriminator, generator, x_train, y_train, epochs=100, batch_size=128)

