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
num_epochs_discriminator = 30
num_epochs_generator = 3
img_shape = (28, 28, 1)

# Define the generator model
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(7*7*128, input_dim=latent_dim),
    tf.keras.layers.Reshape((7, 7, 128)),
    tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'),
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

# Define the optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the checkpoint directory and filename
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Create the checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    save_freq=10*batch_size
)

# Train the model with the callback
for epoch in range(num_epochs_discriminator + num_epochs_generator):
    for images, labels in dataset:
        train_step(images, labels)
    # Save the model weights every 10 epochs
    if epoch % 10 == 0:
        generator.save_weights(checkpoint_prefix + '_generator_' + str(epoch))
        discriminator.save_weights(checkpoint_prefix + '_discriminator_' + str(epoch))

# Define a function to load and preprocess images
def load_images(folder_path, label):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(folder_path, filename)).convert('L')
            img = np.expand_dims(img, axis=-1)
            img = img.astype('float32') / 255
            images.append((img, label))
    return images

# Define a function to create the dataset
def create_dataset(folders, batch_size):
    # Load images from each folder and create a dataset
    datasets = []
    for label, folder in enumerate(folders):
        images = load_images(folder, label)
        dataset = tf.data.Dataset.from_generator(lambda: images, output_types=(tf.float32, tf.int32))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        datasets.append(dataset)
    # Combine all datasets and return the result
    #return tf.data.Dataset.zip(tuple(datasets)).repeat()
    return datasets

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


#Define the generator loss function
def generator_loss(fake_output, fake_labels):
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    classification_loss_fake = classification_loss(fake_labels, fake_output[:, :-1])
    total_loss = fake_loss + classification_loss_fake
    return total_loss

#Define the training loop
@tf.function
def train_step(images, labels):
    # Sample random noise for the generator input
    noise = tf.random.normal([batch_size, latent_dim])

    # Train the discriminator
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        disc_loss = discriminator_loss(real_output, fake_output, labels, tf.fill((batch_size,), num_classes))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Train the generator
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output, tf.fill((batch_size,), num_classes))

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

#Load the data and create the dataset
folders = ['/home/nallurn1/Anime_model/Gan_Exploration/anime_faces_reshape_imgs', '/home/nallurn1/Anime_model/Gan_Exploration/avatar_crops_reshape_imgs', '/home/nallurn1/Anime_model/Gan_Exploration/naruto_crop_reshape_imgs']
dataset = create_dataset(folders, batch_size)

#Train the model
for epoch in range(num_epochs_discriminator + num_epochs_generator):
    for images, labels in dataset:
        train_step(images, labels)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{num_epochs_discriminator + num_epochs_generator}')
