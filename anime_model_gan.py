import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# Define the parameters of the model
latent_dim = 100
num_classes = 3
batch_size = 32 #from 128
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
        # image = keras.preprocessing.image.load_img(image_path, target_size=img_shape[:2])
        #The size is (28, 28, 3)
        image = keras.preprocessing.image.load_img(image_path, target_size=(28, 28))
        image = keras.preprocessing.image.img_to_array(image) / 255.0
    
        # Resize the image to the desired shape
        image = cv2.resize(image, (img_shape[1], img_shape[0]))

        images.append(image)
        labels.append(dataset_labels[i])

print("Data type:", np.shape(images))
print("Shape:", np.shape(labels))

images = np.array(images)
labels = np.array(labels)

# tf.keras.layers.Conv2DTranspose(
#     filters,
#     kernel_size,
#     strides=(1, 1),
#     padding='valid',
#     output_padding=None,
#     data_format=None,
#     dilation_rate=(1, 1),
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )

# Define the generator model
noise_input = keras.layers.Input(shape=(latent_dim))

#label_input = keras.layers.Input(shape=(1))
label_input = keras.layers.Input(shape=(1))
# # label_embedding = keras.layers.Embedding(num_classes, latent_dim)(label_input)
label_embedding = keras.layers.Embedding(num_classes, latent_dim)(label_input)
label_embedding = keras.layers.Flatten()(label_embedding)
# # model_input = keras.layers.multiply([noise_input, label_embedding])
#model_input = keras.layers.concatenate([noise_input, label_embedding[1,:]],axis=1 )
model_input = keras.layers.concatenate([noise_input, label_embedding],axis=1)

# # x = keras.layers.Dense(1024, input_dim=latent_dim)(model_input) # output is 1024 values
x = keras.layers.Dense(1024, input_dim=latent_dim+2)(model_input) # output is 1024 values
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Dense(128 * 28 * 28)(x)    # output is 32768 values, could make it 28 * 28 * 128, and then no stride (assumed 1x1)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Reshape((28, 28, 128))(x)  # output is 16x16x128
x = keras.layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same')(x)  # output should be 32x32x128
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same')(x)  # output should be 64x64x128
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same')(x)  # output should be 128x128x128
x = keras.layers.BatchNormalization()(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
# #Tanh or sigmond?
# #Tanh: -1 to 1, using (gen_imgs +1)/2
# #Sigmond: 0 to 1
x = keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)          # output should be 128x128x3
#output = x
#output = keras.layers.Reshape((28, 28, 3))(x)
# generator = keras.models.Model(inputs=[noise_input, label_input], outputs=output)
generator = keras.models.Model(inputs=[noise_input, label_input], outputs=x)
print(generator)

discriminator = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(0.4),
    #Changing the size from 128 to 64
    keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding = 'same'),
    #keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding = 'same'),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation='sigmoid')
])

# print the model summaries
generator.summary()
discriminator.summary()

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

# Define the ACGAN model
#.trainable => 
discriminator.trainable = True
# Define the inputs for the generator
gan_input_noise = tf.keras.Input(shape=(latent_dim,), name='noise')
#gan_input_label = tf.keras.Input(shape=(1,), name='label'), possible issue
gan_input_label = tf.keras.Input(shape=(1,), name='label')

# Get the output from the generator
gan_output = discriminator(generator([gan_input_noise, gan_input_label]))

# Create the GAN
gan = tf.keras.models.Model(inputs=[gan_input_noise, gan_input_label], outputs=gan_output)

# Compile the GAN
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# Train the ACGAN (GAN) model
for epoch in range(num_epochs_generator):
    #Train the discriminator
    for _ in range(num_epochs_discriminator):
        # Select a random batch of images and labels
        idx = np.random.randint(0, images.shape[0], batch_size)
        real_images = images[idx]
        real_labels = labels[idx]
        print(batch_size)
        
        # Generate fake images and labels
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        print(noise.shape)
        # fake_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
        fake_labels = np.random.randint(0, num_classes, batch_size)
        
        #inputs=['tf.Tensor(shape=(32, 100), dtype=float32)', 'tf.Tensor(shape=(100, 2), dtype=float32)']
        fake_images = generator.predict([noise, fake_labels])
        
        #train_on_batch() keras 
        # Train the discriminator on real and fake images and labels
        discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
        
    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    # fake_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
    fake_labels = np.random.randint(0, num_classes, batch_size )

    print(noise.shape, fake_labels.shape )

    gan_loss = gan.train_on_batch([noise, fake_labels], [np.ones((batch_size, 1)), fake_labels])
    

    print(f"Epoch {epoch+1}, Discriminator Loss: {discriminator_loss_real, discriminator_loss_fake}, GAN Loss: {gan_loss}")
    
    # Save the generated images
    os.makedirs('gen_images_1', exist_ok=True)
    for i in range(num_classes):
        label = np.array([i] * batch_size).reshape(-1, 1)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict([noise, label])
        print(np.shape(generated_images))
        for j in range(batch_size):
            plt.imsave(f"gen_images_1/epoch_{epoch+1}_class_{i}_img_{j}.png",  generated_images)