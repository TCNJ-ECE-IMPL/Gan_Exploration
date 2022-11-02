#Part 1 Creating the GAN Model

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
import optuna
import glob
import cv2

#Checking available 
print('Num of GPUs avaiable',len(tf.config.list_physical_devices('GPU')))
# tf.config.list_physical_devices('GPU')
# with tf.device('/GPU:0'):
#Size and the path of images

os.environ["CUDA_VISIBLE_DEVICES"]="0" 

gen_res = 3 #generation resolution

generate_square = 32 *gen_res #rows by cols, images will be as squares
image_channels = 3 #RGB values

# Preview image 
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

#size vector generate images from
seed_size = 100

#Configuration
data_path = '/home/nallurn1/VQGAN/resize'
# data_path_training = '/home/nallurn1/VQGAN/resized_images/trained/'

# for i in range(50):
#   img = cv2.imread(data_path)
#   if not os.path.isfile(data_path_training): 
#     cv2.imwrite((str(data_path_training) + '{}_training.jpg'.format(i)),img)

#Number of times network gets trained 
epochs = 5
#number of images read in 
batch_size = 32
#
buffer_size = 60000

print(f"Will generate {generate_square}px square images")

training_binary_path = os.path.join(data_path,f'training_data_{generate_square}_{generate_square}.npy')

print(f"Looking for file: {training_binary_path}")

if not os.path.isfile(training_binary_path):
  start = time.time()
  print("Loading training images...")

#Saving Training Data
  training_data = []
  faces_path = os.path.join(data_path)
  for filename in tqdm(os.listdir(faces_path)):
      path = os.path.join(faces_path,filename)
      # print(path,"TEST")
      if path != "/home/nallurn1/VQGAN/images/training_data_32_32.npy":
        if path != "/home/nallurn1/VQGAN/images/output":
          image = Image.open(path).resize((generate_square,generate_square),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  training_data = np.reshape(training_data,(-1,generate_square,generate_square,image_channels))
  training_data = training_data.astype(np.float32)
  training_data = training_data / 127.5 - 1.


  print("Saving training image binary...")
  np.save(training_binary_path,training_data)
  elapsed = time.time()-start
  print (f'Image preprocess time: {str(elapsed)}')
else:
  print("Loading previous training pickle...")
  training_data = np.load(training_binary_path)

#Tensorflow Dataset Object
train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(buffer_size).batch(batch_size)

def build_generator(seed_size, channels):
    # n_layers = trial.suggest_int("n_layers", 1, 3 )
    model = Sequential()

    model.add(Dense(4*4*256,activation="relu",input_dim=seed_size))
    model.add(Reshape((4,4,256)))

    # #Using Optuna
    # for i in range(n_layers):
    #     num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
    #     model.add(
    #         tf.keras.layers.Dense(
    #             num_hidden,
    #             activation="relu",
    #             kernel_regularizer=tf.keras.regularizers.l2(weight_decay),))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
   
    # Output resolution, additional upsampling
    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    if gen_res>1:
     model.add(UpSampling2D())
     model.add(Conv2D(128,kernel_size=3,padding="same"))
     model.add(BatchNormalization(momentum=0.8))
     model.add(Activation("relu"))
    

    # Final CNN layer
    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    #(Range from 1 to -1 outputs)
    model.add(Activation("tanh"))

    model.summary()
    return model


def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, 
                     padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    #Printing the model
    model.summary()
    return model

def save_images(cnt,noise):
  image_array = np.full(( PREVIEW_MARGIN + (PREVIEW_ROWS * (generate_square+PREVIEW_MARGIN)), PREVIEW_MARGIN + (PREVIEW_COLS * (generate_square+PREVIEW_MARGIN)), 3), 255, dtype=np.uint8)
  
  # noise comes from np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, seed_size))
  generated_images = generator.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(PREVIEW_ROWS):
      for col in range(PREVIEW_COLS):
        r = row * (generate_square+16) + PREVIEW_MARGIN
        c = col * (generate_square+16) + PREVIEW_MARGIN
        image_array[r:r+generate_square,c:c+generate_square] = generated_images[image_count] * 255
        image_count += 1

  output_path = os.path.join(data_path,'output')
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  filename = os.path.join(output_path,f"train_4-{cnt}.png")
  im = Image.fromarray(image_array)
  im.save(filename)

generator = build_generator(seed_size, image_channels)
noise = tf.random.normal([1, seed_size])
generated_image = generator(noise, training=False)
# plt.imshow(generated_image[0, :, :, 0])

#This will give us tensors
image_shape = (generate_square,generate_square,image_channels)

discriminator = build_discriminator(image_shape)
# decision = discriminator(generated_image)
# print(decision)

#How well it classifiys ML models. The loss is between 0 and 1
cross_entropy = tf.keras.losses.BinaryCrossentropy()

#Loss for disriminator: total loss = real_loss + fake_loss
def discriminator_loss(real_output, fake_output):
    #Looking for the 
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#Loss for generator: fake_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#Learning rates, ADAM: LR and momentum: .0002, .5
#Change to 0 
generator_optimizer = tf.keras.optimizers.Adam(.0015, .5)
discriminator_optimizer = tf.keras.optimizers.Adam(.0015,0.5)

@tf.function
def train_step(images):
  seed = tf.random.normal([batch_size, seed_size])

  #Graident Decsent
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(seed, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)
    #fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(
        gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(
        gradients_of_discriminator, 
        discriminator.trainable_variables))
  
  return gen_loss,disc_loss


#Starting Training
def train(dataset, epochs):
  fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, seed_size))
  print(fixed_seed)

  start = time.time()

  for epoch in range(epochs):
    epoch_start = time.time()

    gen_loss_list = []
    disc_loss_list = []

    for image_batch in dataset:
      t = train_step(image_batch)
      gen_loss_list.append(t[0])
      disc_loss_list.append(t[1])

    g_loss = sum(gen_loss_list) / len(gen_loss_list)
    d_loss = sum(disc_loss_list) / len(disc_loss_list)

    epoch_elapsed = time.time()-epoch_start
    print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss},'' {hms_string(epoch_elapsed)}')
    save_images(epoch,fixed_seed)

  elapsed = time.time()-start
  # print (f'Training time: {hms_string(elapsed)}')

train(train_dataset, epochs)
