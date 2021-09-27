# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

from typing import Dict, Optional

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Sequential
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

import matplotlib.pyplot as plt

def generator_loss(cross_entropy: Loss, fake_output: Tensor):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(cross_entropy: Loss, real_output: Tensor, fake_output: Tensor, 
                       label_smoothing: Dict = {'fake': 0.0, 'real': 0.0}):
    
    real_loss = cross_entropy(tf.ones_like(real_output) - label_smoothing['real'], real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + label_smoothing['fake'], fake_output)
    total_loss = 0.5 * (real_loss + fake_loss)
    
    return total_loss

def train_step(images: Tensor, 
               generator: Sequential, 
               discriminator: Sequential, 
               generator_optimizer: Optimizer, 
               discriminator_optimizer: Optimizer, 
               cross_entropy: Loss, 
               noise_dim: int,
               batch_size: int,
               stddev: Optional[float] = 0.5,
               label_smoothing: Dict = {'fake': 0.0, 'real': 0.0}):

    ### Training the discriminator

    with tf.GradientTape() as disc_tape:
    
        noise = tf.random.normal([batch_size, noise_dim])
        generated_images = generator(noise, training=True)

        # Adding Gaussian noise to all images 
        images = images + tf.random.normal(shape=images.shape, stddev=stddev)
        generated_images = generated_images + tf.random.normal(shape=generated_images.shape, stddev=stddev)

        # Discriminator predictions
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        disc_loss = discriminator_loss(cross_entropy, real_output, fake_output, label_smoothing)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    #from IPython import embed; embed()
    
    ### Training the generator
    
    with tf.GradientTape() as gen_tape:
    
        for _ in range(2):
            noise = tf.random.normal([batch_size, noise_dim])
            generated_images = generator(noise, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = generator_loss(cross_entropy, fake_output)
            
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss, disc_loss

def read_npy_file(item):
    data = np.load(item)
    return data.reshape(128,128,1).astype(np.float32)

def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('./data/generated/image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()
