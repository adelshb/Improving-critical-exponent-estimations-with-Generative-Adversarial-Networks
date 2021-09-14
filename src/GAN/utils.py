# -*- coding: utf-8 -*-
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from re import S
import numpy as np
import tensorflow as tf

import time
import matplotlib.pyplot as plt

import os

def generator_loss(cross_entropy, fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(cross_entropy, real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy, noise, freeze=False):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        images = images + tf.random.normal(shape=images.shape, stddev=1.0)
        generated_images = generated_images + tf.random.normal(shape=generated_images.shape, stddev=1.0)
        
        #images = tf.random.normal(shape=images.shape, stddev=5.0)
        #generated_images = tf.random.normal(shape=generated_images.shape, stddev=5.0)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        #real_output = tf.where(real_output==1, 0.9, 0)
        #fake_output = tf.where(fake_output==1, 0.9, 0)

        gen_loss = generator_loss(cross_entropy, fake_output)
        disc_loss = discriminator_loss(cross_entropy, real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    if not freeze:
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return gen_loss, disc_loss
    else:
        return gen_loss, None

    # print(real_output, fake_output)
    # exit()


    

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
