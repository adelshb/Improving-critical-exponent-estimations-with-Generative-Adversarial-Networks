# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi and Matthieu Sarkis, https://github.com/adelshb, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import json
import os

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

def cnn_loss(sparse_cross_entropy: Loss, 
             images: Tensor,
             cnn: Sequential,
             bin: int = 24):
    
    predictions = cnn(images) 
    wanted_output = np.full(predictions.shape[0], bin, dtype=int)
    
    return sparse_cross_entropy(wanted_output, predictions)

def plot_cnn_histogram(generator: Sequential,
                       cnn: Sequential,
                       epoch: int,
                       save_dir: str,
                       labels: str = "saved_models/CNN_L128_N10000/labels.json",
                       noise_dim: int = 100):
    
    with open(labels, 'r') as f:
        labels = json.load(f)
    reversed_labels = {value : float(key) for (key, value) in labels.items()}
    
    noise = tf.random.normal([1600, noise_dim])

    with open("saved_models/CNN_L128_N10000/labels.json", 'r') as f:
        labels = json.load(f)
    reversed_labels = {value : float(key) for (key, value) in labels.items()}
    
    noise = tf.random.normal([100, noise_dim])
    images = generator(noise, training=False)
    
    y_pred = cnn.predict(images).argmax(axis=1)
    y_pred = [reversed_labels[i] for i in y_pred]
    
    fig, ax = plt.subplots(1, 1)
    ax.hist(y_pred, color='b')
    ax.set_title("Distribution of the value of p for GAN generated critical configurations")
    path = os.path.join(save_dir, "histograms/")
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + "generatedImages_epoch{}.png".format(epoch))

def plot_losses(losses_history: Dict,
                figure_file: str):
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 7)
    ax.plot(losses_history["generator_loss"], label='generator')
    ax.grid(True)
    ax.legend()
    ax.set_title("Generator Loss history")
    fig.savefig(figure_file)

def plot_losses(loss_history: Dict,
                save_dir: str):
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 7)
    ax.plot(loss_history["generator"], label='generator')
    ax.plot(loss_history["discriminator"], label='discriminator')
    ax.plot(loss_history["cnn"], label='cnn')
    ax.grid(True)
    ax.legend()
    ax.set_title("Losses history")
    fig.savefig(save_dir + "/losses.png")

def train_step(images: Tensor, 
               generator: Sequential, 
               discriminator: Sequential, 
               cnn: Sequential,
               generator_optimizer: Optimizer, 
               discriminator_optimizer: Optimizer,
               cross_entropy: Loss, 
               noise: Tensor, 
               sparse_cross_entropy: Loss,
               noise_dim: int,
               batch_size: int,
               stddev: Optional[float] = 0.5,
               label_smoothing: Dict = {'fake': 0.0, 'real': 0.0},
               waiting: int = 2):

    ### Training the discriminator

    with tf.GradientTape() as disc_tape:
    
        noise = tf.random.normal([batch_size, noise_dim])
        generated_images = generator(noise, training=True)

    # compute the CNN predicitions for logging, not used in training yet
    CNN_loss = cnn_loss(sparse_cross_entropy=sparse_cross_entropy, 
                        images=images, 
                        cnn=cnn)
    # Adding Gaussian noise to all images 
    images = images + tf.random.normal(shape=images.shape, stddev=stddev)
    generated_images = generated_images + tf.random.normal(shape=generated_images.shape, stddev=stddev)
    # Discriminator predictions
    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)
    disc_loss = discriminator_loss(cross_entropy, real_output, fake_output, label_smoothing)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    ### Training the generator
    
    for _ in range(waiting):

        with tf.GradientTape() as gen_tape:
        
            noise = tf.random.normal([batch_size, noise_dim])
            generated_images = generator(noise, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = generator_loss(cross_entropy, fake_output)
            
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    # from IPython import embed; embed()

    return gen_loss, disc_loss, CNN_loss

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
  