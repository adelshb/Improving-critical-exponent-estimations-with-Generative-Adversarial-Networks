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

def generator_loss(cross_entropy: Loss, 
                   generated_images: Tensor,
                   cnn: Sequential,
                   category: int = 24):
    
    predicted_probabilities = cnn(generated_images) 
    wanted_output = np.full(predicted_probabilities.shape[0], category, dtype=int)
    
    return cross_entropy(wanted_output, predicted_probabilities)

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

def train_step(generator: Sequential, 
               cnn: Sequential, 
               generator_optimizer: Optimizer, 
               cross_entropy: Loss, 
               noise: Tensor, 
               ):

    with tf.GradientTape() as gen_tape:
        
        generated_images = generator(noise, training=True)
        gen_loss = generator_loss(cross_entropy, generated_images, cnn)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss

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
