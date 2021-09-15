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

"""Train the GAN model."""

from argparse import ArgumentParser
import time
import numpy as np
import os

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings

from dcgan import make_generator_model, make_discriminator_model
from utils import *

def main(args):

    with np.load(args.data_path) as data:
        train_images = data['arr_0']
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(args.batch_size)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    noise = tf.random.normal([args.batch_size, args.noise_dim])

    checkpoint_dir = args.save_dir + '/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    for epoch in range(args.epochs):

        # Input Gaussian noise schedule
        stddev = 0.5/(epoch+1)

        start = time.time()
        for image_batch in train_dataset:

            gen_loss, disc_loss = train_step(images= image_batch, 
                                                generator= generator, 
                                                discriminator= discriminator, 
                                                generator_optimizer= generator_optimizer, 
                                                discriminator_optimizer= discriminator_optimizer, 
                                                cross_entropy= cross_entropy, 
                                                noise= noise, 
                                                stddev= stddev)

        print("Epochs {}: generator loss:{}, discriminator loss:{} in {} sec.".format(epoch, gen_loss, disc_loss, time.time()-start))

        #Save the model every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tf.keras.models.save_model(generator, args.save_dir)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Data
    parser.add_argument("--data_path", type=str, default="./data/simulation/L=128_p=0.5928.npz")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--noise_dim", type=int, default=100)
    # Save model
    parser.add_argument("--save_dir", type=str, default="./data/models/gan")

    args = parser.parse_args()
    main(args)
