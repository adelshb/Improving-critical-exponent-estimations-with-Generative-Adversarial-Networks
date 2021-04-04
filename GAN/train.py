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
import os 
import glob

import numpy as np

from dcgan import make_generator_model, make_discriminator_model
from utils import *

def main(args):

    train_images = []
    for filename in glob.glob(os.path.join(args.data_dir[0], '*.npy')):
        with open(os.path.join(os.getcwd(), filename), 'rb') as f: 
            train_images.append(np.load(f))
    train_labels = [0]*len(train_images)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(args.batch_size)

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    train(train_dataset, generator, discriminator, args.epochs, args.batch_size, args.noise_dim)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Data
    parser.add_argument("--data_dir", nargs=1, default=os.getcwd()+"/data/")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--noise_dim", type=int, default=100)
    # Save model
    parser.add_argument("--save_dir", nargs=1, default=os.getcwd())

    args = parser.parse_args()
    main(args)
