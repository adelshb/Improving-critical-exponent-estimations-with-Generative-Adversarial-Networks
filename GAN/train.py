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
import glob
import time
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings

#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf 

from dcgan import make_generator_model, make_discriminator_model
from utils import *

def main(args):

    train_images = []
    for filename in glob.glob(os.path.join(args.data_dir, '*.npy')):
        with open(os.path.join(os.getcwd(), filename), 'rb') as f: 
            train_images.append(np.load(f).reshape(128,128,1))
    train_labels = [0]*len(train_images)
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

    num_examples_to_generate = 1
    seed = tf.random.normal([num_examples_to_generate, args.noise_dim])
    noise = tf.random.normal([args.batch_size, args.noise_dim])

    checkpoint_dir = './data/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    if not os.path.exists('./data/generated/'):
        os.makedirs('./data/generated/')

    for epoch in range(args.epochs):
        start = time.time()

        for image_batch in train_dataset:
            train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy, noise)

        # Produce images for the GIF as you go
        predictions = generator(seed, training=False)
        np.save('./data/generated/image_at_epoch_{:04d}.png'.format(epoch), predictions[0])

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    predictions = generator(seed, training=False)
    np.save('./data/generated/image_at_epoch_{:04d}.png'.format(epoch), predictions[0])
    

if __name__ == "__main__":
    parser = ArgumentParser()

    # Data
    parser.add_argument("--data_dir", nargs=1, default=os.getcwd()+"/data/0.5928")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--noise_dim", type=int, default=100)
    # Save model
    parser.add_argument("--save_dir", nargs=1, default=os.getcwd())

    args = parser.parse_args()
    main(args)
