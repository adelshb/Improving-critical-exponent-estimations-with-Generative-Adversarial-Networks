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

"""Train the GAN model."""

from argparse import ArgumentParser
import time
import os

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings

from dcgan import make_generator_model
from utils import *

def main(args):

    generator = make_generator_model(args.noise_dim)
    cnn = tf.keras.models.load_model(args.CNN_model_path, custom_objects={'tf': tf})
    
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(1e-3)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
    checkpoint_dir = args.save_dir + '/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    loss_history = {'generator': []}

    for epoch in range(args.epochs):

        start = time.time()

        noise = tf.random.normal([args.batch_size, args.noise_dim], mean=0, stddev=1.0)

        gen_loss = train_step(generator= generator, 
                              cnn=cnn, 
                              generator_optimizer= generator_optimizer,  
                              cross_entropy= cross_entropy, 
                              noise= noise, 
                              )

        print("Epochs {}: generator loss:{:4f}, in {} sec.".format(epoch, gen_loss, time.time()-start))

        if (epoch + 1) % args.ckpt_freq == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

    loss_history['generator'].append(gen_loss)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    plot_cnn_histogram(generator=generator,
                       cnn=cnn,
                       epoch=epoch,
                       save_dir=args.save_dir,
                       labels="saved_models/CNN_L128_N10000/labels.json",
                       noise_dim=args.noise_dim)
    plot_losses(loss_history=loss_history,
                save_dir=args.save_dir)

    tf.keras.models.save_model(generator, args.save_dir)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Data
    # parser.add_argument("--data_path", type=str, default="./data/simulation/L=128_p=0.5928.npz")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--noise_dim", type=int, default=100)

    # Save model
    parser.add_argument("--save_dir", type=str, default="./saved_models/gan_cnn-loss")
    parser.add_argument("--ckpt_freq", type=int, default=10)
    
    parser.add_argument("--CNN_model_path", type=str, default="./saved_models/CNN_L128_N10000/saved-model.h5")

    args = parser.parse_args()
    main(args)
