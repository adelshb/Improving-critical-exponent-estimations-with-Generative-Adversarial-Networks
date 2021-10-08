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
import os
from datetime import datetime

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings

from dcgan import make_generator_model
from utils import *
from logger import Logger

def main(args):

    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))

    generator = make_generator_model(args.noise_dim)
    cnn = tf.keras.models.load_model(args.CNN_model_path, custom_objects={'tf': tf})
    
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(1e-3)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
    checkpoint_dir = save_dir + '/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    logger = Logger(save_dir=save_dir)

"""    
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    cnn = tf.keras.models.load_model(args.CNN_model_path, custom_objects={'tf': tf})

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # for the generator and discriminator
    sparse_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy() # for the cnn loss

    generator_optimizer = tf.keras.optimizers.Adam(1e-3)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    checkpoint_dir = args.save_dir + '/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    os.makedirs(args.save_dir, exist_ok=True)

    loss_history = {"discriminator": [], "generator": [], "cnn": []}

    for epoch in range(args.epochs):
"""


    for epoch in range(args.epochs):

        logger.set_time_stamp(1)

        noise = tf.random.normal([args.batch_size, args.noise_dim], mean=0, stddev=1.0)

        gen_loss = train_step(generator= generator, 
                              cnn=cnn, 
                              generator_optimizer= generator_optimizer,  
                              cross_entropy= cross_entropy, 
                              noise= noise)

        if (epoch + 1) % args.ckpt_freq == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        logger.set_time_stamp(2)
        logger.logs['generator_loss'].append(gen_loss)
        logger.print_status(epoch=epoch)
        logger.save_logs()
        logger.generate_plots(generator=generator,
                              cnn=cnn,
                              epoch=epoch,
                              labels="saved_models/CNN_L128_N10000/labels.json",
                              noise_dim=args.noise_dim)
    
    tf.keras.models.save_model(generator, save_dir)
    
    logger.save_metadata(vars(args))

"""            gen_loss, disc_loss, cnn_loss = train_step(images=image_batch, 
                                                       generator=generator, 
                                                       discriminator=discriminator, 
                                                       cnn=cnn,
                                                       generator_optimizer=generator_optimizer, 
                                                       discriminator_optimizer=discriminator_optimizer, 
                                                       cross_entropy=cross_entropy, 
                                                       sparse_cross_entropy=sparse_cross_entropy,
                                                       noise_dim=args.noise_dim,
                                                       batch_size=args.batch_size, 
                                                       stddev=args.input_noise_stddev,
                                                       label_smoothing={'fake': args.label_smoothing_fake, 'real': args.label_smoothing_real},
                                                       waiting=args.waiting)

        loss_history["discriminator"].append(disc_loss)
        loss_history["generator"].append(gen_loss)
        loss_history["cnn"].append(cnn_loss)
        
        print("Epochs {}: generator loss:{:.4f}, discriminator loss:{:.4f}, cnn loss:{:.4f}, in {} sec.".format(epoch, gen_loss, disc_loss, cnn_loss, time.time()-start))

        #Save the model every args.save_ckpt epochs
        if (epoch + 1) % args.save_ckpt == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        # Some plots for logs
        plot_losses(loss_history, args.save_dir)
        plot_cnn_histogram(generator=generator,
                           cnn=cnn,
                           epoch=epoch,
                           save_dir=args.save_dir,
                           labels="saved_models/CNN_L128_N10000/labels.json",
                           noise_dim=args.noise_dim)

    tf.keras.models.save_model(generator, args.save_dir)
    """


if __name__ == "__main__":


    parser = ArgumentParser()

    # Data
    #parser.add_argument("--data_path", type=str, default="./data/simulation/L=128_p=0.5928.npz")
    #parser.add_argument("--CNN_model_path", type=str, default="./saved_models/CNN_L128_N10000/saved-model.h5")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--noise_dim", type=int, default=100)

    parser.add_argument("--save_dir", type=str, default="./saved_models/gan_cnn")
    parser.add_argument("--ckpt_freq", type=int, default=10)
    parser.add_argument("--CNN_model_path", type=str, default="./saved_models/CNN_L128_N10000/saved-model.h5")

    #parser.add_argument("--input_noise_stddev", type=float, default=0.3)
    #parser.add_argument("--label_smoothing_real", type=float, default=0.1)
    #parser.add_argument("--label_smoothing_fake", type=float, default=0)
    #parser.add_argument("--waiting", type=int, default=2)

    # Save model
    #parser.add_argument("--save_dir", type=str, default="./data/models/gan")
    #parser.add_argument("--save_ckpt", type=int, default=20)


    args = parser.parse_args()
    main(args)
