# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Train the HYDRA model."""

from argparse import ArgumentParser
import os
from datetime import datetime

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from generator import Generator
from discriminator import Discriminator
from hydra import Hydra

from src.statphy.models.percolation import percolation_configuration

from logger import Logger

def main(args, logger_plots = False):

    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))

    tbw_cnn = tf.summary.create_file_writer(save_dir + "/tensorboard/cnn_loss")
    tbw_gen_dis = tf.summary.create_file_writer(save_dir + "/tensorboard/gen_dis_dis")
    tbw_dis = tf.summary.create_file_writer(save_dir + "/tensorboard/dis_loss")
    print("Tensorboard command line: {}".format("tensorboard --logdir " + save_dir + "/tensorboard"))

    generator = Generator(args.noise_dim)
    generator_optimizer = tf.keras.optimizers.Adam(args.lr)
    discriminator = Discriminator()
    discriminator_optimizer = tf.keras.optimizers.Adam(args.lr)

    hydra = Hydra(generator = generator,
                generator_optimizer = generator_optimizer,
                discriminator = discriminator,
                discriminator_optimizer = discriminator_optimizer,
                discriminator_loss = tf.keras.losses.BinaryCrossentropy(),
                cnn = tf.keras.models.load_model(args.CNN_model_path),
                cnn_loss = tf.keras.losses.MeanAbsoluteError(),
                targeted_parameter = args.crit_parameter
                )
                
    real_images = tf.reshape(tf.constant([percolation_configuration(args.lattice_size, args.crit_parameter) for __ in range(args.batch_size)]), [args.batch_size, args.lattice_size, args.lattice_size, 1] )    

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    checkpoint_dir = os.path.join(save_dir, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # logger = Logger(save_dir=save_dir)

    for epoch in range(args.epochs):

        # logger.set_time_stamp(1)

        noise = tf.random.normal([args.batch_size, args.noise_dim], mean=args.noise_mean, stddev=args.noise_std)

        # Training the discriminator
        d_loss = hydra.train_discriminator_step(noise = noise, real_images= real_images)

        # Training the generator
        loss = hydra.train_generator_step(noise = noise, l_cnn=1, l_dis=1)

        loss["discriminator_loss"] = d_loss

        print(loss)
            
        with tbw_cnn.as_default():
            tf.summary.scalar("Loss", loss['cnn_loss'], step=epoch)
        with tbw_gen_dis.as_default():
            tf.summary.scalar("Loss", loss['generator_dis_loss'], step=epoch)
        with tbw_dis.as_default():
            tf.summary.scalar("Loss", loss['discriminator_loss'], step=epoch)
        
        # if (epoch + 1) % args.ckpt_freq == 0:
        #     checkpoint.save(file_prefix= checkpoint_prefix)

    #     logger.set_time_stamp(2)
    #     logger.logs['generator_loss'].append(gen_loss)
    #     logger.logs['generator_reg'].append(gen_reg)
    #     logger.logs['generator_signed_loss'].append(gen_signed_loss)
    #     logger.logs['val_loss'].append(val_loss)
    #     logger.logs['val_pred_mean'].append(val_mean_pred)
    #     logger.logs['val_pred_stddev'].append(val_stddev)
    #     logger.save_logs()

    #     if logger_plots == True:
    #         logger.generate_plots(generator= generator,
    #                             cnn= cnn,
    #                             epoch= epoch,
    #                             noise_dim= args.noise_dim,
    #                             bins_number= args.bins_number)
    #     logger.print_status(epoch=epoch)

    # tf.keras.models.save_model(generator, save_dir)
    # logger.save_metadata(vars(args))

if __name__ == "__main__":

    parser = ArgumentParser()

    # Model parameters 
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--noise_mean", type=float, default=0.0)
    parser.add_argument("--noise_std", type=float, default=1.0)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--reg_coeff", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Evaluation parameters
    parser.add_argument("--CNN_model_path", type=str, default="./saved_models/cnn/saved-model.h5")
    parser.add_argument("--bins_number", type=int, default=100)
    parser.add_argument("--crit_parameter", type=float, default=0.5928)
    parser.add_argument("--lattice_size", type=int, default=128)
    
    # Save parameters
    parser.add_argument("--save_dir", type=str, default="./saved_models/hydra")
    parser.add_argument("--ckpt_freq", type=int, default=10)
    
    args = parser.parse_args()
    main(args)
