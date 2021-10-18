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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dcgan import make_generator_model
from utils import train_step, val_pred_loss
from logger import Logger

def main(args, logger_plots = False):

    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))

    tensorboard_writer_gen_loss = tf.summary.create_file_writer(save_dir + "/tensorboard/gen_loss")
    tensorboard_writer_gen_reg = tf.summary.create_file_writer(save_dir + "/tensorboard/gen_reg")
    tensorboard_writer_gen_signed_loss = tf.summary.create_file_writer(save_dir + "/tensorboard/gen_signed_loss")
    tensorboard_writer_val_loss = tf.summary.create_file_writer(save_dir + "/tensorboard/val_loss")
    print("Tensorboard command line: {}".format("tensorboard --logdir " + save_dir + "/tensorboard"))

    generator = make_generator_model(args.noise_dim)
    cnn = tf.keras.models.load_model(args.CNN_model_path)
    
    loss_function = tf.keras.losses.MeanAbsoluteError()
    generator_optimizer = tf.keras.optimizers.Adam(args.lr)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
    checkpoint_dir = os.path.join(save_dir, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    logger = Logger(save_dir=save_dir)

    for epoch in range(args.epochs):

        logger.set_time_stamp(1)

        noise = tf.random.normal([args.batch_size, args.noise_dim], mean=args.noise_mean, stddev=args.noise_std)

        (gen_loss, gen_reg), gen_signed_loss= train_step(generator= generator, 
                              cnn= cnn, 
                              generator_optimizer= generator_optimizer,  
                              loss_function= loss_function, 
                              noise= noise,
                              l= args.reg_coeff)

        val_loss, val_mean_pred, val_stddev = val_pred_loss(generator = generator,
                                cnn = cnn,
                                loss_function = tf.keras.losses.MeanAbsoluteError(),
                                wanted_output = 0.5928,
                                noise_dim = 100,
                                test_size = 1000,
                                noise_mean = 0,
                                noise_stddev = 1.0)
            
        with tensorboard_writer_gen_loss.as_default():
            tf.summary.scalar("Loss", gen_loss, step=epoch)
        with tensorboard_writer_gen_reg.as_default():
            tf.summary.scalar("Loss", gen_reg, step=epoch)
        with tensorboard_writer_gen_signed_loss.as_default():
            tf.summary.scalar("Loss", gen_signed_loss, step=epoch)
        with tensorboard_writer_val_loss.as_default():
            tf.summary.scalar("Loss", val_loss, step=epoch)
        
        if (epoch + 1) % args.ckpt_freq == 0:
            checkpoint.save(file_prefix= checkpoint_prefix)

        logger.set_time_stamp(2)
        logger.logs['generator_loss'].append(gen_loss)
        logger.logs['generator_reg'].append(gen_reg)
        logger.logs['generator_signed_loss'].append(gen_signed_loss)
        logger.logs['val_loss'].append(val_loss)
        logger.logs['val_pred_mean'].append(val_mean_pred)
        logger.logs['val_pred_stddev'].append(val_stddev)
        logger.save_logs()

        if logger_plots == True:
            logger.generate_plots(generator= generator,
                                cnn= cnn,
                                epoch= epoch,
                                noise_dim= args.noise_dim,
                                bins_number= args.bins_number)
        logger.print_status(epoch=epoch)

    tf.keras.models.save_model(generator, save_dir)
    logger.save_metadata(vars(args))

if __name__ == "__main__":

    parser = ArgumentParser()

    # Model parameters 
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--noise_mean", type=float, default=0.0)
    parser.add_argument("--noise_std", type=float, default=1.0)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--reg_coeff", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Evaluation parameters
    parser.add_argument("--CNN_model_path", type=str, default="./saved_models/CNN_L128_N10000/saved-model.h5")
    parser.add_argument("--bins_number", type=int, default=100)
    
    # Save parameters
    parser.add_argument("--save_dir", type=str, default="./saved_models/gan_cnn_regression")
    parser.add_argument("--ckpt_freq", type=int, default=10)
    

    args = parser.parse_args()
    main(args)
