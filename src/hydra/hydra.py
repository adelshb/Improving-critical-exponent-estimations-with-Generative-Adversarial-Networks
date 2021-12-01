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

from typing import Optional, Dict, Union

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Sequential
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

class Hydra():
    r"""Class for Hydra structure implementation.

    Implementation of the Hydra model. Takes 3 models:
    - Generator
    - Discriminator
    - CNN

    The Generator is trained via two losses. Typically, BCE from the Discriminator and MSE from the CNN.

    """

    def __init__(self, 
                generator: Sequential,
                generator_optimizer: Optimizer, 
                discriminator: Sequential,
                discriminator_optimizer: Optimizer, 
                discriminator_loss: Loss,
                cnn: Sequential,
                cnn_loss: Loss,
                targeted_parameter: float = 0.5928
                ) -> None:

        r"""
        Args:
            generator: Sequential,
            generator_optimizer: Optimizer, 
            discriminator: Sequential,
            discriminator_optimizer: Optimizer, 
            discriminator_loss: Loss,
            cnn: Sequential,
            cnn_loss: Loss,
            targeted_parameter: float = 0.5928
        """
        
        self.generator = generator
        self.generator_optimizer = generator_optimizer

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_loss = discriminator_loss

        self.cnn = cnn
        self.cnn_loss = cnn_loss
        
        self.targeted_parameter = targeted_parameter

    def loss(self,
            model: Sequential,
            loss_function: Loss,
            generated_images: Tensor,
            targeted_parameter: float
            ) -> float:
    
        r"""
        Args:
            model: Model to run on generated images (cnn or discriminator).
            loss_function: Loss fucntion (BCE or MSE).
            generated_images: Tensor of images generated by the generator.
            targeted_parameter: Single desired value for all generated images (0/1 for discriminator or targeted control parameter for the cnn).
        Return:
            the loss value.
        """

        predicted_output = model(generated_images) 
        wanted_output = tf.fill(predicted_output.shape, targeted_parameter)
        
        return loss_function(wanted_output, predicted_output)

    def regularization(generated_images: Tensor) -> float:

        shape = generated_images.shape
        reg = tf.math.reduce_sum(tf.fill(shape, 1, dtype=float) - tf.abs(generated_images))

        return reg / (shape[0]*shape[1]*shape[2])

    def train_step(self,
            noise: Tensor,
            real_images: Tensor,
            l_cnn: Optional[float] = 1,
            l_dis: Optional[float] = 1,
            ) -> Dict[str,float]:
        r"""
        Args:
            noise: Noise input for the Generator.
            real_images: Tensor of real images.
            l_cnn: Coefficient of the cnn loss in the total generator loss.
            l_dis: Coefficient of the discriminator loss in the total generator loss.
        Return:
            a dictionary containning both losses.
        """

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:

            # Generate images from input noise
            generated_images = self.generator(noise, training=True)

            # Compute the losses
            cnn_loss = self.loss(self.cnn, self.cnn_loss, generated_images, targeted_parameter= self.targeted_parameter)
            
            discriminator_loss = self.loss(self.discriminator, self.discriminator_loss, generated_images, targeted_parameter=1)
            gen_loss = l_cnn * cnn_loss + l_dis * discriminator_loss

            fake_loss = self.loss(self.discriminator, self.discriminator_loss, generated_images, targeted_parameter=0)
            real_loss = self.loss(self.discriminator, self.discriminator_loss, real_images, targeted_parameter=1)
            dis_loss = 0.5 * (fake_loss + real_loss)

        # Compute gradient and new weights
        gradients_of_discriminator = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Compute gradient and new weights
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return {"cnn_loss": cnn_loss, "generator_dis_loss": discriminator_loss, "discriminator_loss": dis_loss}

    def val_cnn_stats(self,
            error_function = tf.keras.losses.MeanAbsoluteError(),
            noise_dim: int = 100,
            test_size: int = 1000,
            noise_mean: Optional[float] = 0,
            noise_stddev: Optional[float] = 1.0,
            ) -> Dict[str,Union[float,Tensor]] :
        r"""
        Compute statics on a CNN model evaluation on the generated data.
        Args:
            loss_function: Function used to evaluate the images
            noise_dim: Noise iput size of the Generator
            test_size: Number of generated images used to collect statistics.
            noise_mean: Noise input mean
            noise_stddev: Noise inmput standard deviation.
        Return:
            Different statistically relevant values.
        """

        noise = tf.random.normal([test_size, noise_dim], mean=noise_mean, stddev=noise_stddev) 
        images = self.generator(noise, training=False)
        images = tf.sign(images)

        y_pred = self.cnn.predict(images)
        mean = tf.math.reduce_mean(y_pred)
        stddev = tf.math.reduce_std(y_pred)

        wanted_output = tf.fill(y_pred.shape, self.targeted_parameter)
        loss = error_function(wanted_output, y_pred)

        return {'val_loss':loss, 
                'val_mean_pred':mean, 
                'max_pred':tf.math.reduce_max(y_pred), 
                'min_pred':tf.math.reduce_min(y_pred), 
                'val_stddev':stddev
            }

    def load(self,
            ckpt_path: str
            ) -> None:
        r"""
        Load a latest checkpoint of hydra model.
        Args:
            checkpoint_path: Path to checkpoint folder
        """

        # from the DCGAN tutorial
        checkpoint = tf.train.Checkpoint(
            generator_optimizer= self.generator_optimizer,
            discriminator_optimizer= self.discriminator_optimizer,
            generator= self.generator,
            discriminator= self.discriminator,
        )

        latest = tf.train.latest_checkpoint(ckpt_path)
        checkpoint.restore(latest)

    def generate(self,
        sample_num: int,
        noise_dim: Optional[int] = 100,
        noise_mean: Optional[float] = 0.0,
        noise_stddev: Optional[float] = 1.0,
        signed: Optional[bool] = True
        ) -> Tensor:
        r"""
        Generate configuration with generative model.
        Args:
            sample_num: Number of configuration to generate
            noise_dim: Dimension of the noise for the generator.
            noise_mean: Mean value of tne noise. Default at 0.
            noise_stddev: Standard deviation of the noise. Default at 1.0.
        """

        noise = tf.random.normal([sample_num, noise_dim], mean=noise_mean, stddev=noise_stddev)
        images = self.generator(noise, training=False)
        
        if signed:
            images = tf.cast(tf.math.sign(images), tf.int8)

        return images