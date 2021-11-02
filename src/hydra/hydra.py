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

from typing import Optional, Dict

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

    def regularization(generated_images: Tensor):

        shape = generated_images.shape
        reg = tf.math.reduce_sum(tf.fill(shape, 1, dtype=float) - tf.abs(generated_images))

        return reg / (shape[0]*shape[1]*shape[2])

    def train_generator_step(self,
                            noise: Tensor,
                            l_cnn: Optional[float] = 1,
                            l_dis: Optional[float] = 1,
                            ) -> Dict[str,float]:

        r"""
        Args:
            noise: Noise input for the Generator.
            l_cnn: Coefficient of the cnn loss in the total generator loss.
            l_dis: Coefficient of the discriminator loss in the total generator loss.
        Return:
            a dictionary containning both losses.
        """

        with tf.GradientTape() as gen_tape:

            # Generate images from input noise
            generated_images = self.generator(noise, training=True)

            # Compute the losses
            cnn_loss = self.loss(self.cnn, self.cnn_loss, generated_images, targeted_parameter= self.targeted_parameter)
            discriminator_loss = self.loss(self.discriminator, self.discriminator_loss, generated_images, targeted_parameter=1)
            loss = l_cnn * cnn_loss + l_dis * discriminator_loss

        # Compute gradient and new weights
        gradients_of_generator = gen_tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return {"cnn_loss": cnn_loss, "generator_dis_loss": discriminator_loss}

    def train_discriminator_step(self,
                            noise: Tensor,
                            real_images: Tensor
                            ) -> float:

        r"""
        Args:
            noise: Noise input for the Generator.
            real_images: Tensor of real images
        Return:
            The loss value.
        """

        with tf.GradientTape() as dis_tape:

            # Generate images from input noise
            generated_images = self.generator(noise, training=False)

            # Compute the losses
            fake_loss = self.loss(self.discriminator, self.discriminator_loss, generated_images, targeted_parameter= 1)
            real_loss = self.loss(self.discriminator, self.discriminator_loss, real_images, targeted_parameter=0)
            loss = 0.5 * (fake_loss + real_loss)

        # Compute gradient and new weights
        gradients_of_discriminator = dis_tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return loss