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

import dcgan

def main(args):

    ### TRAINING HERE ###

    return None

if __name__ == "__main__":
    parser = ArgumentParser()

    # Data
    parser.add_argument("--data_dir", nargs=1, default=os.getcwd()+"/data/")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)

    # Save model
    parser.add_argument("--save_dir", nargs=1, default=os.getcwd())

    args = parser.parse_args()
    main(args)