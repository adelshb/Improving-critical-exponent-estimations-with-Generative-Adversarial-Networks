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

"""Train GAN model."""

import dcgan

def main(args):

    ### TRAINING HERE ###

    return None

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()
    main(args)