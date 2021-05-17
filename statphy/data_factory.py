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

"""Generate data for specify model."""

from argparse import ArgumentParser
import os 

import numpy as np

from models.percolation import percolation_configuration

# Example
# python statphy/data_factory.py \
#     --model square lattice percolation \
#     --L 128 \
#     --crit_parameter 0.5927 \
#     --sample_per_configuration 100

_available_models = [
    "square_lattice_percolation",
    ]

def main(args):

    PATH = args.path + "/data/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    if args.model == "square_lattice_percolation":
        # Generate configurations for critical parameter
        if args.crit_parameter:
            if not os.path.exists(PATH + "/" + str(args.crit_parameter) + "/"):
                    os.makedirs(PATH + "/" + str(args.crit_parameter) + "/")
            for i in range(args.sample_per_configuration):
                x = percolation_configuration(args.L, args.crit_parameter)

                filename = str(args.crit_parameter) + "_" + str(i)
                np.save(os.path.join(PATH + "/" + str(args.crit_parameter) + "/", filename), x)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model", type=str, default="square_lattice_percolation", choices=_available_models)

    # Model Parameters
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--crit_parameter", type=float)

    # Statistics
    parser.add_argument("--sample_per_configuration", type=int, default=10)

    # Save data
    parser.add_argument("--path", nargs=1, default=os.getcwd())

    args = parser.parse_args()
    main(args)