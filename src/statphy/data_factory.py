# -*- coding: utf-8 -*-
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
import itertools

from src.statphy.models.percolation import percolation_configuration

# Example
# python statphy/data_factory.py \
#     --model square_lattice_percolation \
#     --L 64 128 \
#     --control_parameter 0.5 0.6 \
#     --samples 100

_available_models = [
    "square_lattice_percolation",
    ]

def main(args):

    PATH = os.path.join(args.path, "data")
    os.makedirs(PATH, exist_ok=True)

    if args.model == "square_lattice_percolation":

        for L, p in itertools.product(args.L, args.control_parameter):

            print ('Generating data for L={}, p={}'.format(L, p))

            finalpath = os.path.join(PATH, 'L_{}'.format(L), 'p_{}'.format(p))
            os.makedirs(finalpath, exist_ok=True)

            for i in range(args.samples):
                x = percolation_configuration(L, p)
                filepath = os.path.join(finalpath, '{}'.format(i))
                np.save(filepath, x)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model", type=str, default="square_lattice_percolation", 
                        choices=_available_models)

    # Model Parameters
    parser.add_argument("--L", type=int, nargs='+', default=[64, 128])
    parser.add_argument("--control_parameter", type=float, nargs='+', default=[0.5, 0.6])

    # Statistics
    parser.add_argument("--samples", type=int, default=10)

    # Save data
    parser.add_argument("--path", type=str, default=os.getcwd())

    args = parser.parse_args()
    main(args)