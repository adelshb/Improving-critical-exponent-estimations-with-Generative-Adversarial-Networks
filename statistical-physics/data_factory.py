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
import numpy as np

from models.slp import percolation_configuration

_available_models = [
    "square lattice percolation",
    ]

def main(args):

    X = []
    y = []
    param_range = np.arrange(args.param_range_start, args.param_range_end, args.param_range_delta)

    if args.model == "square lattice percolation":
        # Generate configuration within the selected range 
        for p in param_range:
            X.append([percolation_configuration(args.L, p) for __ in range(args.sample_per_configuration)])
            y.append([p] * args.sample_per_configuration)

        # Generate data for the critical parameter
        X.append([percolation_configuration(args.L, args.crit_parameter) for __ in range(args.sample_per_configuration)])
        y.append([args.crit_parameter] * args.sample_per_configuration)

        X = np.array(X).reshape(-1, args.L, args.L, 1)
        y = np.array(y).reshape(-1, )

    return X, y

if __name__ == "__main__":
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model", type=str, default="square lattice percolation", choices=_available_models)

    # Model Parameters
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--crit_parameter", type=float, default=0.5927)

    # Statistics
    parser.add_argument("--sample_per_configuration", type=int, default=100)
    parser.add_argument("--param_range_start", type=float(), default=0)
    parser.add_argument("--param_range_end", type=float(), default=1)
    parser.add_argument("--param_range_delta", type=float(), default=0.01)

    args = parser.parse_args()
    main(args)