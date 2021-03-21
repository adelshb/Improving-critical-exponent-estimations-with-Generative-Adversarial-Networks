# -*- coding: utf-8 -*-
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Square Lattice Percolation model."""

import numpy as np

def percolation_configuration(L, p):
    spin = (np.random.random(size=(L,L)) < p).astype(np.int8)
    return 2 * spin - 1

def read_percolation_data(L, p_arr, pc, max_configs_per_p=2000):
    X = []
    y = []
    j = 0
    for p in p_arr:
        label = j if p != pc else 0
        #label = 1 if p == pc else 0
        for i in range(max_configs_per_p):
            X.append(percolation_configuration(L, p))
            y.append(label)
        j += 1
    X = np.array(X).reshape(-1, L, L, 1)
    y = np.array(y).reshape(-1, )
    return X, y