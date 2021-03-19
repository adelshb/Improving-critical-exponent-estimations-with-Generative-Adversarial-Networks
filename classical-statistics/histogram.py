# -*- coding: utf-8 -*-


import numpy as np
from math import log10


def hist(X, W=None, nbins=33, density=True, log_bin=True):

    xmin, xmax = X.min(), X.max()
    if xmin == xmax:
        return None, None, None

    if (log_bin):
        bins = np.logspace(log10(xmin), log10(xmax),
                           nbins, endpoint=True)

        hist, bin_edges = np.histogram(X, bins=bins,
                                       weights=W, density=density)

    else:
        hist, bin_edges = np.histogram(X, bins=nbins,
                     weights=W, density=density)

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_sizes = np.diff(bin_edges)

    indx = (hist > 0)

    return bin_centers[indx], hist[indx], bin_sizes[indx]

#========================================================
