# -*- coding: utf-8 -*-
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Function to compute the cluster number density and the average cluster size
"""

import numpy as np
from scipy.ndimage import measurements
from tqdm import trange

def cluster_number_density(imgs, show_progress=True): 
    """Compute the clusters number density over a serie of configurations.
    Args:
        imgs: List of imgages with each image has are valuled +/- 1.
    Returns:
        nsp: Clusters number density.
        n_clusters: Number of clusters detected. 
    """
    
    all_area = np.array([], dtype=int)  
    
    if show_progress:
        myrange = trange(len(imgs))
    else:
        myrange = range(len(imgs))
    
    for i in myrange:
        img = imgs[i]

        # Detecting clusters area
        label = measurements.label(img>0)[0]
        area = measurements.sum(img>0, label, index=np.arange(label.max() + 1))
        area = area.astype(int)

        all_area = np.append(all_area, area)
    
    NsM = np.bincount(all_area)
    nsp = NsM.astype(float) / (imgs[0].shape[0]**2 * len(imgs))

    n_clusters = len(all_area)
    
    return nsp, n_clusters
    
def average_cluser_size(imgs, show_progress=True): 
    """Compute the average clusters over a serie of configurations.
    Args:
        imgs: List of imgages with each image has are valuled +/- 1.
    Returns:
        S: average cluster size over all the images.
    """
    
    if show_progress:
        myrange = trange(len(imgs))
    else:
        myrange = range(len(imgs))
    
    S = 0
    for i in myrange:
        img = imgs[i]

        # Detecting clusters area
        label = measurements.label(img>0)[0]
        area = measurements.sum(img>0, label, index=np.arange(label.max() + 1))
        area = area.astype(int)
        
        # Remove spanning cluster by setting its area to zero
        perc_x = np.intersect1d(label[0,:], label[-1,:]) # spanning along x axis
        perc = perc_x[np.where(perc_x > 0)] 
        if len(perc) > 0: 
            area[perc[0]] = 0
        S += np.sum(area * area)  
    
    return S / (imgs[0].shape[0]**2 * len(imgs))