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

from typing import Dict

import os
import json
import time
import numpy as np

class Logger():
    """A helper class to better handle the saving of outputs."""
    
    def __init__(self,
                 save_dir: str,
                 ) -> None:
        """Constructor method for the Logger class.
        
        Args:
            save_dir: Path to the main checkpoint directory, in which the logs subdirectory is located.
                                  
        Returns:
            no value
        """
        
        self.save_dir = save_dir
        self.save_dir_logs = os.path.join(self.save_dir, "logs")
        os.makedirs(self.save_dir_logs, exist_ok=True)
        
        self.logs: dict = {}

        self._keys = ['cnn_loss', 'generator_dis_loss', 'discriminator_loss', 'val_loss', 'val_mean_pred', 'max_pred', 'min_pred', 'val_stddev']

        for k in self._keys:
            self.logs[k]=[]
        
        self.time_stamp = [0, 0]
            
    def set_time_stamp(self,
                       i: int,
                       ) -> None:
        """Method to keep track of time stamps for monitoring job progress"""
        
        self.time_stamp[i-1] = time.time()
                            
    def print_status(self,
                     epoch: int,
                     ) -> None:
        """Method to print on the status of the run on the standard output"""
        
        print('    - Episode: {:<13d} | Gen CNN loss : {:<13.4f} | Gen Dis loss: {:<13.4f} | Dis loss: {:<13.4f} | Val loss: {:<13.4f} | Duration (sec): {:<13.2f}'.format(epoch, 
                            self.logs["cnn_loss"][-1],
                            self.logs['generator_dis_loss'][-1],
                            self.logs['discriminator_loss'][-1],
                            self.logs['val_loss'][-1], 
                            self.time_stamp[1]-self.time_stamp[0])
                            )

    def update_logs(self, vals) -> None:
        
        for k in self._keys:
            self.logs[k].append(vals[k])

    def save_logs(self) -> None:
        """Saves all the necessary logs to 'save_dir_logs' directory."""
        
        for k in self._keys:
            array = np.array(self.logs[k])
            np.save(os.path.join(self.save_dir_logs, k + ".npy"), array)
        
    def save_metadata(self,
                      args: Dict,
                      ) -> None:
        """Method to save the command line arguments into a json file."""

        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as outfile:
            json.dump(args, outfile,  indent=2, separators=(',', ': '))