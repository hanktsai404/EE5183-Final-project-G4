# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:47:41 2020

@author: AustinHsu
"""

from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.logger = SummaryWriter(log_dir=self.save_dir)
                
    def log(self, scalar_name, scalar, current_step):
        self.logger.add_scalar(scalar_name, scalar, current_step)
        
    def log_scalars(self, scalar_name, scalars, current_step):
        self.logger.add_scalars(scalar_name, scalars, current_step)
        
    def log_image(self, image_name, image, current_step):
        self.logger.add_image(image_name, image, current_step)
        
    def log_figure(self, figure_name, figure, current_step):
        self.logger.add_figure(figure_name, figure, current_step)
        
    def log_hparams(self, hparam_dict, metric_dict):
        self.logger.add_hparams(hparam_dict, metric_dict)
    
    def close(self):
        self.logger.close()