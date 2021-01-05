# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:06:40 2021

@author: Austin Hsu
"""

import os
import pandas as pd
import torch
import numpy as np
from typing import Tuple
from .utils import AttributeDict #

class StockDataset(torch.utils.data.Dataset):
    
    def __init__(self, mode: str = 'Train', skip_preprocess: bool = False,
                 save_type: str = 'torch', train_percent: float = 1.,
                 time_step: int = 30, moving_average: int = 7,
                 data_dir: str = '../firms/', data_name: str = '2330台積電',
                 stat_dir: str = '../data_stats/'):
        
        super(StockDataset, self).__init__()
        
        # --- Args ---
        self.data_dir = data_dir
        self.data_name = data_name
        self.stat_dir = stat_dir
        self.mode = mode
        self.skip_preprocess = skip_preprocess
        self.save_type = save_type
        self.time_step = time_step
        self.moving_average = moving_average
        self.train_percent = train_percent # only useful for training dataset
        
        # --- Functions ---
        if self.save_type == 'torch':
            self.normalize = lambda x, min, max: x.sub(min).div(max-min)
        elif self.save_type == 'numpy' or self.save_type == 'prophet':
            self.normalize = lambda x, min, max: (x-min)/(max-min)
            self.ma_func = lambda x, w: np.convolve(x, np.ones((w,))/w, 'valid')
        else:
            raise ValueError(f'Given save_type {self.save_type} is not valid.')
        
        # --- Get Data ---
        self.data = pd.read_csv(os.path.join(self.data_dir, self.data_name+'.csv'), parse_dates=['Date'], index_col=0).dropna()
        self._preprocessing()
        
    def _preprocessing(self) -> None:
        
        if not self.skip_preprocess:
            train_data = self.data[self.data.index <= '2017-12-31'].copy()
            self.data_stat = {'min': torch.Tensor([train_data[i].min() for i in train_data.keys()]),
                              'max': torch.Tensor([train_data[i].max() for i in train_data.keys()])}
            self.label_stat = {'min': torch.Tensor([train_data.Close.min()]),
                               'max': torch.Tensor([train_data.Close.max()])}
            torch.save({'data_stat': self.data_stat, 'label_stat': self.label_stat}, os.path.join(self.stat_dir, f'{self.data_name}_stats.ckpt'))
        else:
            stat_ckpt = torch.load(os.path.join(self.stat_dir, f'{self.data_name}_stats.ckpt'))
            self.data_stat = stat_ckpt['data_stat']
            self.label_stat = stat_ckpt['label_stat']
        self.data_stat = AttributeDict(self.data_stat)
        self.label_stat = AttributeDict(self.label_stat)
            
        # --- Train Test Split ---
        if self.mode == 'Train':
            self.data = self.data[self.data.index <= '2017-12-31']
            self.data = self.data[:int(self.data.shape[0]*self.train_percent)]
        elif self.mode == 'Test':
            self.data = self.data[self.data.index >= '2020-01-01']
        elif self.mode == 'Valid':
            self.data = self.data[self.data.index >= '2018-01-01']
            self.data = self.data[self.data.index <= '2019-12-31']
        else:
            raise ValueError(f'Given mode {self.mode} is not valid.')
        self.data_keys = self.data.keys()
        self.data_index = self.data.index
        
        # --- To Numpy ---
        self.label = self.data.Close.values
        self.data = self.data.values
        
        # --- To Torch ---
        if self.save_type == 'torch':
            
            # --- Convert to torch.Tensor ---
            self.data = torch.from_numpy(self.data)
            self.label = torch.from_numpy(self.label)
            
            # --- Normalization ---
            self.data = self.normalize(self.data, **self.data_stat).float()
            self.label = self.normalize(self.label, **self.label_stat).float().unsqueeze(-1)
        
        # --- Numpy Data Conversion ---
        if self.save_type == 'numpy':
            
            # --- Convert to (num_feat*time_step) size ---
            self.data_stat = AttributeDict({'min': np.hstack([self.data_stat.min.numpy() for i in range(self.time_step)]),
                                            'max': np.hstack([self.data_stat.max.numpy() for i in range(self.time_step)])})
            self.label_stat = AttributeDict({'min': self.label_stat.min.numpy(), 'max': self.label_stat.max.numpy()})
            
            # --- Window Conversion ---
            self.data_index = self.data_index[self.time_step:self.data_index.shape[0]-self.moving_average+1]
            self.data = np.vstack([np.hstack(self.data[i:i+self.time_step]) for i in range(self.data.shape[0]-self.time_step-self.moving_average+1)])
            self.label = self.ma_func(self.label[self.time_step:], self.moving_average)
            
            # --- Normalization ---
            self.data = self.normalize(self.data, **self.data_stat).astype(np.float32)
            self.label = self.normalize(self.label, **self.label_stat).astype(np.float32)
        
        # --- fbprophet Data Conversion ---
        if self.save_type == 'prophet':
            
            # --- Convert to (num_feat*time_step) size ---
            self.label_stat = AttributeDict({'min': self.label_stat.min.numpy(), 'max': self.label_stat.max.numpy()})
            
            # --- Window Conversion ---
            self.data_index = self.data_index[self.time_step:self.data_index.shape[0]-self.moving_average+1]
            self.data = self.label[self.time_step-1:-self.moving_average].copy()
            self.label =self.ma_func(self.label[self.time_step:], self.moving_average)

            # --- Normalization ---
            self.data = self.normalize(self.data, **self.label_stat).astype(np.float32)
            self.label = self.normalize(self.label, **self.label_stat).astype(np.float32)
            
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index:index+self.time_step], self.label[index+self.time_step:index+self.time_step+self.moving_average].mean(dim=0)
        
    def __len__(self):
        return self.data.shape[0] - self.time_step - self.moving_average + 1