# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 20:46:42 2021

@author: Austin Hsu
"""

import torch
import torch.nn as nn
import os
from collections import OrderedDict
from argparse import Namespace
from .dataset import StockDataset #
from .model import RNNModel, TemporalConvNet #
from .utils import AttributeDict #

class NNStockSolver:
    """
    hparams:
        method
        feat_num
        time_step
        moving_average
        stat_dir
        train_percent
        data_dir
        data_name
        batch_size
        shuffle
        num_workers
        pin_memory
    """
    def __init__(self, hparams: Namespace) -> None:
        self.hparams = hparams
        self.recovery = lambda x, min, max: x.mul(max-min).add(min)
        self.__buildmodel()
    
    def __buildmodel(self):
        assert self.hparams.method.split('_')[0] in ['RNN', 'LSTM', 'GRU', 'TCN'], f"Given method {self.hparams.method} is not available."
        if self.hparams.method.split('_')[0] == 'TCN':
            self.model = TemporalConvNet(*self.hparams.method.split('_')[1:], input_size=self.hparams.feat_num)
        else:
            self.model = RNNModel(*self.hparams.method.split('_'), input_size=self.hparams.feat_num)
        self.loss = nn.MSELoss()
        
    def get_pred(self, batch, batch_idx):
        # --- get data ---
        x, y = batch

        # --- forward ---
        y_hat = self.model(x)
        
        # --- to cpu ---
        y = y.detach().cpu()
        y_hat = (y_hat+x[:,-1,0].unsqueeze(-1)).detach().cpu()

        # --- label recovery ---
        y = self.recovery(y, **self.label_stat).numpy()
        y_hat = self.recovery(y_hat, **self.label_stat).numpy()
        
        # --- output ---
        output = {'pred': y_hat, 'target': y}
        return output
    
    def training_step(self, batch, batch_idx):
        # --- get data ---
        x, y = batch

        # --- forward ---
        y_hat = self.model(x) # y_pred = x[:,-1,0] (close of last day) + y_hat (noise calculated)

        # --- loss ---
        train_loss = self.loss(y_hat+x[:,-1,0].unsqueeze(-1), y)

        # --- output ---
        tqdm_dict = {'train_loss': train_loss.item()}
        output = OrderedDict({
            'loss': train_loss,
            'progress_bar': AttributeDict(tqdm_dict),
        })
        return output
        
    def test_step(self, batch, batch_idx):
        # --- get data ---
        x, y = batch

        # --- forward ---
        y_hat = self.model(x)
        
        # --- loss ---
        test_loss = self.loss(y_hat, y-x[:,-1,0].unsqueeze(-1))
        
        # --- output ---
        tqdm_dict = {'test_loss': test_loss.item()}
        output = OrderedDict({
            'loss': test_loss,
            'progress_bar': AttributeDict(tqdm_dict),
        })
        return output
        
    def configure_optimizers(self, lr: float = 0.001):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = None # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return optimizer, scheduler
    
    def __dataloader(self, mode: str = 'Train'):
        dataset = StockDataset(
            mode=mode,
            skip_preprocess=os.path.exists(os.path.join(self.hparams.stat_dir, f"{self.hparams.data_name}_stats.ckpt")),
            save_type='torch',
            time_step=self.hparams.time_step,
            moving_average=self.hparams.moving_average,
            train_percent=self.hparams.train_percent,
            data_dir=self.hparams.data_dir,
            data_name=self.hparams.data_name,
            stat_dir=self.hparams.stat_dir,
            )
        self.label_stat = dataset.label_stat
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=self.hparams.shuffle if mode=='Train' else False,
                                           num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
    
    def train_dataloader(self):
        return self.__dataloader(mode='Train')
    
    def valid_dataloader(self):
        return self.__dataloader(mode='Valid')
        
    def test_dataloader(self):
        return self.__dataloader(mode='Test')
    
    def load_from_checkpoint(self, checkpoint: str, device: torch.device):
        self.checkpoint = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(self.checkpoint['model'])
