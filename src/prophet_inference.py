# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:13:39 2021

@author: AustinHsu
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from argparse import Namespace
from .dataset import StockDataset

class StockInference:
    
    def __init__(self, hparams=None):
        self.hparams = Namespace() #hparams
        self.hparams.method = 'Prophet'
        self.hparams.checkpoint_path = './checkpoint/'
        self.hparams.stat_dir = './data_stats/'
        self.hparams.data_dir = './firms/'
        self.hparams.firm_table = './data/Market_Value_Table.csv'
        
        self.recovery = lambda x, min, max: x*(max-min)+min
        
        # --- Available Firms ---
        self.firm_list = self._get_firm_list(self.hparams.firm_table)
        
        # --- Available Dates ---
        # available dates: 2020-03-10 ~ 2020-10-21
        
    def get_weights(self, pred_date: str = '2020-03-10', save_loc: str = './') -> None:
        close_firmlist = []
        close_incrlist = []
        close_predlist = []
        close_origlist = []
        for firm in tqdm(self.firm_list):
            close_increase, close_pred, close_orig = self._prophet_inference(firm, pred_date)
            close_firmlist.append(firm)
            close_incrlist.append(close_increase)
            close_predlist.append(close_pred)
            close_origlist.append(close_orig)
        result_df = pd.DataFrame({
            'Firm': close_firmlist,
            'Increase': close_incrlist,
            'Predicted_Close': close_predlist,
            'Original_Close': close_origlist,
            })
        result_df.to_csv(os.path.join(save_loc, f'{pred_date}.csv'), index=False, encoding='utf-8-sig')
        return
        
    def _get_firm_list(self, firm_table: str) -> list:
        firm_list = pd.read_csv(firm_table)[:20]
        firm_list = list(zip(firm_list.Stock_id, firm_list.Company_name))
        return [''.join(map(str,i)) for i in firm_list]
        
    def _prophet_inference(self, firm_name: str = '2330台積電', pred_date: str = '2020-03-10'):
        # --- call from checkpoint ---
        checkpoint_name = f"model_{self.hparams.method}_{firm_name}.ckpt"
        checkpoint_dir = os.path.join(self.hparams.checkpoint_path, checkpoint_name)
        checkpoint = torch.load(checkpoint_dir)
        regressor = checkpoint['model']
        
        # --- get dataset ---
        test_dataset = StockDataset(
            mode='Test',
            skip_preprocess=os.path.exists(os.path.join(self.hparams.stat_dir, f"{firm_name}_stats.ckpt")),
            save_type='prophet',
            time_step=40,
            moving_average=7,
            data_dir=self.hparams.data_dir,
            data_name=firm_name,
            stat_dir=self.hparams.stat_dir,
            )
        label_stats = test_dataset.label_stat
        test_data = test_dataset.data
        test_label = test_dataset.label
        test_index = test_dataset.data_index
        test_len = test_label.shape[0]
        
        # --- test ---
        future = pd.DataFrame({'ds': np.concatenate((regressor.history_dates, test_index))})
        pred_noise = regressor.predict(future)[-test_len:].yhat.values
        
        # --- noise to label ---
        pred_label = pred_noise + test_data
        
        # --- recovery ---
        pred_label = self.recovery(pred_label, **label_stats)
        test_data = self.recovery(test_data, **label_stats)
        
        # --- date to index ---
        test_index = {j:i for i,j in enumerate(test_index)}
        try:
            test_date = test_index[pd.Timestamp(pred_date)]
        except KeyError:
            print(f'Given date {pred_date} is not a valid date. (might be weekend or holiday)')
        
        # --- output ---
        pred_close = pred_label[test_date]
        orig_close = test_data[test_date]
        increase = pred_close / orig_close
        
        return increase, pred_close, orig_close