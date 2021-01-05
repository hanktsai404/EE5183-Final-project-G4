# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:49:53 2021

@author: Austin Hsu
"""

import io
import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import Namespace
from PIL import Image
from torchvision.transforms import ToTensor
from fbprophet import Prophet
from .dataset import StockDataset
from .profiler import SimpleProfiler #
from .logger import TensorboardLogger #

class ProphetStockTrainer:
    
    def __init__(self, hparams: Namespace) -> None:
        self.hparams = hparams
        self.recovery = lambda x, min, max: x*(max-min)+min
        
        # --- random seed ---
        self._setup_seed(seed=self.hparams.seed)
        
        # --- mkdir ---
        os.makedirs(os.path.join(self.hparams.log_path, self.hparams.exp_name), exist_ok=True)
        
        # --- profiler ---
        self.profiler = SimpleProfiler(output_filename=os.path.join(self.hparams.log_path, self.hparams.exp_name, 'profile.txt'))
        
        # --- logger ---
        self.logger = TensorboardLogger(save_dir=os.path.join(self.hparams.log_path, self.hparams.exp_name))
    
    def _setup_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        
    def _plot_predict(self, *figures) -> io._io.BytesIO:
        plt.figure()
        for (fig, label) in figures:
            plt.plot(fig, label=label)
        plt.legend(loc='upper left')
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        return buf
    
    def _log_figure(self, fig_buffer: io._io.BytesIO, task_name: str, ma: int=1) -> None:
        image = Image.open(fig_buffer)
        image = ToTensor()(image)
        self.logger.log_image(f'Predicted Close MA({ma}): {task_name}', image, current_step=0)
    
    def fit(self) -> None:
        
        # --- training module ---
        Prophet_Regressor = Prophet()
        
        # --- training setup ---
        with self.profiler.profile('Prepare Train Batch'):
            train_dataset = StockDataset(
                mode='Train',
                skip_preprocess=os.path.exists(os.path.join(self.hparams.stat_dir, f"{self.hparams.data_name}_stats.ckpt")),
                save_type='prophet',
                time_step=self.hparams.time_step,
                moving_average=self.hparams.moving_average,
                train_percent=1.,
                data_dir=self.hparams.data_dir,
                data_name=self.hparams.data_name,
                stat_dir=self.hparams.stat_dir,
                )
            valid_dataset = StockDataset(
                mode='Valid',
                skip_preprocess=True,
                save_type='prophet',
                time_step=self.hparams.time_step,
                moving_average=self.hparams.moving_average,
                train_percent=1.,
                data_dir=self.hparams.data_dir,
                data_name=self.hparams.data_name,
                stat_dir=self.hparams.stat_dir,
                )
            train_data = np.concatenate((train_dataset.data, valid_dataset.data))
            train_label = np.concatenate((train_dataset.label, valid_dataset.label))
            train_index = np.concatenate((train_dataset.data_index, valid_dataset.data_index))
            train_noise = train_label-train_data # pred_data = train_data (close of last day) + trained_noise
            train_df = pd.DataFrame({
                'ds': train_index,
                'y': train_noise,
                })
            
        # --- trainer fit ---
        with self.profiler.profile('Fit Model'):
            Prophet_Regressor.fit(train_df)
        
        # --- test ---
        with self.profiler.profile('Test Loop'):
            avg_test_loss = self.test(Prophet_Regressor)
        self.test_loss = avg_test_loss
        
        # --- checkpoint ---
        with self.profiler.profile('Save Model'):
            print("Checkpointing model...")
            checkpoint_name = f"model_{self.hparams.method}_{self.hparams.data_name}.ckpt"
            checkpoint = {
                'model': Prophet_Regressor,
                'method': self.hparams.method,
                'firm': self.hparams.data_name,
                'moving_average': self.hparams.moving_average,
                'train_loss': 0,
                'valid_loss': 0,
                'test_loss': avg_test_loss,
                'epoch': 0,
                }
            torch.save(checkpoint, os.path.join(self.hparams.checkpoint_path, checkpoint_name))
        
        # --- checkpoint done ---
        print("Checkpoint completed.")

        # --- log hparams ---
        model_hparams = {'method': self.hparams.method, 'firm': self.hparams.data_name, 'moving_average': self.hparams.moving_average}
        self.logger.log_hparams(hparam_dict=model_hparams, metric_dict={i:checkpoint[i] for i in checkpoint.keys() if 'loss' in i})
        
        # --- close logger ---
        self.logger.close()
        
        # --- Profiler Summarization ---
        self.profiler.describe()
        
        return
    
    def test(self, regressor=None):
        
        # --- testing module ---
        if regressor is None:
            # --- call from checkpoint ---
            checkpoint_name = f"model_{self.hparams.method}_{self.hparams.data_name}.ckpt"
            checkpoint_dir = os.path.join(self.hparams.checkpoint_path, checkpoint_name)
            print(f"Using checkpointed model from {checkpoint_dir}")
            checkpoint = torch.load(checkpoint_dir)
            regressor = checkpoint['model']
        
        # --- testing setup ---
        test_dataset = StockDataset(
            mode='Test',
            skip_preprocess=os.path.exists(os.path.join(self.hparams.stat_dir, f"{self.hparams.data_name}_stats.ckpt")),
            save_type='prophet',
            time_step=self.hparams.time_step,
            moving_average=self.hparams.moving_average,
            data_dir=self.hparams.data_dir,
            data_name=self.hparams.data_name,
            stat_dir=self.hparams.stat_dir,
            )
        test_data = test_dataset.data
        test_label = test_dataset.label
        test_index = test_dataset.data_index
        test_len = test_label.shape[0]
        test_noise = test_label-test_data # pred_data = train_data (close of last day) + trained_noise
        
        # --- test ---
        future = pd.DataFrame({'ds': np.concatenate((regressor.history_dates, test_index))})
        pred_noise = regressor.predict(future)[-test_len:].yhat.values
        
        # --- get loss ---
        avg_test_loss = np.mean((test_noise-pred_noise)**2)
        
        return avg_test_loss
        
    def predict(self, regressor=None, task_name="default") -> None:
        # --- testing module ---
        if regressor is None:
            # --- call from checkpoint ---
            checkpoint_name = f"model_{self.hparams.method}_{self.hparams.data_name}.ckpt"
            checkpoint_dir = os.path.join(self.hparams.checkpoint_path, checkpoint_name)
            print(f"Using checkpointed model from {checkpoint_dir}")
            checkpoint = torch.load(checkpoint_dir)
            regressor = checkpoint['model']
        
        # --- testing setup ---
        test_dataset = StockDataset(
            mode='Test',
            skip_preprocess=os.path.exists(os.path.join(self.hparams.stat_dir, f"{self.hparams.data_name}_stats.ckpt")),
            save_type='prophet',
            time_step=self.hparams.time_step,
            moving_average=self.hparams.moving_average,
            data_dir=self.hparams.data_dir,
            data_name=self.hparams.data_name,
            stat_dir=self.hparams.stat_dir,
            )
        label_stats = test_dataset.label_stat
        test_data = test_dataset.data
        test_label = test_dataset.label
        test_index = test_dataset.data_index
        test_len = test_label.shape[0]
        test_noise = test_label-test_data # pred_data = train_data (close of last day) + trained_noise
        
        # --- test ---
        future = pd.DataFrame({'ds': np.concatenate((regressor.history_dates, test_index))})
        pred_noise = regressor.predict(future)[-test_len:].yhat.values

        # --- get loss ---
        avg_test_loss = np.mean((test_noise-pred_noise)**2)
        
        # --- noise to label ---
        pred_label = pred_noise + test_data
        
        # --- recovery ---
        pred_label = self.recovery(pred_label, **label_stats)
        test_label = self.recovery(test_label, **label_stats)
        
        # --- result ---
        tqdm_info = f"loss={avg_test_loss:8.6f}"
        print(f"Test result: {tqdm_info}")
        
        # --- plot prediction ---
        target_list = (test_label, 'real')
        pred_list = (pred_label, 'predict')
        image_buffer = self._plot_predict(target_list, pred_list)
        self._log_figure(fig_buffer=image_buffer, task_name=task_name+f" (loss={avg_test_loss})", ma=self.hparams.moving_average)
        
        # --- close logger ---
        self.logger.close()
        
        return