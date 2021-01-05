# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:41:49 2020

@author: AustinHsu
"""

import argparse
import torch
import os
import pandas as pd
from src.dataset import StockDataset
from src.nn_solver import NNStockSolver
from src.nn_trainer import NNStockTrainer
from src.rf_trainer import RFStockTrainer
from src.xgb_trainer import XGBStockTrainer
from src.prophet_trainer import ProphetStockTrainer

def get_firm_list(firm_table: str, firm_num: int) -> list:
    firm_list = pd.read_csv(firm_table)[:firm_num]
    firm_list = list(zip(firm_list.Stock_id, firm_list.Company_name))
    return [''.join(map(str,i)) for i in firm_list]

def main(hparams):
    
    firm_list = get_firm_list(hparams.firm_table, hparams.firm_num)
    
    if hparams.train:
        trained_tag = True
        for firm in firm_list:
            print(f'Training on: {firm}')
            hparams.data_name = firm
            hparams.exp_name = f"{hparams.method}_{firm}"

            # --- NN Train ---
            if hparams.method.split('_')[0] in ["GRU", "RNN", "LSTM", "TCN"]:
                rnn_solver = NNStockSolver(hparams)
                rnn_trainer = NNStockTrainer(hparams)
                rnn_trainer.fit(rnn_solver)
                print(f'method: {hparams.method} | firm: {hparams.data_name} | test loss: {rnn_trainer.test_loss}')

            # --- RandomForest Model Train ---
            if hparams.method.split('_')[0] == "RandomForest":
                rf_trainer = RFStockTrainer(hparams)
                rf_trainer.fit()
                print(f'method: {hparams.method} | firm: {hparams.data_name} | test loss: {rf_trainer.test_loss}')

            # --- XGBoost Model Train ---
            if hparams.method.split('_')[0][:3] == "XGB":
                xgb_trainer = XGBStockTrainer(hparams)
                xgb_trainer.fit()
                print(f'method: {hparams.method} | firm: {hparams.data_name} | test loss: {xgb_trainer.test_loss}')

            # --- Facebook Prophet Train ---
            if hparams.method.split('_')[0] == "Prophet":
                prophet_trainer = ProphetStockTrainer(hparams)
                prophet_trainer.fit()
        
    # --- Plot Prediction (not available for ma prediction) ---
    if hparams.plot_pred or trained_tag:
        for firm in firm_list:
            print(f'Testing on: {firm}')
            hparams.data_name = firm
            hparams.exp_name = f"{hparams.method}_{firm}"

            if hparams.method.split('_')[0] in ["GRU", "RNN", "LSTM", "TCN"]:
                rnn_solver = NNStockSolver(hparams)
                rnn_trainer = NNStockTrainer(hparams)
                checkpoint_name = f"model_{hparams.method}_{hparams.data_name}.ckpt"
                checkpoint_dir = os.path.join(hparams.checkpoint_path, checkpoint_name)
                print(f"Using checkpointed model from {checkpoint_dir}")
                rnn_solver.load_from_checkpoint(checkpoint=checkpoint_dir, device=rnn_trainer.device)
                rnn_trainer.predict(rnn_solver, task_name=f"{hparams.method} | {hparams.data_name}")

            if hparams.method.split('_')[0] == "RandomForest":
                rf_trainer = RFStockTrainer(hparams)
                rf_trainer.predict(task_name=f"{hparams.method} | {hparams.data_name}")

            if hparams.method.split('_')[0][:3] == "XGB":
                xgb_trainer = XGBStockTrainer(hparams)
                xgb_trainer.predict(task_name=f"{hparams.method} | {hparams.data_name}")

            if hparams.method.split('_')[0] == "Prophet":
                prophet_trainer = ProphetStockTrainer(hparams)
                prophet_trainer.predict(task_name=f"{hparams.method} | {hparams.data_name}")
    
    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='final_project_g4')
    
    # --- NNStockTrainer Args ---
    parser.add_argument('--seed', help='Random seed.', default=7, type=int)
    parser.add_argument('--device_id', help='"cpu" or id of your gpu.', default="0", type=str)
    parser.add_argument('--log_path', help='Location to store loss/acc.', default="./log/", type=str)
    parser.add_argument('--exp_name', help='Experiment name.', default='model_default', type=str)
    parser.add_argument('--lr', help='Learning rate.', default=1e-3, type=float)
    parser.add_argument('--epoch', help='Training epochs.', default=10, type=int)
    parser.add_argument('--checkpoint_path', help='Location to store checkpoint.', default="./checkpoint/", type=str)
    
    # --- NNStockSolver Args ---
    """
    method tag:
        neural networks: {model}_{layers}_{hidden_units} | (ex. GRU_3_32) | valid model: [RNN, LSTM, GRU, TCN]
        random forests: RandomForest_{n_estimators} | (ex. RandomForest_100)
        xgboost: {model}_{n_estimators} | (ex. XGBRandomForest_100) | valid model: [XGB, XGBRandomForest]
        prophet: Prophet | (ex. Prophet)
    """
    parser.add_argument('--method', help='Selected {model}_{layers}_{hidden_units} for learning. (ex. GRU_3_32)', default='GRU_3_32', type=str)
    parser.add_argument('--feat_num', help='Number of features for input.', default=24, type=int)
    parser.add_argument('--time_step', help='How many days of data for predicting the next close value.', default=40, type=int)
    parser.add_argument('--moving_average', help='Period of MA prediction.', default=1, type=int)
    parser.add_argument('--stat_dir', help='Directory for storing statistical data.', default='./data_stat/', type=str)
    parser.add_argument('--train_percent', help='How many percent of data to be trained.', default=1., type=float)
    parser.add_argument('--data_dir', help='Directory for firm data.', default='./firms/', type=str)
    parser.add_argument('--batch_size', help='Batch size.', default=128, type=int)
    parser.add_argument('--num_workers', help='Number of workers.', default=1, type=int)
    parser.add_argument('--pin_memory', help='Pin memory (for label)', action='store_true')
    parser.add_argument('--shuffle', help='Shuffle dataset. (training only)', action='store_true')
    
    # --- Function Mode ---
    parser.add_argument('--train', help='Train model.', action='store_true')
    parser.add_argument('--plot_pred', help='Plot predictions.', action='store_true')

    # --- Addtional Tags ---
    parser.add_argument('--firm_table', help='Path to Market_Value_Table.csv', default='./data/Market_Value_Table.csv', type=str)
    parser.add_argument('--firm_num', help='Number of firms to be trained.', default=1, type=int)

    # --- parse args ---
    hparams = parser.parse_args()
    
    main(hparams)