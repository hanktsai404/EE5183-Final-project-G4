# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:44:52 2021

@author: AustinHsu
"""

import os
from src.prophet_inference import StockInference

def inference_main(pred_date: str = '2020-03-10', save_loc: str = './'):
    
    # --- Inferencer ---
    Prophet_Inferencer = StockInference()
    
    # --- Get Weights ---
    Prophet_Inferencer.get_weights(pred_date=pred_date, save_loc=save_loc)
    save_location = os.path.join(save_loc, f'{pred_date}.csv')
    print(f'Inference completed. Please browse results from {save_location}')
    
    return

if __name__ == '__main__':
    pred_date = '2020-03-10' # Which date to be predicted
    save_loc = './' # Location for predicted dataframe to be stored
    inference_main(pred_date, save_loc)