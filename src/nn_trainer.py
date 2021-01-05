# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:01:10 2020

@author: AustinHsu
"""

import torch
import random
import numpy as np
import os
import io
from torchvision.transforms import ToTensor
from argparse import Namespace
from typing import Union
from PIL import Image
import matplotlib.pyplot as plt
from .utils import AttributeDict #
from .profiler import SimpleProfiler #
from .logger import TensorboardLogger #

class NNStockTrainer:
    """
    hparams:
        seed
        device_id
        log_path
        exp_name
        lr
        epoch
        checkpoint_path
    """
    def __init__(self, hparams: Namespace) -> None:
        self.hparams = hparams
        
        # --- random seed ---
        self._setup_seed(seed=self.hparams.seed)
        
        # --- setup device ---
        self._setup_device(device_id=self.hparams.device_id)
        
        # --- mkdir ---
        os.makedirs(os.path.join(self.hparams.log_path, self.hparams.exp_name), exist_ok=True)

        # --- profiler ---
        self.profiler = SimpleProfiler(output_filename=os.path.join(self.hparams.log_path, self.hparams.exp_name, 'profile.txt'))
        
        # --- logger ---
        self.logger = TensorboardLogger(save_dir=os.path.join(self.hparams.log_path, self.hparams.exp_name))
    
    def _setup_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        
    def _setup_device(self, device_id: Union[int, str]) -> None:
        if device_id == 'cpu':
            self.device = torch.device('cpu')
        else:
            try:
                self.device = torch.device(f'cuda:{device_id}')
            except:
                print(f'Given device id cuda:{device_id} is not availabe, use cpu instead.')
                self.device = torch.device('cpu')
        
    def _free_gpu(self) -> None:
        torch.cuda.empty_cache()
        
    def _model_checkpoint(self, model: dict, model_name: str) -> None:
        torch.save(model, model_name)
        
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
    
    def fit(self, solver) -> None:
        
        # --- training setup ---
        optimizer, scheduler = solver.configure_optimizers(lr=self.hparams.lr)
        with self.profiler.profile('Prepare Train Batch'):
            train_dataloader = solver.train_dataloader()
        solver.model = solver.model.to(self.device)
        
        # --- training iteration ---
        self.train_step = 0
        self.best_result = {
            'loss': 1e8,
            }
        try:
            for epoch in self.profiler.profile_iterable(range(self.hparams.epoch), 'Train Epoch'):
                solver.model.train()
                loss = []
                for b_idx, batch in self.profiler.profile_iterable(enumerate(train_dataloader), 'Train Batch'):
                    
                    # --- zero grad ---
                    optimizer.zero_grad()
                    
                    # --- to device ---
                    with self.profiler.profile('To Device'):
                        batch[0] = batch[0].to(self.device)
                        batch[1] = batch[1].to(self.device, non_blocking=True)
                    
                    # --- training step ---
                    with self.profiler.profile('Train Forward'):
                        get_train_output = solver.training_step(batch=batch, batch_idx=b_idx)
                        get_train_output = AttributeDict(get_train_output)
                    
                    # --- backward ---
                    with self.profiler.profile('Train Backward'):
                        get_train_output.loss.backward()
                    
                    # --- update model ---
                    with self.profiler.profile('Optimizer Step'):
                        optimizer.step()
                    
                    # --- record label/pred/loss ---
                    loss.append(get_train_output.progress_bar.train_loss)
                    
                    # --- progress bar ---
                    tqdm_info = f"loss={get_train_output.progress_bar.train_loss:8.6f}"
                    print(f"Epoch:{epoch:2d}/{self.hparams.epoch} | Batch:{b_idx:4d} | {tqdm_info}", end='\r')

                self._free_gpu()
                print()
                
                # --- train result ---
                avg_train_loss = np.mean(loss)
    
                # --- result ---
                tqdm_info = f"loss={avg_train_loss:8.6f}"
                print(f"Train result: {tqdm_info}")
                
                # --- validation ---
                with self.profiler.profile('Validation Loop'):
                    avg_valid_loss = self.test(solver=solver, valid=True)
                self._free_gpu()

                # --- record history ---
                self.logger.log(f'loss/train', avg_train_loss, epoch)
                self.logger.log(f'loss/valid', avg_valid_loss, epoch)
                    
                # --- update scheduler ---
                if scheduler is not None:
                    with self.profiler.profile('Scheduler Step'):
                        scheduler.step(avg_valid_loss)
                
                # --- update best epoch ---
                if avg_valid_loss < self.best_result['loss']:
                    with self.profiler.profile('Save Model'):
                        # --- best record ---
                        self.best_result = {
                            'loss': avg_valid_loss,
                            }
                    
                        # --- checkpoint ---
                        print("Checkpointing model...")
                        checkpoint_name = f"model_{solver.hparams.method}_{solver.hparams.data_name}.ckpt"
                        checkpoint = {
                            'model': solver.model.state_dict(),
                            'method': solver.hparams.method,
                            'firm': solver.hparams.data_name,
                            'moving_average': solver.hparams.moving_average,
                            'train_loss': avg_train_loss,
                            'valid_loss': avg_valid_loss,
                            'epoch': epoch,
                            }
                        torch.save(checkpoint, os.path.join(self.hparams.checkpoint_path, checkpoint_name))
                        
                        # --- checkpoint done ---
                        print("Checkpoint completed.")
        except KeyboardInterrupt:
            self.profiler.describe()
            print('Exiting from training early.')

        # --- test ---
        with self.profiler.profile('Testing Loop'):
            # === recover from best model ===
            checkpoint_name = f"model_{self.hparams.method}_{self.hparams.data_name}.ckpt"
            checkpoint_dir = os.path.join(self.hparams.checkpoint_path, checkpoint_name)
            solver.load_from_checkpoint(checkpoint=checkpoint_dir, device=self.device)
            checkpoint = solver.checkpoint
            # === test ===
            avg_test_loss = self.test(solver=solver, valid=False)
        self.test_loss = avg_test_loss
        self._free_gpu()

        # --- record history ---
        self.logger.log(f'loss/test', avg_test_loss, epoch)

        # --- checkpoint ---
        with self.profiler.profile('Save Model'):
            checkpoint_name = f"model_{solver.hparams.method}_{solver.hparams.data_name}.ckpt"
            checkpoint['test_loss'] = avg_test_loss
            torch.save(checkpoint, os.path.join(self.hparams.checkpoint_path, checkpoint_name))

        # --- checkpoint done ---
        print("Checkpoint completed.")

        # --- log hparams ---
        model_hparams = {'method': solver.hparams.method, 'firm': solver.hparams.data_name, 'moving_average': solver.hparams.moving_average}
        self.logger.log_hparams(hparam_dict=model_hparams, metric_dict={i:checkpoint[i] for i in checkpoint.keys() if 'loss' in i})
        
        # --- close logger ---
        self.logger.close()
        
        # --- Profiler Summarization ---
        self.profiler.describe()
        
        return
    
    def test(self, solver, data_loader=None, valid=False):
        
        # --- testing setup ---
        if data_loader is None:
            if valid:
                test_dataloader = solver.valid_dataloader()
            else:
                test_dataloader = solver.test_dataloader()
        else:
            test_dataloader = data_loader
        solver.model = solver.model.to(self.device)
        solver.model.eval()
        
        # --- testing iteration ---
        loss = []
        for b_idx, batch in enumerate(test_dataloader):
            # --- to device ---
            batch[0] = batch[0].to(self.device)
            batch[1] = batch[1].to(self.device, non_blocking=True)
            
            # --- testing step ---
            get_test_output = solver.test_step(batch=batch, batch_idx=b_idx)
            get_test_output = AttributeDict(get_test_output)
            
            # --- record label/pred/loss ---
            loss.append(get_test_output.progress_bar.test_loss)
            
        # --- checkpoint summary ---
        avg_test_loss = np.mean(loss)
        
        # --- result ---
        tqdm_info = f"loss={avg_test_loss:8.6f}"
        print(f"Test result:  {tqdm_info}")
        
        return avg_test_loss
    
    def predict(self, solver, data_loader=None, task_name="default"):            
        # --- testing setup ---
        if data_loader is None:
            test_dataloader = solver.test_dataloader()
        else:
            test_dataloader = data_loader
        solver.model = solver.model.to(self.device)
        solver.model.eval()
        
        # --- testing iteration ---
        target_list = []
        pred_list = []
        loss = []
        for b_idx, batch in enumerate(test_dataloader):
            # --- to device ---
            batch[0] = batch[0].to(self.device)
            batch[1] = batch[1].to(self.device, non_blocking=True)
            
            # --- predict step ---
            get_pred = solver.get_pred(batch, 0)
            get_pred = AttributeDict(get_pred)
            
            # --- record pred ---
            target_list.append(get_pred.target)
            pred_list.append(get_pred.pred)
            
            # --- testing step ---
            get_test_output = solver.test_step(batch=batch, batch_idx=b_idx)
            get_test_output = AttributeDict(get_test_output)
            
            # --- record label/pred/loss ---
            loss.append(get_test_output.progress_bar.test_loss)
        
        # --- concatenate predictions ---
        target_list = np.concatenate(target_list)
        pred_list = np.concatenate(pred_list)

        # --- checkpoint summary ---
        avg_test_loss = np.mean(loss)
        
        # --- result ---
        tqdm_info = f"loss={avg_test_loss:8.6f}"
        print(f"Test result: {tqdm_info}")
        
        # --- plot prediction ---
        target_list = (target_list, 'real')
        pred_list = (pred_list, 'predict')
        image_buffer = self._plot_predict(target_list, pred_list)
        self._log_figure(fig_buffer=image_buffer, task_name=task_name+f" (loss={avg_test_loss})", ma=self.hparams.moving_average)
        
        # --- close logger ---
        self.logger.close()
        
        return 