#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 10:01:53 2025

Need to cle3an up the hybrid learner class

@author: akshaykudva
"""

import torch
import numpy as np
import pytorch_lightning as pl
import pdb
import sys
import torch.nn as nn
import sys


class HybridLearner_v1(pl.LightningModule):
    def __init__(self, node_model, hybrid_model, observed_idx, BIMT_model = None, observed_state = None, params_to_train=None, lr=5e-2,
                 l2_reg=0., param_reg_targets=None, optimizer_type = None, grad_clip_val=1.0,
                 max_epochs = None, swap_log = None, weight_factor = None, plot_log = None, lambda_BIMT = 0.001):
        """
        node_model: NeuralODE class for training
        hybrid_mnodel: (nn.Module) This consists of the physics components and the neural network whose parameters will be trained
        observed_idx: (list) Used to compare observed index with the predictions. Default will be all_idx (use linear interpolation by default)
        BIMT_model: (Optional/ BioMLP class) Enter the brain inspired moular training class and inherit the the functions that do the 3 steps from the paper
        observed_state: (list) Eneter the observed state numbers for training
        params_to_train:(DNN class.parameters()) Collects the DNN class parameters
        lr: (float) Learning rate for training
        l2_reg: (float = 0.) Regularization for training - ridge
        param_reg_targets: (nn.Parameters list) This is when we have an approximate idea about the parameters and their distribuion
        optimizer_type: (str): Decide the optimizer type- current options ("adamw","lbfgs","sgd")
        grad_clip: value for gradient clipping for stable training

        ## For BBIMT style training

        max_epochs: (int) This is typically set as None but adding this for adaptive leanring BIMT style
        swap_log: (int) This is used for

        """
        super().__init__()
        # Force CPU device
        self.device_type = torch.device("cpu")
        
        
        self.model = node_model # This is the neural ode model
        self.hybrid_model = hybrid_model # This goes back to the nn.Module to switch over the input sequence
        self.BIMT_model = BIMT_model
        self.loss_fn = nn.MSELoss()
        self.obs_state = observed_state
        self.lr = lr
        self.i = 0
        self.obs_idx = observed_idx
        self.l2_reg = l2_reg
        self.param_reg_targets = param_reg_targets or {}
        self.grad_clip_val = grad_clip_val
        self.optimizer_type = optimizer_type

        # default: train everything
        if params_to_train is None:
            self.params_to_train = list(self.model.func.parameters())
        else:
            self.params_to_train = list(params_to_train)

        if self.BIMT_model is not None:
            self.max_epochs = max_epochs
            self.swap_log = swap_log
            self.weight_factor = weight_factor
            self.plot_log = plot_log
            self.lambda_BIMT = lambda_BIMT
            self.bias_penalty = False

        if self.hybrid_model.state_bounds is not None:
            self.state_lower = self.hybrid_model.state_bounds[:, 0].view(1, 1, -1).clone()  # shape (1, 1, D)
            self.state_upper = self.hybrid_model.state_bounds[:, 1].view(1, 1, -1).clone()
            self.state_scaling = True
        else:
            self.state_scaling = False


    def forward(self, x0, t_eval):
        t_span = t_eval[0]  # assume all trajectories share same time steps
        _, traj_out = self.model(x0, t_span)
        return traj_out  # shape (B, T, D)

    def compute_regularization_loss(self):
        reg_loss = torch.tensor(0.0, device=self.device)
        if self.l2_reg > 0:
            for p in self.params_to_train:
                if p.requires_grad:
                    reg_loss += torch.sum(p ** 2)
            reg_loss = self.l2_reg * reg_loss
        return reg_loss

    def training_step(self, batch, batch_idx):
        if self.BIMT_model is None:
            return self.training_simple(batch, batch_idx)

        else:
            ################################################
            # Enter the BIMT style training here
            ###############################################
            return self.training_BIMT(batch, batch_idx)



    def training_BIMT(self, batch, batch_idx):
        x0, x_obs, t_obs, u_obs, fixed_params_obs = batch
    
        # Move to device
        x0 = x0.to(self.device)
        x_obs = x_obs.to(self.device)
        t_obs = t_obs.to(self.device)
        u_obs = u_obs.to(self.device)
        fixed_params_obs = fixed_params_obs.to(self.device)
    
        # Forward pass
        x0_squeezed = x0.squeeze(1)
        self.hybrid_model.u_obs = u_obs
        self.hybrid_model.t_obs = t_obs[0]  # assuming consistent t_obs
        self.hybrid_model.fixed_params = fixed_params_obs.squeeze(1)
    
        pred = self.forward(x0_squeezed, t_obs)  # (T, B, D)
    
        # Apply scaling once upfront
        if self.state_scaling:
            scale = (self.state_upper - self.state_lower).clamp(min=1e-8)
            pred = (pred - self.state_lower) / scale
            x_obs = (x_obs - self.state_lower) / scale
    
        # Replace NaNs before slicing
        pred = torch.nan_to_num(pred, nan=-1.0)
        x_obs = torch.nan_to_num(x_obs, nan=-1.0)
    
        ###############################
        # Handle observation indices
        ###############################
        if isinstance(self.obs_idx, list):
            # Different observation indices per trajectory
            loss_total = 0.0
            n_traj = len(self.obs_idx)
    
            for i in range(n_traj):
                obs_idx_i = self.obs_idx[i]
                pred_obs_i = pred[obs_idx_i, i, :][:, self.obs_state]        # (len(obs_idx_i), D)
                true_obs_i = x_obs[i, obs_idx_i, :][:, self.obs_state]      # (len(obs_idx_i), D)
                loss_total += self.loss_fn(pred_obs_i, true_obs_i)
    
            loss_main = loss_total / n_traj
    
        else:
            # Common observation indices for all trajectories
            pred_obs = pred[self.obs_idx, :, self.obs_state]           # (len(idx), B, D)
            true_obs = x_obs[:, self.obs_idx, self.obs_state].permute(1, 0, 2)
            loss_main = self.loss_fn(pred_obs, true_obs)       
        
        
        #pdb.set_trace()



        reg = self.BIMT_model.get_cc(bias_penalize=self.bias_penalty, weight_factor=self.weight_factor)

        # Enforce physics by adding physics constraint penalty
        violation_lower = torch.relu(-pred_obs)          # penalize below 0
        violation_upper = torch.relu(pred_obs - 1.0)     # penalize above 1
        loss_constraint = (violation_lower + violation_upper).mean()

        # Geometric loss
        geometric_loss = self.lambda_BIMT*reg
        geometric_loss = torch.nan_to_num(geometric_loss, nan=10)

        total_loss = loss_main + geometric_loss + loss_constraint
        
        
        if torch.isnan(total_loss):
            print("NaN in loss!")
            print('pause here')                   
        if any(torch.isnan(p).any() for p in self.hybrid_model.parameters()):
            print("NaN in parameters!")
            with torch.no_grad():
                for p in self.hybrid_model.parameters():
                    # Reset NaN parameters
                    # if torch.isnan(p).any():
                    #     print(f"⚠️ NaN in parameter, resetting to 0: {p.shape}")
                    #     p.data[torch.isnan(p.data)] = 0.0
                    # Reset NaN gradients
                    if p.grad is not None and torch.isnan(p.grad).any():
                        print(f"⚠️ NaN in gradient, setting to 0: {p.shape}")
                        p.grad[torch.isnan(p.grad)] = 0.0        
            
            
        if any(torch.isnan(p.grad).any() for p in self.hybrid_model.parameters() if p.grad is not None):
            print("NaN in gradients!")
        
        


        self.log("train_loss", total_loss)

        print(f"Loss={total_loss.item():.4e} (main={loss_main.item():.4e}, geometric={geometric_loss.item():.4e}, physics_const={loss_constraint.item():.4e})")

        return total_loss
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        # Replace NaNs in gradients before stepping
        with torch.no_grad():
            for p in self.hybrid_model.parameters():
                if p.grad is not None:
                    nan_mask = torch.isnan(p.grad)
                    if nan_mask.any():
                        print(f"⚠️ NaN in gradient, setting to 0: {p.shape}")
                        p.grad[nan_mask] = 0.0
    
        # Clip gradients if needed
        torch.nn.utils.clip_grad_norm_(self.hybrid_model.parameters(), max_norm=self.grad_clip_val)
    
        # Take the optimizer step
        optimizer.step(closure=optimizer_closure)


    def training_simple(self, batch, batch_idx):
        x0, x_obs, t_obs, u_obs, fixed_params_obs = batch
    
        # Move to device
        x0 = x0.to(self.device)
        x_obs = x_obs.to(self.device)
        t_obs = t_obs.to(self.device)
        u_obs = u_obs.to(self.device)
        fixed_params_obs = fixed_params_obs.to(self.device)
    
        # Forward pass
        x0_squeezed = x0.squeeze(1)
        self.hybrid_model.u_obs = u_obs
        self.hybrid_model.t_obs = t_obs[0]  # assuming consistent t_obs
        self.hybrid_model.fixed_params = fixed_params_obs.squeeze(1)
    
        pred = self.forward(x0_squeezed, t_obs)  # (T, B, D)
    
        # Apply scaling once upfront
        if self.state_scaling:
            scale = (self.state_upper - self.state_lower).clamp(min=1e-8)
            pred = (pred - self.state_lower) / scale
            x_obs = (x_obs - self.state_lower) / scale
    
        # Replace NaNs before slicing
        pred = torch.nan_to_num(pred, nan=-1.0)
        x_obs = torch.nan_to_num(x_obs, nan=-1.0)
    
        ###############################
        # Handle observation indices
        ###############################
        if isinstance(self.obs_idx, list):
            # Different observation indices per trajectory
            loss_total = 0.0
            n_traj = len(self.obs_idx)
            all_pred_obs = []  # for constraint penalty
            all_true_obs = []
    
            for i in range(n_traj):
                try:
                    obs_idx_i = self.obs_idx[i]
                    pred_obs_i = pred[obs_idx_i, i, :][:, self.obs_state]  # (len(obs_idx_i), D)
                    true_obs_i = x_obs[i, obs_idx_i, :][:, self.obs_state]  # (len(obs_idx_i), D)
                    loss_total += self.loss_fn(pred_obs_i, true_obs_i)
                    all_pred_obs.append(pred_obs_i)
                    all_true_obs.append(true_obs_i)
                except Exception as e:
                    print(f"Something messed up at traj {i}: {e}")
    
            loss_main = loss_total / n_traj
            # Combine for constraint computation
            pred_obs = torch.cat(all_pred_obs, dim=0) if len(all_pred_obs) > 0 else torch.zeros(1, len(self.obs_state)).to(self.device)
    
        else:
            # Common observation indices for all trajectories
            pred_obs = pred[self.obs_idx][:, :, self.obs_state]  # (len(idx), B, D)
            true_obs = x_obs[:, self.obs_idx, :].permute(1, 0, 2)
            
            # This happends when only one state is observed
            if pred_obs.dim() == 2:
                pred_obs = pred_obs.unsqueeze(-1)
            
            
            loss_main = self.loss_fn(pred_obs, true_obs)
    
        ###############################
        # Constraint penalty (ReLU)
        ###############################
        violation_lower = torch.relu(-pred_obs)         # penalize below 0
        violation_upper = torch.relu(pred_obs - 1.0)    # penalize above 1
        loss_constraint = (violation_lower + violation_upper).mean()
    
        ###############################
        # Regularization and total loss
        ###############################
        loss_reg = self.compute_regularization_loss()
        total_loss = loss_main + loss_reg + 10*loss_constraint
    
        ###############################
        # NaN checks and cleanup
        ###############################
        if torch.isnan(total_loss):
            print("NaN in loss!")
            sys.exit()
    
        for p in self.hybrid_model.parameters():
            if torch.isnan(p).any():
                print("NaN in parameters — resetting to 0.")
                with torch.no_grad():
                    p.data[torch.isnan(p.data)] = 0.0
            if p.grad is not None and torch.isnan(p.grad).any():
                print("NaN in gradients — resetting to 0.")
                p.grad[torch.isnan(p.grad)] = 0.0
    
        ###############################
        # Logging
        ###############################
        self.log("train_loss", total_loss)
        self.log("main_loss", loss_main)
        self.log("reg_loss", loss_reg)
        print(f"Loss={total_loss.item():.4e} (main={loss_main.item():.4e}, reg={loss_reg.item():.4e})")
    
        return total_loss

    def configure_optimizers(self):

        optimizer_type = self.optimizer_type
        #optimizer_type = optimizer_type.lower()

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.params_to_train,
                lr=self.lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=1e-5
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr * 5,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.2,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=1e4
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "monitor": "train_loss"
                }
            }

        elif optimizer_type == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.params_to_train,
                lr=self.lr,
                max_iter=20,
                history_size=10,
                line_search_fn='strong_wolfe'
            )

        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.params_to_train,
                lr=self.lr,
                momentum=0.9,  # optional
                weight_decay=0.0  # optional
            )

        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

    # # --- Lightning-native gradient clipping ---
    # def on_before_optimizer_step(self, optimizer):
    #     torch.nn.utils.clip_grad_norm_(self.params_to_train, max_norm=self.grad_clip_val)
        
    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_val)
        
        # Check for NaNs in parameters or gradients
        nan_found = False
        for p in self.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    nan_found = True
                    p.grad = torch.zeros_like(p.grad)  # zero out bad gradients