#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:16:17 2025
In this file, we define the hybridODE class which will serve as the parent class for 
training the hybrid models 

@author: akshaykudva
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import sys


#---------------------------------
# Parent Class with the ODE skeleton
#---------------------------------

class HybridODE(nn.Module):
    def __init__(self, 
                 param_net=None,
                 mode="inference",
                 mech_params = None,
                 mech_params_bounds = None,
                 fixed_params = None,
                 state_bounds = None,
                 input_bounds = None,
                 dnn_input_bounds = None, 
                 output_bounds = None, # Will be needed in some of the 
                 jitter = 1e-1):
        """
        Hybrid ODE can either combine the mechanistic model with machine learning components 
        using the following components:
            
        Args:
            param_net   : optional (nn.Module) data driven model for correction term/ prediction term
            
            mode        : optional (str) to treat which parameter to activate/freeze
            
            mech_params : dict of mechanistic parameters to fix (non-trainable) 
            
            mech_params_bounds: optional (torch tensor) these are bounds of the mechanistic parameters. Set to None for options other than "physical"
                                Note: Under certain cases, same as output_bounds, but this distinguishes between output_bounds
                                by setting a fixed value for parameters whereas output_bounds set data driven model for prediction
                                
            fixed_params: optional (dict) key-value pairs of parameters that do not change
            
            state_bounds: optional (torch.Tensor) Upper and lower bounds of the state (x)
            
            input_bounds: optional (torch.Tensor) Upper and lower bounds of the inputs (u)
            
            output_bounds: optional (torch.tensor) Upper and lower bounds for predictions of param_net (theta)               
                
            
            # TODO: Add scaling functions for each mode              
            
        
        
        mode options:
            "physical"  : train only mechanistic parameters
            "nn"        : train only neural correction
            "joint"     : train both mechanistic and NN parameters
            "inference" : freeze all parameters
            
            # TODO: Work on this mode later: Do not include in this skeleton
            Maybe this wont even be used in this step: Just used for training
            "mpc"       : treat input u as trainable variable (requires T)

        
        """
        super().__init__()
        
        #----------------------
        # Part 1: Set the parameters for training to default for training + stability
        # Typically, each of these are unique per trajectory. Need to think of cases where they are not
        #----------------------
        self.u_obs = None  # This is set for training
        self.t_obs = None  # This is set for training
        self.fixed_params = fixed_params # This is set for training 
        self.jitter = jitter # For numerical stability of the denominator
        self.test = False  # Can be used for debugging: likely to be legacy in IDE
        
        
        #----------------------
        # Part 2: Set attributes for the data driven and training modes + set bounds
        #----------------------

        self.param_net = param_net
        self.mode = mode
        self.input_bounds = input_bounds
        self.state_bounds = state_bounds
        self.output_bounds = output_bounds 
        self.mech_params_bounds = mech_params_bounds 

        #--------------------------------------
        if state_bounds is not None:
            self.x_lower = self.state_bounds[:, 0].view(1, 1, -1).clone()  # shape (1, 1, D)
            self.x_upper = self.state_bounds[:, 1].view(1, 1, -1).clone()

        if input_bounds is not None:
            self.u_lower = self.input_bounds[:, 0].view(1, 1, -1).clone()  # shape (1, 1, D)
            self.u_upper = self.input_bounds[:, 1].view(1, 1, -1).clone()
            
        if output_bounds is not None:
            self.theta_lower = self.output_bounds[:, 0].view(1, 1, -1).clone()  # shape (1, 1, D)
            self.theta_upper = self.output_bounds[:, 1].view(1, 1, -1).clone()
            self.theta_box = (self.theta_upper - self.theta_lower).clamp(min=1e-8)
            
            
        if state_bounds is not None and input_bounds is not None:            
            x_u_bounds = torch.cat((self.state_bounds, self.input_bounds), dim=0)
            self.x_u_lower = x_u_bounds[:, 0].view(1, 1, -1).clone()  # shape (1, 1, D)
            self.x_u_upper = x_u_bounds[:, 1].view(1, 1, -1).clone()
            self.x_u_box = (self.x_u_upper - self.x_u_lower).clamp(min=1e-8)
            
        if dnn_input_bounds is not None:            
            self.dnn_lower = dnn_input_bounds[:, 0].view(1, 1, -1).clone()  # shape (1, 1, D)
            self.dnn_upper = dnn_input_bounds[:, 1].view(1, 1, -1).clone()
            self.dnn_box = (self.dnn_upper - self.dnn_lower).clamp(min=1e-8)
            
        #--------------------------------------

        #----------------------
        # Part 3: Set attributes for the data driven and training modes + set bounds
        #----------------------
        
        # Set attributes for the mechanistic parameters: These can be fixed or set as Torch.nn depending on the mode:        
        self.mech_param_names = self.set_from_dict(mech_params)
        
        # Set attributes for the fixed parameters if they exist
        if type(fixed_params) is dict:
            _ = self.set_from_dict(fixed_params)   
            
            
        # Set mode after everything is set
        self._set_train_mode()
            

    def _set_train_mode(self):
        """Freeze/unfreeze depending on mode"""
        mode = self.mode

        # Reset gradients
        for name in self.mech_param_names:
            p = getattr(self, name)
            if isinstance(p, nn.Parameter):
                p.requires_grad = False
        if self.param_net is not None:
            for p in self.param_net.parameters():
                p.requires_grad = False

        # Set according to mode
        if mode == "physical":
            print("Training only mechanistic parameters!")
            for name in self.mech_param_names:
                p = getattr(self, name)
                if isinstance(p, nn.Parameter):
                    p.requires_grad = True

        elif mode == "nn":
            print("Training only NN correction!")
            if self.param_net is not None:
                for p in self.param_net.parameters():
                    p.requires_grad = True

        elif mode == "joint":
            print("Training both mechanistic and NN parameters!")
            for name in self.mech_param_names:
                p = getattr(self, name)
                if isinstance(p, nn.Parameter):
                    p.requires_grad = True
            if self.param_net is not None:
                for p in self.param_net.parameters():
                    p.requires_grad = True

        elif mode == "inference":
            print("Inference mode: all parameters frozen.")
            print("Ready for simulation!")

        else:
            print('Select a valid mode for training!')
            sys.exit(1)
    

    # Helper functions 1: For setting attributes
    def set_from_dict(self, d):
        """Set each key-value pair of a dictionary as an attribute.
           Wrap as nn.Parameter if mode allows training ('physical' or 'joint').
        """
        for k, v in d.items():
            if self.mode in ["physical", "joint"]:
                # Trainable parameter
                setattr(self, k, nn.Parameter(torch.tensor(v, dtype=torch.float32)))
            else:
                # Fixed tensor
                setattr(self, k, torch.tensor(v, dtype=torch.float32))
        return list(d.keys())
    
    
    def dict_from_set(self, list_name):
        """
        Given a list of attribute names, return a dictionary
        mapping each name to its current value (as float tensors).
        """
        d = {}
        for name in list_name:
            attr = getattr(self, name)
            if isinstance(attr, torch.nn.Parameter):
                d[name] = attr.detach().clone().cpu().numpy()
            elif torch.is_tensor(attr):
                d[name] = attr.clone().detach().cpu().numpy()
            else:
                d[name] = attr  # fallback for non-tensor attributes
        return d
    
    
    # Helper functions 2: Transforming data types and scaling
    def scale_mechanistic_parameters(self, list_vals, bounds):
        """
        Scales mechanistic parameters (stored as attributes) to [0, 1]
        using the provided lower and upper bounds.
    
        Parameters
        ----------
        bounds : torch.Tensor
            Tensor of shape (N, 2) where [:, 0] are lower bounds and [:, 1] are upper bounds.
    
        Returns
        -------
        scaled_dict : dict
            Dictionary with the same keys as `self.mech_param_names`,
            containing scaled parameter values.
        """
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        scaled_dict = {}
    
        for i, name in enumerate(list_vals):
            val = torch.as_tensor(getattr(self, name), dtype=torch.float32)
            scaled_val = (val - lb[i]) / (ub[i] - lb[i])
            scaled_dict[name] = scaled_val
            setattr(self, name + "_scaled", scaled_val)
    
        return scaled_dict        
    
    

    # Helper Functions 3: Scaling parameters
    def scale_u(self):
        return 0

    def scale_x(self):
        return 0
    
    def scale_x_u(self, x_u_tensor):
        """
        Typically the appended present state (x) and present inputs (u) serve as the 
        inputs to the 

        Returns
        -------
        int
            DESCRIPTION.

        """        
        scaled_x_u = (x_u_tensor - self.x_u_lower)/(self.x_u_box)
        
        return scaled_x_u
    
    
    def scale_dnn_inputs(self, dnn_tensor):
        """
        Typically the appended present state (x) and present inputs (u) serve as the 
        inputs to the 

        Returns
        -------
        int
            DESCRIPTION.

        """        
        scaled_dnn_input = (dnn_tensor - self.dnn_lower)/(self.dnn_box)
        
        return scaled_dnn_input
    

    def re_scale_dnn_outputs(self, scaled_output_tensor):
        """
        Rescale DNN outputs from [0, 1] or normalized units back to the
        original physical range using stored scaling parameters.
        
        Parameters
        ----------
        scaled_output_tensor : torch.Tensor
            Normalized output tensor from the DNN.
    
        Returns
        -------
        torch.Tensor
            Rescaled tensor in the original output range.
        """
        rescaled_output = scaled_output_tensor * self.theta_box + self.theta_lower
        return rescaled_output
    
    def predict_data_driven(self):
        return 0
    
    
    
    
if __name__ == "__main__":
    
    print("Testing the hybrid ode")
    
    learned_params = {"mu_max_s1": 0.62093, #(0, 1),
                    "mu_max_s2": 0.651371, #(0, 5),
                    "Ks1":2.111, #(0, 1),
                    "Ks2":3.3942, #(0, 0.1),
                    "a12": 2.1218, #(0, 1),
                    "a21": 0.7514, #(0, 10),
                    "alpha1": 0.8044, #(0, 0.1),
                    "alpha2": 6.9128e-6, #(0, 1e-2),
                    "beta": 0.6865, #(0, 1),
                    "Yxs1": 0.06724, #(0, 1),
                    "Yxs2": 1.3768, #(0, 1),
                    "K_ip": 0.02293, #(0, 0.1)
                    "Kd_max": 1.694, #(0.,1.)
                    "kd1": 0.94688,
                    "kd2":1.243333
                    }
    
    
    output_bounds = torch.tensor([[0., 5.], #mu_max_s1
                [0., 5.], #mu_max_s2
                [0., 5.], #Ks1
                [0., 5.], #Ks2
                [0., 2.], #a12
                [0., 5.], #a21
                [0., 5.], #alpha1
                [0., 5.], #alpha2
                [0., 5.],  #beta
                [0., 5.], #Yxs1
                [0., 5.], #Yxs2
                [0., 5.], #K_ip
                [0., 5.], #Kd
                [0., 5.], #kd1
                [0., 5.]]) #kd2
    
    a = HybridODE(mech_params=learned_params, mech_params_bounds=output_bounds)
    b = a.scale_mechanistic_parameters(a.mech_param_names, a.mech_params_bounds)


