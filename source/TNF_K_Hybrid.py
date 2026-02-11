# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 12:03:37 2025

@author: kudva.7
"""

from hybrid_ode import HybridODE
from hybrid_model_learner_v1 import HybridLearner_v1
import torch
import sys
import pytorch_lightning as pl
import torch.nn as nn
from TNF_alpha_training_data import *


#[5,3,3,1]
#[1,2,1]
#[1,2,2,1] - LekyReLU

class ParamNet(nn.Module):
    def __init__(self, state_dim=5, hidden=[3,3], output_dim=1, output_bounds=None):
        super().__init__()

        if isinstance(hidden, int):
            hidden = [hidden]

        layers = []
        input_dim = state_dim

        # Hidden layers
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.LeakyReLU())
            input_dim = h

        # Output layer
        layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Register bounds as buffers (not trainable)
        if output_bounds is not None:
            self.register_buffer('output_min', output_bounds[:, 0].float())
            self.register_buffer('output_max', output_bounds[:, 1].float())
        else:
            self.output_min = None
            self.output_max = None

    def forward(self, x):
        y = self.net(x)
        #y = torch.sigmoid(y)

        # Apply bounds if specified
        if self.output_min is not None and self.output_max is not None:
            # Squash to [0,1] via sigmoid and then scale to bounds
            y = torch.sigmoid(y)
            y = self.output_min + (self.output_max - self.output_min) * y

        return y



#######################################
class NBFK_FPM_hybrid(HybridODE):
    """
    In this code, we will just do the first principles model, training / 
    
    
    
    """
    def __init__(self, 
                 param_net=None,
                 mode="inference",
                 mech_params = None,
                 mech_params_bounds = None,
                 fixed_params = None,
                 state_bounds = None,
                 input_bounds = None,
                 output_bounds = None,
                 jitter = 0):
        
        # Call parent constructor properly
        super().__init__(
            param_net=param_net,
            mode=mode,
            mech_params=mech_params,
            mech_params_bounds = mech_params_bounds,
            fixed_params=fixed_params,
            state_bounds=state_bounds,
            input_bounds=input_bounds,
            output_bounds=output_bounds,
            jitter = jitter
        )
        
        self.act = lambda x, bi: bi**2 / (bi**2 + x**2)
        self.inh = lambda x, ai: x**2 / (ai**2 + x**2)
    
    def forward(self, t, x, *args, **kwargs):
        
        """
        This is the first principles model exactly from what satchit (Kwon group did)
        Use this forward method as a boiler plate for future hybrid models
        
        """
        
        # -------------------------------------------------------------------------        
        # Step 1: Ensure that t is not nan, otherwise its stuck in an endless loop
        # Typically happens when denominator/ exponents blow up the ODE
        # -------------------------------------------------------------------------
        #t = t.detach()
        if torch.isnan(t).any():
            print("⚠️ Singularity: solution reached. Exiting...")
            sys.exit(1)
            
         # -------------------------------------------------------------------------        
         # Step 2: Fix the fixed parameter values - Fixed parameters is None
         # in this case study
         # -------------------------------------------------------------------------
        
        # try:
        #     if self.fixed_params is None:
        #         raise ValueError("fixed_params not set — set it first!")
    
        #     else:
        #         #pdb.set_trace()
        #         s1_feed = self.fixed_params[:,0]
        #         s2_feed = self.fixed_params[:,1]
        # except:
        #     print('Error with the fixed parameters!')
        #     sys.exit(1) 
            
        
        # -------------------------------------------------------------------------        
        # Step 3: Fix the u values, I can calling u as u_obs
        # -------------------------------------------------------------------------      
        
        try:
            if self.u_obs is None:
                raise ValueError("input values not set — set it first!")
    
            else:
                # Find the closest time step to applied 
                idx = torch.argmin(torch.abs(self.t_obs - t)).item()
                u = self.u_obs[:,idx]
                
        except:
            print('Error with the input parameters!')
            sys.exit(1)        
            
        # -------------------------------------------------------------------------        
        # Step 4: write the ODE equations
        # -------------------------------------------------------------------------  
        #theta = self.param_net(x[:,2].unsqueeze(-1)).squeeze(-1)
        theta = self.param_net(torch.cat([x, u], dim=1)).squeeze(-1)
        #theta = 0.48930943 * x[:,2] - 0.003768991 + 0.0039814943 / x[:,2]
        #theta = 0.467806101250684*x[:,2] + 0.0152467729012642

        
        # -------------------------------------------------------------------------        
        # Step 5: write the ODE equations
        # -------------------------------------------------------------------------        
        
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        # System dynamics
        dx1 = -x1 + 0.5 * (self.act(x3, self.b4) * self.inh(u.squeeze(-1), self.a1) + self.inh(x2, self.a3))
        dx2 = -x2 + self.inh(x1, self.a2) * self.act(x3, self.b3)
        dx3 = -x3 + self.act(x4, self.b5)*self.act(x2, self.b2)
        dx4 = -x4 + 0.5 * self.act(u.squeeze(-1), self.b1) + theta
        
        # Stack derivatives
        dx = torch.cat(
            [dx1[:, None], dx2[:, None], dx3[:, None], dx4[:, None]],
            dim=1
        )
        
        return dx
    
if __name__ == "__main__":  
    
    from torchdyn.models import NeuralODE
    import pickle
    from Helper_Fun import *
    
    plot_trained = True
    plot_test_data = True
    
    inference_mode = False
    
    
    param_net = ParamNet()
    
    
    ###################################
    if inference_mode: # save the model and then load
        mode = "inference"
        state_dict = torch.load("DNN_saved/Perfect_coeff_inputx3_state_2.pth")
        param_net.load_state_dict(state_dict)   
        
    else:
        mode = "nn"
    ####################################
    
    with open("NBF_K_data", "rb") as f:
        train_data = pickle.load(f)
        
        
    train_tensors = train_data['train_tensors']
    obs_idx = train_data['obs_idx']

    #obs_idx = torch.unique(torch.cat(obs_idx))
    observed_state_idx = [2] # Only 3rd state observed
    
    dataset = TrajectoryDataset(train_tensors)
    dataloader = data.DataLoader(dataset, batch_size=13, shuffle= False)
    
    
    
    # Guess values
    # mech_param_dict = {'a1': 0.5,
    #  'a2': 0.5,
    #  'a3': 0.5,
    #  'b1': 0.5,
    #  'b2': 0.5,
    #  'b3': 0.5,
    #  'b4': 0.5,
    #  'b5': 0.5
    #  }
    
    # Guess values - optimized    
    # mech_param_dict = {'a1': 0.5520068,
    #  'a2': 0.37427804,
    #  'a3': 0.41637018, 
    #  'b1': 0.6325303,
    #  'b2': 0.3795553, 
    #  'b3': 0.6210632,
    #  'b4': 0.67357457,
    #  'b5': 0.32503083}
    
    
    # Guess values - optimized all states observed
    # mech_param_dict = {'a1': 0.5316647,
    #  'a2': 0.5329925,
    #  'a3': 0.4895714,
    #  'b1': 1.470833,
    #  'b2': 0.4218885,
    #  'b3': 0.6398287, 
    #  'b4': 1.1941378,
    #  'b5': 0.46071804}
    

    # True values:
    mech_param_dict = {'a1': 0.6,
     'a2': 0.2,
     'a3': 0.2,
     'b1': 0.4,
     'b2': 0.7,
     'b3': 0.3, 
     'b4': 0.5,
     'b5': 0.4}
    
    
    # Joint values
    # {'a1': 0.23077323, 
    #  'a2': 0.28322357,
    #  'a3': 0.66844124,
    #  'b1': 0.3061371, 
    #  'b2': 1.1017553, 
    #  'b3': 0.33873066,
    #  'b4': 0.99117756,
    #  'b5': 0.37275356}
    
     
    # Fermentor FPM:
    nbf_ode = NBFK_FPM_hybrid(mech_params = mech_param_dict, mode = mode, param_net=param_net)
    
    hybrid_model = NeuralODE(nbf_ode, solver='euler', sensitivity='interpolated_adjoint',   #'adjoint' 'tsit5
                          rtol=1e-2, atol=1e-2).to(device)
    
    learner_hybrid = HybridLearner_v1(node_model = hybrid_model,
                    hybrid_model = nbf_ode,
                    observed_idx = obs_idx,
                    BIMT_model = None,
                    observed_state = observed_state_idx,
                    params_to_train = nbf_ode.parameters(),   # only physical params
                    optimizer_type = "adamw",
                    lr = 0.01)
    
    
    if not inference_mode:
        max_epochs = 500
        trainer2 = pl.Trainer(max_epochs=max_epochs, enable_checkpointing=False, logger = True, accelerator='cpu', log_every_n_steps=1)
        # Train this first
        trainer2.fit(learner_hybrid, dataloader)   
        
    # torch.save(param_net.state_dict(), "DNN_saved/Perfect_coeff_inputall_state_2.pth")
    
    # else:    
    #     nbf_ode2 = NBFK_FPM(mech_params = mech_param_dict_guess, mode = "inference")
    #     hybrid_model2 = NeuralODE(nbf_ode2, solver='rk4', sensitivity='interpolated_adjoint',   #'adjoint' 'tsit5
    #                           rtol=1e-3, atol=1e-3).to(device)
        
    #     learner_hybrid2 = HybridLearner_v1(node_model = hybrid_model2,
    #                     hybrid_model = nbf_ode2,
    #                     observed_idx = obs_idx,
    #                     BIMT_model = None,
    #                     observed_state = observed_state_idx,
    #                     params_to_train = nbf_ode.parameters(),   # only physical params
    #                     optimizer_type = "adamw",
    #                     lr = 0.001)
    
    predictor = {'Hybrid_optimal': learner_hybrid}
    
    if plot_test_data:
        visualize_using_plots_test(n = 25 ,base = base, x_label = ['x1','x2','x3','x4'] ,u_label = ['u'], predictors = predictor, obs_state_idx = [2], T = 30, seed = 300)
    